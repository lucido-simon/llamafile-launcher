use anyhow::Result;
use flate2::{write::GzEncoder, Compression};
use futures_util::StreamExt;
use log::{debug, error, info};
use tar::Header;

pub(crate) struct Docker {
    docker: bollard::Docker,
}

impl Docker {
    pub fn new() -> Result<Self> {
        let docker = bollard::Docker::connect_with_local_defaults()?;
        Ok(Self { docker })
    }

    pub async fn build_image(
        &self,
        image_name: &str,
        model_path: Vec<&str>,
        llama_path: &str,
    ) -> Result<()> {
        info!("Building image: {}", image_name);
        let dockerfile = self.dockerfile(&model_path);
        debug!("Dockerfile: {}", dockerfile);
        info!("Building tarball.. This may take a while.");
        let tarball = self.tarball(dockerfile, model_path, llama_path)?;

        let image_options = bollard::image::BuildImageOptions {
            dockerfile: "Dockerfile",
            t: image_name,
            rm: true,
            ..Default::default()
        };

        info!("Building image.. This may take a while.");
        let mut build_image = self
            .docker
            .build_image(image_options, None, Some(tarball.into()));

        while let Some(msg) = build_image.next().await {
            if let Ok(msg) = msg {
                info!("{:?}", msg);
            } else {
                error!("{:?}", msg);
            }
        }

        Ok(())
    }

    fn dockerfile(&self, models_path: &[&str]) -> String {
        let mut dockerfile = String::from(
            r#"
FROM debian:bullseye-slim AS final
RUN addgroup --gid 1000 user
RUN adduser --uid 1000 --gid 1000 --disabled-password --gecos "" user
USER user
WORKDIR /usr/src/app
COPY /llamafile-server ./llamafile-server
"#,
        );

        for (i, _) in models_path.iter().enumerate() {
            dockerfile.push_str(&format!("COPY /model-{} ./model-{}\n", i, i));
        }

        dockerfile.push_str(
            r#"
# Expose 8080 port.
EXPOSE 8080

# Set entrypoint.
ENTRYPOINT ["/bin/sh", "/usr/src/app/llamafile-server", "-m", "/usr/src/app/model-0", "--host", "0.0.0.0"]
"#,
        );

        dockerfile
    }

    fn tarball(
        &self,
        dockerfile: String,
        models_path: Vec<&str>,
        llama_path: &str,
    ) -> Result<Vec<u8>> {
        let enc = GzEncoder::new(Vec::new(), Compression::new(0));

        let mut tarball = tar::Builder::new(enc);

        debug!("Appending llamafile-server..");
        tarball.append_path_with_name(llama_path, "./llamafile-server")?;

        debug!("Appending Dockerfile..");
        let mut header = Header::new_gnu();
        header.set_path("./Dockerfile")?;
        header.set_size(dockerfile.len() as u64);
        header.set_mode(0o755);
        header.set_cksum();

        tarball.append_data(&mut header, "./Dockerfile", dockerfile.as_bytes())?;

        for (i, model_path) in models_path.iter().enumerate() {
            debug!("Appending model-{} from {}..", i, model_path);
            tarball.append_path_with_name(model_path, &format!("./model-{}", i))?;
        }

        let tarball = tarball.into_inner()?;

        Ok(tarball.finish()?)
    }
}
