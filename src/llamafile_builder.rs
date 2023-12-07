use anyhow::{Context, Result};
use log::{debug, info, warn};
use std::{
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

use crate::http_client::HttpClient;

const LLAMAFILE_GITHUB_RELEASE_URL: &str =
    "https://api.github.com/repos/Mozilla-Ocho/llamafile/releases/latest";

pub struct LlamafileBuilder {
    temp_path: PathBuf,
    output_dir: Option<PathBuf>,
    llamafile_path: PathBuf,
    zipalign_path: PathBuf,
    http_client: HttpClient,
}

impl LlamafileBuilder {
    pub async fn new(
        output_dir: Option<PathBuf>,
        llamafile_path: Option<PathBuf>,
        zipalign_path: Option<PathBuf>,
    ) -> Result<LlamafileBuilder> {
        let temp_path = tempfile::tempdir()?.into_path();

        if let Some(output_dir) = output_dir.as_ref() {
            if !output_dir.exists() {
                info!("Creating output directory");
                std::fs::create_dir_all(output_dir)?;
            }
            info!("Using output directory {}", output_dir.display());
        }

        if zipalign_path
            .as_ref()
            .is_some_and(|p| p.exists() && p.is_file())
            && llamafile_path
                .as_ref()
                .is_some_and(|p| p.exists() && p.is_file())
        {
            let llamafile_path = llamafile_path.unwrap();
            let zipalign_path = zipalign_path.unwrap();
            info!("Using existing llamafile at {}", llamafile_path.display());
            info!("Using existing zipalign at {}", zipalign_path.display());
            return Ok(LlamafileBuilder {
                temp_path,
                output_dir,
                llamafile_path,
                zipalign_path,
                http_client: HttpClient::new(),
            });
        }

        info!("Using existing llamafile repo at {}", temp_path.display());
        let llamafile_path = temp_path.join("llamafile-server");
        let zipalign_path = temp_path.join("zipalign");

        Ok(LlamafileBuilder {
            temp_path,
            output_dir,
            llamafile_path,
            zipalign_path,
            http_client: HttpClient::new(),
        })
    }

    pub async fn build(&mut self, models: &[&Path], output: Option<PathBuf>) -> Result<()> {
        info!("Building models..");
        debug!("Models: {:?}", models);

        if !self.llamafile_path.exists() {
            warn!("llamafile-server not found in {}", self.temp_path.display());
            info!("Downloading llamafile..");
            self.download_llamafile_github_release(GithubReleaseAsset::LlamafileServer)
                .await?;
        }

        let mut llamafile = OpenOptions::new().read(true).open(&self.llamafile_path)?;

        let output = self.get_output_path(models[0], output)?;
        debug!("Building into: {}", output.display());

        let mut output_llamafile = OpenOptions::new();

        #[cfg(unix)]
        debug!("Setting permissions on output to 0o755");
        output_llamafile.mode(0o755);

        let mut output_llamafile = output_llamafile
            .write(true)
            .create_new(true)
            .open(&output)?;

        std::io::copy(&mut llamafile, &mut output_llamafile)?;
        output_llamafile.sync_all()?;
        drop(output_llamafile);

        let args_file_path = self.temp_path.join(".args");
        let mut args_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args_file_path)
            .context("Failed to create .args file")?;

        args_file.write_all(
            format!(
                r#"
-m
{}
--host
0.0.0.0
"#,
                models[0].file_name().unwrap().to_str().unwrap()
            )
            .as_bytes(),
        )?;
        args_file.sync_all()?;
        drop(args_file);

        if !self.zipalign_path.exists() {
            warn!("zipalign not found in {}", self.temp_path.display());
            info!("Downloading zipalign..");
            self.download_llamafile_github_release(GithubReleaseAsset::Zipalign)
                .await?;
        }

        #[cfg(unix)]
        debug!("Setting permissions on zipalign to 0o755");
        let zipalign = OpenOptions::new().read(true).open(&self.zipalign_path)?;
        zipalign.set_permissions(std::fs::Permissions::from_mode(0o755))?;

        info!("Zipaligning models..");
        debug!("Zipalign: {}", self.zipalign_path.display());
        debug!("Llamafile: {}", output.display());
        tokio::process::Command::new(self.zipalign_path.as_path())
            .arg("-j0")
            .arg(output)
            .arg(models[0])
            .arg(args_file_path)
            .spawn()?
            .wait()
            .await?;

        info!("Finished building models");

        Ok(())
    }

    async fn download_llamafile_github_release(
        &mut self,
        github_release: GithubReleaseAsset,
    ) -> Result<()> {
        let file_path = match github_release {
            GithubReleaseAsset::LlamafileServer => &self.llamafile_path,
            GithubReleaseAsset::Zipalign => &self.zipalign_path,
        };

        self.download_llamafile_github_release_into(github_release, &file_path.clone())
            .await
    }

    pub async fn download_llamafile_github_release_into(
        &mut self,
        github_release: GithubReleaseAsset,
        path: &Path,
    ) -> Result<()> {
        if path.exists() {
            anyhow::bail!("{} already exists", path.display());
        }

        let release: GithubRelease = self
            .http_client
            .get(LLAMAFILE_GITHUB_RELEASE_URL)
            .await
            .context("Failed to get latest llamafile release")?;

        let asset = release
            .assets
            .iter()
            .find(|a| a.name.starts_with(github_release.to_string().as_str()))
            .context("Failed to find asset in release")?;

        info!("Downloading {}..", asset.name);
        self.http_client
            .download_to(&asset.browser_download_url, path, false)
            .await?;

        Ok(())
    }

    fn get_output_path(&self, model_path: &Path, output_path: Option<PathBuf>) -> Result<PathBuf> {
        if let Some(output_path) = output_path {
            Ok(output_path)
        } else if let Some(output_dir) = &self.output_dir {
            Ok(output_dir.join(model_path.file_stem().unwrap()))
        } else {
            anyhow::bail!("Neither output_dir nor output_path were specified");
        }
    }
}

#[derive(serde::Deserialize, Debug)]
struct GithubAsset {
    name: String,
    browser_download_url: String,
}

#[derive(serde::Deserialize, Debug)]
struct GithubRelease {
    assets: Vec<GithubAsset>,
}

pub enum GithubReleaseAsset {
    LlamafileServer,
    Zipalign,
}

impl ToString for GithubReleaseAsset {
    fn to_string(&self) -> String {
        match self {
            GithubReleaseAsset::LlamafileServer => "llamafile-server",
            GithubReleaseAsset::Zipalign => "zipalign",
        }
        .to_string()
    }
}
