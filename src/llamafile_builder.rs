use anyhow::{Context, Result};
use log::{debug, info, warn};
use std::{
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

pub struct LlamafileBuilder {
    temp_path: PathBuf,
    output_dir: Option<PathBuf>,
    llamafile_path: PathBuf,
    zipalign_path: PathBuf,
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
            });
        }

        info!("Using existing llamafile repo at {}", temp_path.display());
        let llamafile_path = temp_path.join("llamafile-server");
        let zipalign_path = temp_path.join("zipalign");

        if !llamafile_path.exists() {
            warn!("llamafile-server not found in {}", temp_path.display());
            info!("Downloading llamafile..");
            super::download_llamafile_release(llamafile_path.as_path(), "llamafile-server").await?;
        }

        if !zipalign_path.exists() {
            warn!("zipalign not found in {}", temp_path.display());
            info!("Downloading zipalign..");
            super::download_llamafile_release(zipalign_path.as_path(), "zipalign").await?;
        }

        Ok(LlamafileBuilder {
            temp_path,
            output_dir,
            llamafile_path,
            zipalign_path,
        })
    }

    pub async fn build(&self, models: &[&Path], output: Option<PathBuf>) -> Result<()> {
        info!("Building models..");
        debug!("Models: {:?}", models);

        let mut llamafile = OpenOptions::new().read(true).open(&self.llamafile_path)?;

        #[cfg(unix)]
        debug!("Setting permissions on llamafile to 0o755");
        llamafile.set_permissions(std::fs::Permissions::from_mode(0o755))?;

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
