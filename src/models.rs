use anyhow::{Context, Result};
use log::{debug, info};
use std::path::{Path, PathBuf};

use crate::http_client::HttpClient;

pub struct Models {
    base_dir: PathBuf,
    http_client: HttpClient,
}

impl Models {
    pub fn new(basedir: Option<String>) -> Result<Self> {
        debug!("Creating LocalFiles");
        let basedir = basedir.unwrap_or_else(|| "./models/".to_string());
        let basedir = Path::new(&basedir);
        if !basedir.exists() {
            info!("Creating models directory at {}", basedir.display());
            std::fs::create_dir_all(basedir)?;
        } else {
            info!(
                "Using models directory at {}",
                basedir.canonicalize()?.display()
            )
        }

        Ok(Self {
            base_dir: PathBuf::from(basedir),
            http_client: HttpClient::new(),
        })
    }

    fn exists(&self, filename: &str) -> bool {
        std::path::Path::new(&self.base_dir).join(filename).exists()
    }

    fn exists_hf(&self, model: &str, filename: &str) -> bool {
        self.exists(format!("{}/{}", model, filename).as_str())
    }

    pub async fn get_hf_model(&mut self, model: &str, filename: &str) -> Result<PathBuf> {
        if !self.exists_hf(model, filename) {
            info!("Downloading {}/{}", model, filename);
            let mut model_dir = self.base_dir.clone();
            model_dir.push(model);
            std::fs::create_dir_all(&model_dir)?;
            model_dir.push(filename);
            let url = format!(
                "https://huggingface.co/{}/resolve/main/{}?download=true",
                model, filename
            );

            self.http_client
                .download_to(&url, &model_dir, false)
                .await?
        } else {
            info!("Found {}/{} locally", model, filename);
        }

        Ok(self.base_dir.join(model).join(filename))
    }

    pub async fn get_model(&mut self, url: &str) -> Result<PathBuf> {
        let filename = url
            .split('/')
            .last()
            .context("Couldn't extract filename from URL")?;

        if !self.exists(filename) {
            info!("Downloading {} to {}", url, filename);
            let filename = self.base_dir.join(filename);
            self.http_client.download_to(url, &filename, false).await?;
        } else {
            info!("Found {} locally", filename);
        }

        Ok(self.base_dir.join(filename))
    }
}
