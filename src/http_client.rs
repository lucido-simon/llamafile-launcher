use std::{fs::OpenOptions, io::Write, path::Path};

use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::{Client, Error};
use serde::de::DeserializeOwned;

#[derive(Debug)]
pub struct HttpClient {
    client: Client,
}

impl HttpClient {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }

    pub async fn download(
        &mut self,
        url: &str,
    ) -> anyhow::Result<(u64, impl Stream<Item = Result<Bytes, Error>>)> {
        let res = self
            .client
            .get(url)
            .header("User-Agent", "reqwest")
            .send()
            .await
            .or(Err(anyhow::anyhow!(format!(
                "Failed to GET from '{}'",
                &url
            ))))?;

        let total_size = res.content_length().ok_or(anyhow::anyhow!(format!(
            "Failed to get content length from '{}'",
            &url
        )))?;

        Ok((total_size, res.bytes_stream()))
    }

    pub async fn download_to(
        &mut self,
        url: &str,
        path: &Path,
        set_executable: bool,
    ) -> anyhow::Result<()> {
        let (total_size, mut stream) = self.download(url).await?;

        let pb = ProgressBar::new(total_size);
        pb.set_style(ProgressStyle::default_bar()
        .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})").unwrap()
        .progress_chars("#>-"));

        pb.set_message(format!("Downloading {}", &url));

        let mut options = OpenOptions::new();
        options.write(true).create(true);

        #[cfg(target_family = "unix")]
        if set_executable {
            std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o755);
        }

        options.open(path).or(Err(anyhow::anyhow!(format!(
            "Failed to open file '{}'",
            &path.display()
        ))))?;

        let mut file = std::fs::File::create(path)?;
        let mut downloaded = 0;

        while let Some(item) = stream.next().await {
            let chunk = item.or(Err(anyhow::anyhow!(format!(
                "Error while downloading file"
            ))))?;
            file.write_all(&chunk)
                .or(Err(anyhow::anyhow!(format!("Error while writing to file"))))?;
            let new = std::cmp::min(downloaded + (chunk.len() as u64), total_size);
            downloaded = new;
            pb.set_position(new);
        }

        pb.finish_with_message(format!("Downloaded {} to {}", &url, &path.display()));
        Ok(())
    }

    pub async fn get<T: DeserializeOwned>(&mut self, url: &str) -> anyhow::Result<T> {
        let res = self
            .client
            .get(url)
            .header("User-Agent", "reqwest")
            .send()
            .await
            .or(Err(anyhow::anyhow!(format!(
                "Failed to GET from '{}'",
                &url
            ))))?;

        let body = res.json::<T>().await.or(Err(anyhow::anyhow!(format!(
            "Failed to parse JSON from '{}'",
            &url
        ))))?;

        Ok(body)
    }
}
