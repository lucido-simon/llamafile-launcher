use anyhow::{Context, Result};
use clap::Parser;
use futures_util::stream::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info, trace};
use std::{
    cmp::min,
    fs::{File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::exit,
};

mod docker;

#[cfg(target_family = "unix")]
use std::os::unix::fs::OpenOptionsExt;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(flatten)]
    args: ModelSource,

    #[arg(
        short = 'd',
        long,
        env,
        help = "Models directory. This is where the models are downloaded to, and checked for before downloading"
    )]
    model_dir: Option<String>,

    #[arg(
        long,
        default_value = "false",
        env,
        help = "Only download the model, don't run llama"
    )]
    only_download: bool,

    #[arg(short, long, env, help = "Path to llamafile-server")]
    llamafile_server_path: Option<String>,

    #[arg(
        short = 'D',
        long,
        env,
        help = "Download llamafile-server if it doesn't exist",
        default_value = "true"
    )]
    llamafile_server_download: bool,

    #[arg(
        short = 'b',
        long,
        env,
        help = "Build docker image with the model",
        default_value = "false"
    )]
    docker_build: bool,

    #[arg(
        long,
        env,
        help = "Image name for the docker image",
        requires("docker_build")
    )]
    image_name: Option<String>,
}

#[derive(Debug, clap::Args)]
#[group(required = true, multiple = true)]
struct ModelSource {
    #[arg(
        short = 'm',
        long,
        requires("hf_file_name"),
        env,
        help = "Hugging face repository"
    )]
    hf_model_name: Option<String>,

    #[arg(
        short = 'n',
        long,
        requires("hf_model_name"),
        env,
        help = "Hugging face file name, within the repository"
    )]
    hf_file_name: Option<String>,

    #[arg(
        short = 'f',
        long,
        conflicts_with("hf_file_name"),
        conflicts_with("hf_model_name"),
        conflicts_with("file_url"),
        env,
        help = "Local model file path"
    )]
    file_path: Option<String>,

    #[arg(
        short = 'u',
        long,
        conflicts_with("hf_file_name"),
        conflicts_with("hf_model_name"),
        conflicts_with("file_path"),
        env,
        help = "Model URL"
    )]
    file_url: Option<String>,
}

struct Runner {
    llama_path: String,
}

impl Runner {
    fn new(llama_path: String) -> Result<Self> {
        if !Path::new(&llama_path).exists() {
            return Err(anyhow::anyhow!(format!(
                "Llama path '{}' does not exist",
                llama_path
            )));
        }

        Ok(Self { llama_path })
    }

    async fn run(&self, model_path: &Path) -> Result<()> {
        tokio::process::Command::new(&self.llama_path)
            .arg("-m")
            .arg(model_path)
            .spawn()?
            .wait()
            .await?;

        Ok(())
    }
}

struct Models {
    base_dir: PathBuf,
}

impl Models {
    fn new(basedir: Option<String>) -> Result<Self> {
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
        })
    }

    fn exists(&self, filename: &str) -> bool {
        std::path::Path::new(&self.base_dir).join(filename).exists()
    }

    fn exists_hf(&self, model: &str, filename: &str) -> bool {
        self.exists(format!("{}/{}", model, filename).as_str())
    }

    async fn get_hf_model(&self, model: &str, filename: &str) -> Result<PathBuf> {
        if !self.exists_hf(model, filename) {
            info!("Downloading {}/{}", model, filename);
            let mut model_dir = self.base_dir.clone();
            model_dir.push(model);
            std::fs::create_dir_all(&model_dir)?;
            model_dir.push(filename);
            from_hf(model, filename, &model_dir).await?;
        } else {
            info!("Found {}/{} locally", model, filename);
        }

        Ok(self.base_dir.join(model).join(filename))
    }

    async fn get_model(&self, url: &str) -> Result<PathBuf> {
        let filename = url
            .split('/')
            .last()
            .context("Couldn't extract filename from URL")?;

        if !self.exists(filename) {
            info!("Downloading {} to {}", url, filename);
            let filename = self.base_dir.join(filename);
            from_url(url, &filename, false).await?;
        } else {
            info!("Found {} locally", filename);
        }

        Ok(self.base_dir.join(filename))
    }
}

#[tokio::main]
async fn main() {
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    let args = Args::parse();

    debug!("Args: {:?}", args);

    let mut model_path: Option<PathBuf> = None;

    if let Some(file_path) = args.args.file_path {
        let file_path = PathBuf::from(file_path);
        if !file_path.exists() {
            crash(&format!(
                "File path '{}' does not exist",
                file_path.display()
            ));
        }
        model_path = Some(file_path);
    } else {
        info!("Initializing models directory");
        let files = match Models::new(args.model_dir.clone()) {
            Ok(files) => files,
            Err(e) => crash(&format!("Failed to initialize models directory: {}", e)),
        };

        if let Some(model) = args.args.hf_model_name {
            if let Some(filename) = args.args.hf_file_name {
                let path = match files.get_hf_model(&model, &filename).await {
                    Ok(path) => path,
                    Err(e) => crash(&format!("Failed to get model: {}", e)),
                };

                model_path = Some(path);
            }
        } else if let Some(url) = args.args.file_url {
            let path = match files.get_model(&url).await {
                Ok(path) => path,
                Err(e) => crash(&format!("Failed to get model: {}", e)),
            };

            model_path = Some(path);
        }
    }

    info!("Located model");
    let model_path = model_path.unwrap();
    debug!("Model path: {:?}", model_path);

    let llama_path = args
        .llamafile_server_path
        .clone()
        .unwrap_or("./llamafile-server".to_string());

    let llama_path = Path::new(&llama_path);
    let exists = llama_path.exists();
    if !exists && args.llamafile_server_download {
        info!("Downloading llamafile-server");
        if let Err(e) = download_llamafile_release(llama_path, "llamafile-server").await {
            crash(&format!("Failed to download llamafile-server: {}", e));
        }
    } else if !exists {
        crash(&format!(
            "Llamafile-server path '{}' does not exist. Set the -D flag to download it",
            llama_path.display()
        ));
    }
    info!("Using llamafile-server at {}", llama_path.display());

    if args.docker_build {
        info!("Building docker image");
        let docker = match docker::Docker::new() {
            Ok(docker) => docker,
            Err(e) => crash(&format!("Failed to initialize docker: {}", e)),
        };

        let image_name = args.image_name.unwrap_or(
            model_path
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string(),
        );

        match docker
            .build_image(&image_name, vec![&model_path], llama_path)
            .await
        {
            Ok(_) => info!("Built docker image"),
            Err(e) => crash(&format!("Failed to build docker image: {}", e)),
        }
    }

    if args.only_download {
        return;
    }

    info!("Running the model");

    let runner = match Runner::new(
        args.llamafile_server_path
            .unwrap_or("./llamafile-server".to_string()),
    ) {
        Ok(runner) => runner,
        Err(e) => crash(&format!("Failed to initialize llama: {}", e)),
    };

    match runner.run(&model_path).await {
        Ok(_) => info!("Llama exited successfully"),
        Err(e) => crash(&format!("Llama exited with error: {}", e)),
    };
}

async fn from_url(url: &str, filename: &Path, set_executable: bool) -> Result<()> {
    let client = reqwest::Client::new();

    // Reqwest setup
    let res = client
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

    // Indicatif setup
    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})").unwrap()
        .progress_chars("#>-"));

    pb.set_message(format!("Downloading {}", &url));

    let mut options = OpenOptions::new();
    options.write(true).create(true);

    #[cfg(target_family = "unix")]
    if set_executable {
        options.mode(0o755);
    }

    options.open(filename).or(Err(anyhow::anyhow!(format!(
        "Failed to open file '{}'",
        &filename.display()
    ))))?;

    // download chunks
    let mut file = File::create(filename).or(Err(anyhow::anyhow!(format!(
        "Failed to create file '{}'",
        &filename.display()
    ))))?;

    let mut downloaded: u64 = 0;
    let mut stream = res.bytes_stream();

    while let Some(item) = stream.next().await {
        let chunk = item.or(Err(anyhow::anyhow!(format!(
            "Error while downloading file"
        ))))?;
        file.write_all(&chunk)
            .or(Err(anyhow::anyhow!(format!("Error while writing to file"))))?;
        let new = min(downloaded + (chunk.len() as u64), total_size);
        downloaded = new;
        pb.set_position(new);
    }

    pb.finish_with_message(format!("Downloaded {} to {}", &url, &filename.display()));
    Ok(())
}

async fn from_hf(model: &str, filename: &str, destination_file: &Path) -> Result<()> {
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}?download=true",
        model, filename
    );

    from_url(&url, destination_file, false).await
}

async fn download_llamafile_release(
    file_path: &Path,
    release_starts_with: &str,
) -> Result<PathBuf> {
    let client = reqwest::Client::new();
    let res = client
        .get("https://api.github.com/repos/Mozilla-Ocho/llamafile/releases/latest")
        .header("User-Agent", "reqwest")
        .send()
        .await?;

    let release = res.json::<GithubRelease>().await?;

    trace!("{:?}", &release);

    let download_url = release
        .assets
        .into_iter()
        .find(|asset| asset.name.starts_with(release_starts_with))
        .context(format!("Couldn't find {} asset", release_starts_with))?
        .browser_download_url;

    debug!("Downloading {} from {}", release_starts_with, &download_url);

    from_url(&download_url, file_path, true).await?;

    Ok(PathBuf::from(file_path))
}

fn crash(msg: &str) -> ! {
    error!("{}", msg);
    error!("Exiting");
    exit(1);
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
