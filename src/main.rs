use anyhow::{Context, Result};
use clap::Parser;
use futures_util::stream::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info};
use std::{
    cmp::min,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    process::exit,
};

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
            from_hf(model, filename, model_dir.to_str().unwrap()).await?;
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
            let filename = filename.to_str().unwrap();
            from_url(url, filename).await?;
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
    debug!("Model path: {:?}", model_path);

    if args.only_download {
        return;
    }

    info!("Running llama");
    let runner = match Runner::new(
        args.llamafile_server_path
            .unwrap_or("./llamafile-server".to_string()),
    ) {
        Ok(runner) => runner,
        Err(e) => crash(&format!("Failed to initialize llama: {}", e)),
    };

    match runner.run(model_path.as_ref().unwrap()).await {
        Ok(_) => info!("Llama exited successfully"),
        Err(e) => crash(&format!("Llama exited with error: {}", e)),
    };
}

async fn from_url(url: &str, filename: &str) -> Result<()> {
    let client = reqwest::Client::new();

    // Reqwest setup
    let res = client
        .get(url)
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

    // download chunks
    let mut file = File::create(filename).or(Err(anyhow::anyhow!(format!(
        "Failed to create file '{}'",
        &filename
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

    pb.finish_with_message(format!("Downloaded {} to {}", &url, &filename));
    Ok(())
}

async fn from_hf(model: &str, filename: &str, destination_file: &str) -> Result<()> {
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}?download=true",
        model, filename
    );

    from_url(&url, destination_file).await
}

fn crash(msg: &str) -> ! {
    error!("{}", msg);
    error!("Exiting");
    exit(1);
}
