use anyhow::Result;
use clap::Parser;
use log::{debug, error, info};
use std::{
    path::{Path, PathBuf},
    process::exit,
};

mod docker;
mod http_client;
mod llamafile_builder;
mod models;

use crate::{llamafile_builder::LlamafileBuilder, models::Models};

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(flatten)]
    args: ModelSource,

    #[clap(flatten)]
    build_args: BuildArgs,

    #[arg(
        short = 'd',
        long,
        env,
        help = "Models directory. This is where the models are downloaded to, and checked for before downloading"
    )]
    model_dir: Option<String>,

    #[arg(
        short = 'e',
        long,
        default_value = "false",
        env,
        help = "Execute the model"
    )]
    execute: bool,

    #[arg(short, long, env, help = "Path to llamafile-server")]
    llamafile_server_path: Option<String>,

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
#[group(required = false, multiple = true)]
struct BuildArgs {
    #[arg(
        short = 'B',
        long,
        env,
        help = "Build llamafile with embedded model",
        default_value = "false",
        requires("llamafile_output")
    )]
    build_llamafile: bool,

    #[arg(
        short = 'o',
        long,
        env,
        help = "Output file of llamafile build",
        requires("build_llamafile"),
        conflicts_with("llamafile_output_dir")
    )]
    llamafile_output: Option<String>,

    #[arg(
        long = "output-dir",
        env,
        help = "Output folder of all llamafile builds",
        requires("build_llamafile"),
        conflicts_with("llamafile_output")
    )]
    llamafile_output_dir: Option<String>,

    #[arg(long, env, help = "Path to zipalign")]
    zipalign_path: Option<String>,
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
        let mut files = match Models::new(args.model_dir.clone()) {
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
    if !exists {
        info!("Downloading llamafile-server");
        let mut llamafile_builder = match LlamafileBuilder::new(None, None, None).await {
            Ok(llamafile_builder) => llamafile_builder,
            Err(e) => crash(&format!("Failed to initialize llamafile builder: {}", e)),
        };

        let download = llamafile_builder
            .download_llamafile_github_release_into(
                llamafile_builder::GithubReleaseAsset::LlamafileServer,
                llama_path,
            )
            .await;

        if let Err(e) = download {
            crash(&format!("Failed to download llamafile-server: {}", e));
        }
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

    if args.build_args.build_llamafile {
        info!("Building llamafile");
        let mut llamafile_builder = match LlamafileBuilder::new(
            args.build_args
                .llamafile_output_dir
                .as_ref()
                .map(From::from),
            args.llamafile_server_path.as_ref().map(From::from),
            args.build_args.zipalign_path.as_ref().map(From::from),
        )
        .await
        {
            Ok(llamafile_builder) => llamafile_builder,
            Err(e) => crash(&format!("Failed to initialize llamafile builder: {}", e)),
        };

        let path: Option<PathBuf> = args.build_args.llamafile_output.as_ref().map(From::from);

        match llamafile_builder.build(&[&model_path], path).await {
            Ok(_) => info!("Built llamafile"),
            Err(e) => crash(&format!("Failed to build llamafile: {}", e)),
        }
    }

    if args.execute {
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
}

fn crash(msg: &str) -> ! {
    error!("{}", msg);
    error!("Exiting");
    exit(1);
}
