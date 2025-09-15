use anyhow::Result;
use clap::Parser;
use melior::Context;
use simt_step::frontends::{hlsl::import_hlsl_to_mlir, cuda::import_cuda_to_mlir};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value="hlsl", value_parser=["hlsl","cuda"])]
    frontend: String,
    #[arg(long)]
    input: String,
    #[arg(long, default_value = "-")]
    output: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let ctx = Context::new();
    let src = std::fs::read_to_string(&args.input)?;
    let module = match args.frontend.as_str() {
        "hlsl" => import_hlsl_to_mlir(&ctx, &src)?,
        _      => import_cuda_to_mlir(&ctx, &src)?,
    };
    if args.output == "-" { println!("{}", module); }
    else { std::fs::write(&args.output, format!("{}", module))?; }
    Ok(())
}
