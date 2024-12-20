use amt::{AmtParams, CreateMode, PowerTau};
use anyhow::{bail, Result};
use tracing::Level;

fn parse_param() -> Result<(usize, usize, Option<String>)> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        bail!(
            "Usage: {} <amt-depth> <verify-depth> [<power_tau_dir>]",
            args[0]
        );
    }

    let path = if args.len() == 4 {
        Some(args[3].parse()?)
    } else {
        None
    };

    Ok((args[1].parse()?, args[2].parse()?, path))
}

fn main() {
    let (depth, verify_depth, ptau_dir) = match parse_param() {
        Ok(x) => x,
        Err(e) => {
            eprintln!("Cannot parse input: {:?}", e);
            std::process::exit(1);
        }
    };

    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_target(false)
        .init();

    let create_mode = ptau_dir.is_none();
    let dir = ptau_dir.unwrap_or("./params/test".into());
    let pp = PowerTau::from_dir(&dir, depth, create_mode);

    AmtParams::from_dir_mont(&dir, depth, verify_depth, CreateMode::AmtOnly, Some(&pp));
}
