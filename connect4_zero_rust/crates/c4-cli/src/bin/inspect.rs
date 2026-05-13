use anyhow::Result;
use c4_core::{Position, constants::ACTION_COUNT};
use clap::Parser;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = false)]
    legal_actions: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let position = Position::new();
    if args.legal_actions {
        let actions: Vec<_> = position
            .legal_actions()
            .into_iter()
            .map(|action| action.index())
            .collect();
        println!("legal_actions={actions:?}");
    } else {
        println!("empty board has {ACTION_COUNT} actions");
    }
    Ok(())
}
