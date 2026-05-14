use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use anyhow::Result;
use c4_cli::{AnyEvaluator, make_search_config, parse_model_spec};
use c4_data::{DatasetMetadata, SelfPlaySample, write_dataset};
use clap::Parser;
use rayon::prelude::*;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    uniform: bool,
    #[arg(long, default_value_t = 4)]
    games: usize,
    #[arg(long, default_value_t = 128)]
    simulations_per_move: usize,
    #[arg(long, default_value_t = 12)]
    temperature_cutoff_ply: u8,
    #[arg(long, default_value_t = 1)]
    workers: usize,
    #[arg(long, default_value_t = 16_384)]
    samples_per_shard: usize,
    #[arg(long, default_value = "rust-selfplay-out")]
    out: PathBuf,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let model = parse_model_spec(args.model.clone(), args.uniform)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers)
        .build()?;
    let started = Instant::now();
    let chunks = make_chunks(args.games, args.workers.max(1));
    let completed_games = AtomicUsize::new(0);
    let completed_samples = AtomicUsize::new(0);
    let completed_plies = AtomicUsize::new(0);
    let games: Vec<_> = pool.install(|| {
        chunks
            .into_par_iter()
            .flat_map(|range| {
                let model = model.clone();
                let mut evaluator = AnyEvaluator::from_spec(&model).expect("create evaluator");
                range
                    .map(move |game_index| {
                        let config = make_search_config(
                            args.simulations_per_move,
                            true,
                            1.0,
                            args.seed.wrapping_add(game_index as u64),
                        );
                        c4_cli::generate_selfplay_game_with_evaluator(
                            &mut evaluator,
                            config,
                            args.seed.wrapping_add(10_000 + game_index as u64),
                            args.temperature_cutoff_ply,
                        )
                    })
                    .inspect(|game| {
                        let games_done = completed_games.fetch_add(1, Ordering::Relaxed) + 1;
                        let samples_done =
                            completed_samples.fetch_add(game.samples.len(), Ordering::Relaxed)
                                + game.samples.len();
                        let plies_done =
                            completed_plies.fetch_add(game.plies, Ordering::Relaxed) + game.plies;
                        if games_done == args.games || games_done % 100 == 0 {
                            let elapsed = started.elapsed().as_secs_f64().max(1e-9);
                            eprintln!(
                                "progress games={}/{} samples={} avg_plies={:.2} games_per_sec={:.3} samples_per_sec={:.3}",
                                games_done,
                                args.games,
                                samples_done,
                                plies_done as f64 / games_done.max(1) as f64,
                                games_done as f64 / elapsed,
                                samples_done as f64 / elapsed,
                            );
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });

    let mut samples: Vec<SelfPlaySample> = Vec::new();
    let mut total_plies = 0_usize;
    let mut total_nodes = 0_usize;
    let mut total_leaf_evals = 0_usize;
    for game in games {
        total_plies += game.plies;
        total_nodes += game.nodes;
        total_leaf_evals += game.leaf_evals;
        samples.extend(game.samples);
    }
    let metadata = DatasetMetadata {
        generator: "c4-selfplay".to_string(),
        model_path: args.model.as_ref().map(|path| c4_cli::display_path(path)),
    };
    let manifest = write_dataset(&args.out, &samples, args.samples_per_shard, metadata)?;
    let elapsed = started.elapsed().as_secs_f64();
    println!("selfplay_out={}", args.out.display());
    println!("games={}", args.games);
    println!("samples={}", samples.len());
    println!("shards={}", manifest.shards.len());
    println!("elapsed_seconds={elapsed:.3}");
    println!("games_per_sec={:.3}", args.games as f64 / elapsed.max(1e-9));
    println!(
        "samples_per_sec={:.3}",
        samples.len() as f64 / elapsed.max(1e-9)
    );
    println!(
        "avg_plies={:.3}",
        total_plies as f64 / args.games.max(1) as f64
    );
    println!(
        "avg_nodes_per_game={:.1}",
        total_nodes as f64 / args.games.max(1) as f64
    );
    println!(
        "avg_leaf_evals_per_game={:.1}",
        total_leaf_evals as f64 / args.games.max(1) as f64
    );
    Ok(())
}

fn make_chunks(games: usize, workers: usize) -> Vec<std::ops::Range<usize>> {
    let chunk_size = games.div_ceil(workers).max(1);
    (0..games)
        .step_by(chunk_size)
        .map(|start| start..(start + chunk_size).min(games))
        .collect()
}
