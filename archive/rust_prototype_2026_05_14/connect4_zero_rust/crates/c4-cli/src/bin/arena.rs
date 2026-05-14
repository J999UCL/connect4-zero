use std::{path::PathBuf, time::Instant};

use anyhow::{Result, bail};
use c4_arena::ArenaSummary;
use c4_cli::{AnyEvaluator, make_search_config, parse_model_spec, random_opening};
use clap::Parser;
use rayon::prelude::*;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    candidate: Option<PathBuf>,
    #[arg(long)]
    baseline: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    uniform: bool,
    #[arg(long, default_value_t = 64)]
    games: usize,
    #[arg(long, default_value_t = 128)]
    simulations_per_move: usize,
    #[arg(long, default_value_t = 6)]
    opening_plies: usize,
    #[arg(long, default_value_t = true)]
    paired_openings: bool,
    #[arg(long, default_value_t = 1)]
    workers: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    json_out: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.paired_openings && args.games % 2 != 0 {
        bail!("paired openings require an even number of games");
    }
    let candidate_model = parse_model_spec(args.candidate.clone(), args.uniform)?;
    let baseline_model = if args.uniform {
        parse_model_spec(None, true)?
    } else {
        parse_model_spec(args.baseline.clone(), false)?
    };
    let started = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers)
        .build()?;
    let chunks = make_chunks(args.games, args.workers.max(1));
    let results: Vec<_> = pool.install(|| {
        chunks
            .into_par_iter()
            .flat_map(|range| {
                let mut candidate =
                    AnyEvaluator::from_spec(&candidate_model).expect("create candidate evaluator");
                let mut baseline =
                    AnyEvaluator::from_spec(&baseline_model).expect("create baseline evaluator");
                range
                    .map(|game_index| {
                        let opening_seed = if args.paired_openings {
                            args.seed.wrapping_add((game_index / 2) as u64)
                        } else {
                            args.seed.wrapping_add(game_index as u64)
                        };
                        let opening = random_opening(opening_seed, args.opening_plies);
                        let candidate_starts = game_index % 2 == 0;
                        let config = make_search_config(
                            args.simulations_per_move,
                            false,
                            0.0,
                            args.seed.wrapping_add(100_000 + game_index as u64),
                        );
                        c4_cli::play_arena_game(
                            &mut candidate,
                            &mut baseline,
                            config,
                            candidate_starts,
                            &opening,
                            args.seed.wrapping_add(200_000 + game_index as u64),
                        )
                        .expect("arena game failed")
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });
    let mut summary = ArenaSummary::default();
    let mut total_plies = 0_usize;
    for result in &results {
        total_plies += result.plies;
        match result.winner {
            1 => summary.candidate_wins += 1,
            -1 => summary.baseline_wins += 1,
            _ => summary.draws += 1,
        }
    }
    summary.unique_openings = if args.paired_openings {
        args.games / 2
    } else {
        args.games
    };
    let elapsed = started.elapsed().as_secs_f64();
    let score =
        (summary.candidate_wins as f64 + 0.5 * summary.draws as f64) / args.games.max(1) as f64;
    println!("games={}", args.games);
    println!("candidate_wins={}", summary.candidate_wins);
    println!("baseline_wins={}", summary.baseline_wins);
    println!("draws={}", summary.draws);
    println!("candidate_score_rate={score:.4}");
    println!("unique_openings={}", summary.unique_openings);
    println!(
        "avg_plies={:.3}",
        total_plies as f64 / args.games.max(1) as f64
    );
    println!("elapsed_seconds={elapsed:.3}");
    println!("games_per_sec={:.3}", args.games as f64 / elapsed.max(1e-9));
    if let Some(path) = args.json_out {
        std::fs::write(path, serde_json::to_vec_pretty(&summary)?)?;
    }
    Ok(())
}

fn make_chunks(games: usize, workers: usize) -> Vec<std::ops::Range<usize>> {
    let chunk_size = games.div_ceil(workers).max(1);
    (0..games)
        .step_by(chunk_size)
        .map(|start| start..(start + chunk_size).min(games))
        .collect()
}
