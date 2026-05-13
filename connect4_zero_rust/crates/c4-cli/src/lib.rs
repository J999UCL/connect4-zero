use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use c4_core::{Action, GameOutcome, Position, constants::ACTION_COUNT};
use c4_data::{SelfPlaySample, legal_mask_from_position};
#[cfg(feature = "onnx")]
use c4_infer::onnx::OnnxCpuEvaluator;
use c4_infer::{Evaluator, UniformEvaluator};
use c4_search::{PuctConfig, PuctSearch, SearchResult, SearchTree};
use rand::{Rng, SeedableRng, rngs::SmallRng};

#[derive(Clone, Debug)]
pub enum ModelSpec {
    Uniform,
    Onnx(PathBuf),
}

pub enum AnyEvaluator {
    Uniform(UniformEvaluator),
    #[cfg(feature = "onnx")]
    Onnx(OnnxCpuEvaluator),
}

impl AnyEvaluator {
    pub fn from_spec(spec: &ModelSpec) -> Result<Self> {
        match spec {
            ModelSpec::Uniform => Ok(Self::Uniform(UniformEvaluator)),
            ModelSpec::Onnx(path) => {
                #[cfg(feature = "onnx")]
                {
                    Ok(Self::Onnx(OnnxCpuEvaluator::new(path)?))
                }
                #[cfg(not(feature = "onnx"))]
                {
                    let _ = path;
                    bail!("c4-cli was built without ONNX support")
                }
            }
        }
    }
}

impl Evaluator for AnyEvaluator {
    fn evaluate(&mut self, positions: &[Position]) -> Vec<c4_infer::Evaluation> {
        match self {
            Self::Uniform(evaluator) => evaluator.evaluate(positions),
            #[cfg(feature = "onnx")]
            Self::Onnx(evaluator) => evaluator.evaluate(positions),
        }
    }
}

pub fn parse_model_spec(model: Option<PathBuf>, uniform: bool) -> Result<ModelSpec> {
    match (model, uniform) {
        (Some(path), false) => Ok(ModelSpec::Onnx(path)),
        (None, true) => Ok(ModelSpec::Uniform),
        (Some(_), true) => bail!("choose either --model or --uniform, not both"),
        (None, false) => bail!("provide --model PATH or --uniform"),
    }
}

pub fn make_search_config(
    simulations_per_move: usize,
    add_root_noise: bool,
    policy_temperature: f32,
    seed: u64,
) -> PuctConfig {
    PuctConfig {
        simulations_per_move,
        add_root_noise,
        policy_temperature,
        seed,
        ..PuctConfig::default()
    }
}

pub struct GeneratedGame {
    pub samples: Vec<SelfPlaySample>,
    pub plies: usize,
    pub outcome: Option<GameOutcome>,
    pub nodes: usize,
    pub leaf_evals: usize,
}

struct PendingSample {
    position: Position,
    policy: [f32; ACTION_COUNT],
    visit_counts: [u32; ACTION_COUNT],
    q_values: [f32; ACTION_COUNT],
    legal_mask: [bool; ACTION_COUNT],
    action: u8,
    ply: u8,
    player: i8,
}

pub fn generate_selfplay_game(
    mut evaluator: AnyEvaluator,
    config: PuctConfig,
    seed: u64,
    temperature_cutoff_ply: u8,
) -> GeneratedGame {
    generate_selfplay_game_with_evaluator(&mut evaluator, config, seed, temperature_cutoff_ply)
}

pub fn generate_selfplay_game_with_evaluator(
    evaluator: &mut AnyEvaluator,
    config: PuctConfig,
    seed: u64,
    temperature_cutoff_ply: u8,
) -> GeneratedGame {
    let mut rng = SmallRng::seed_from_u64(seed);
    struct Borrowed<'a>(&'a mut AnyEvaluator);
    impl Evaluator for Borrowed<'_> {
        fn evaluate(&mut self, positions: &[Position]) -> Vec<c4_infer::Evaluation> {
            self.0.evaluate(positions)
        }
    }
    let mut search = PuctSearch::new(Borrowed(evaluator), config);
    let mut position = Position::new();
    let mut tree = SearchTree::new(position);
    let mut player = 1_i8;
    let mut pending = Vec::new();
    let mut total_nodes = 0_usize;
    let mut total_leaf_evals = 0_usize;

    while !position.is_terminal() {
        let result = search.search_tree(&mut tree);
        total_nodes += result.diagnostics.nodes;
        total_leaf_evals += result.diagnostics.leaf_evals;
        let temperature = if position.ply < temperature_cutoff_ply {
            1.0
        } else {
            0.0
        };
        let action_index =
            choose_action(&result.policy, position.legal_mask(), temperature, &mut rng);
        let action = Action::from_index(action_index).expect("chosen action is valid");
        pending.push(PendingSample {
            position,
            policy: result.policy,
            visit_counts: result.visits,
            q_values: result.q_values,
            legal_mask: legal_mask_from_position(&position),
            action: action_index as u8,
            ply: position.ply,
            player,
        });

        position = position
            .play(action)
            .expect("chosen action is legal")
            .position;
        if !tree.advance_to_child(action) {
            tree = SearchTree::new(position);
        }
        player = -player;
    }

    let winner = match position.outcome {
        Some(GameOutcome::CurrentPlayerLoss) => -player,
        Some(GameOutcome::Draw) | None => 0,
    };
    let samples = pending
        .into_iter()
        .map(|sample| SelfPlaySample {
            position: sample.position,
            policy: sample.policy,
            value: if winner == 0 {
                0
            } else if sample.player == winner {
                1
            } else {
                -1
            },
            visit_counts: sample.visit_counts,
            q_values: sample.q_values,
            legal_mask: sample.legal_mask,
            action: sample.action,
            ply: sample.ply,
        })
        .collect();

    GeneratedGame {
        samples,
        plies: position.ply as usize,
        outcome: position.outcome,
        nodes: total_nodes,
        leaf_evals: total_leaf_evals,
    }
}

pub fn choose_action(
    policy: &[f32; ACTION_COUNT],
    legal_mask: u16,
    temperature: f32,
    rng: &mut SmallRng,
) -> usize {
    if temperature <= 0.0 {
        return argmax_legal(policy, legal_mask);
    }
    let mut weights = [0.0_f32; ACTION_COUNT];
    let mut total = 0.0_f32;
    for action in 0..ACTION_COUNT {
        if legal_mask & (1_u16 << action) == 0 {
            continue;
        }
        let weight = policy[action].max(0.0).powf(1.0 / temperature);
        weights[action] = weight;
        total += weight;
    }
    if total <= 0.0 {
        return first_legal(legal_mask).expect("no legal actions");
    }
    let mut cursor = rng.random::<f32>() * total;
    for (action, weight) in weights.iter().enumerate() {
        cursor -= *weight;
        if cursor <= 0.0 {
            return action;
        }
    }
    argmax_legal(policy, legal_mask)
}

pub fn argmax_legal(values: &[f32; ACTION_COUNT], legal_mask: u16) -> usize {
    let mut best = first_legal(legal_mask).expect("no legal actions");
    let mut best_value = values[best];
    for (action, &value) in values.iter().enumerate().skip(best + 1) {
        if legal_mask & (1_u16 << action) != 0 && value > best_value {
            best = action;
            best_value = value;
        }
    }
    best
}

pub fn first_legal(legal_mask: u16) -> Option<usize> {
    (0..ACTION_COUNT).find(|&action| legal_mask & (1_u16 << action) != 0)
}

#[derive(Clone, Debug)]
pub struct ArenaGameResult {
    pub winner: i8,
    pub plies: usize,
}

pub fn play_arena_game(
    candidate: &mut AnyEvaluator,
    baseline: &mut AnyEvaluator,
    config: PuctConfig,
    candidate_starts: bool,
    opening: &[usize],
    seed: u64,
) -> Result<ArenaGameResult> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut position = Position::new();
    let mut player = if candidate_starts { 1_i8 } else { -1_i8 };

    for &action_index in opening {
        if position.is_terminal() {
            break;
        }
        let action = Action::from_index(action_index).context("opening action out of range")?;
        if !position.is_legal(action) {
            bail!("opening action {action_index} is illegal");
        }
        position = position.play(action)?.position;
        player = -player;
    }

    while !position.is_terminal() {
        let result = if player == 1 {
            search_with_existing_evaluator(candidate, position, config.clone())
        } else {
            search_with_existing_evaluator(baseline, position, config.clone())
        };
        let action_index = choose_action(&result.policy, position.legal_mask(), 0.0, &mut rng);
        let action = Action::from_index(action_index).expect("chosen action is valid");
        position = position.play(action)?.position;
        player = -player;
    }

    let winner = match position.outcome {
        Some(GameOutcome::CurrentPlayerLoss) => -player,
        Some(GameOutcome::Draw) | None => 0,
    };
    Ok(ArenaGameResult {
        winner,
        plies: position.ply as usize,
    })
}

pub fn search_with_existing_evaluator(
    evaluator: &mut AnyEvaluator,
    position: Position,
    config: PuctConfig,
) -> SearchResult {
    struct Borrowed<'a>(&'a mut AnyEvaluator);
    impl Evaluator for Borrowed<'_> {
        fn evaluate(&mut self, positions: &[Position]) -> Vec<c4_infer::Evaluation> {
            self.0.evaluate(positions)
        }
    }
    let mut search = PuctSearch::new(Borrowed(evaluator), config);
    search.search_position(position).1
}

pub fn random_opening(mut seed: u64, plies: usize) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut position = Position::new();
    let mut opening = Vec::new();
    for _ in 0..plies {
        let legal: Vec<_> = position.legal_actions();
        if legal.is_empty() {
            break;
        }
        let action = legal[rng.random_range(0..legal.len())];
        opening.push(action.index());
        position = position
            .play(action)
            .expect("random legal opening")
            .position;
        if position.is_terminal() {
            break;
        }
        seed = seed.wrapping_add(1);
        let _ = seed;
    }
    opening
}

pub fn display_path(path: &Path) -> String {
    path.to_string_lossy().to_string()
}
