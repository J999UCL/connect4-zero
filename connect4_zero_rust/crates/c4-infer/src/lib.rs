use c4_core::{
    Position,
    constants::{ACTION_COUNT, BOARD_SIZE},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Evaluation {
    pub priors: [f32; ACTION_COUNT],
    pub value: f32,
}

pub trait Evaluator {
    fn evaluate(&mut self, positions: &[Position]) -> Vec<Evaluation>;
}

#[derive(Clone, Debug, Default)]
pub struct UniformEvaluator;

impl Evaluator for UniformEvaluator {
    fn evaluate(&mut self, positions: &[Position]) -> Vec<Evaluation> {
        positions
            .iter()
            .map(|position| {
                let mut priors = [0.0; ACTION_COUNT];
                let legal_actions = position.legal_actions();
                let probability = 1.0 / legal_actions.len().max(1) as f32;
                for action in legal_actions {
                    priors[action.index()] = probability;
                }
                Evaluation { priors, value: 0.0 }
            })
            .collect()
    }
}

pub fn encode_positions(positions: &[Position]) -> Vec<f32> {
    let mut planes = vec![0.0_f32; positions.len() * 2 * BOARD_SIZE * BOARD_SIZE * BOARD_SIZE];
    for (batch, position) in positions.iter().enumerate() {
        for z in 0..BOARD_SIZE {
            for y in 0..BOARD_SIZE {
                for x in 0..BOARD_SIZE {
                    let cell = z * BOARD_SIZE * BOARD_SIZE + y * BOARD_SIZE + x;
                    let offset = batch * 2 * 64 + cell;
                    match position.cell(x, y, z) {
                        c4_core::Cell::Current => planes[offset] = 1.0,
                        c4_core::Cell::Opponent => planes[offset + 64] = 1.0,
                        c4_core::Cell::Empty => {}
                    }
                }
            }
        }
    }
    planes
}

pub fn masked_softmax(logits: &[f32; ACTION_COUNT], legal_mask: u16) -> [f32; ACTION_COUNT] {
    if legal_mask == 0 {
        return [0.0; ACTION_COUNT];
    }
    let mut max_logit = f32::NEG_INFINITY;
    for (action, &logit) in logits.iter().enumerate() {
        if legal_mask & (1_u16 << action) != 0 {
            max_logit = max_logit.max(logit);
        }
    }

    let mut priors = [0.0_f32; ACTION_COUNT];
    let mut total = 0.0_f32;
    for action in 0..ACTION_COUNT {
        if legal_mask & (1_u16 << action) == 0 {
            continue;
        }
        let weight = (logits[action] - max_logit).exp();
        priors[action] = weight;
        total += weight;
    }
    if total > 0.0 {
        for prior in &mut priors {
            *prior /= total;
        }
    }
    priors
}

#[cfg(feature = "onnx")]
pub mod onnx {
    use std::path::Path;

    use anyhow::{Context, Result, anyhow, bail};
    use c4_core::{Position, constants::ACTION_COUNT};
    use ort::{session::Session, value::Tensor};

    use crate::{Evaluation, Evaluator, encode_positions, masked_softmax};

    pub struct OnnxCpuEvaluator {
        session: Session,
        input_name: String,
        policy_name: String,
        value_name: String,
    }

    impl OnnxCpuEvaluator {
        pub fn new(path: impl AsRef<Path>) -> Result<Self> {
            let mut builder = Session::builder()
                .map_err(|error| anyhow!("{error}"))
                .context("create ONNX Runtime session builder")?;
            builder = builder
                .with_intra_threads(1)
                .map_err(|error| anyhow!("{error}"))
                .context("set ONNX intra threads")?
                .with_inter_threads(1)
                .map_err(|error| anyhow!("{error}"))
                .context("set ONNX inter threads")?
                .with_parallel_execution(false)
                .map_err(|error| anyhow!("{error}"))
                .context("disable ONNX parallel execution")?;
            let session = builder
                .commit_from_file(path.as_ref())
                .map_err(|error| anyhow!("{error}"))
                .with_context(|| format!("load ONNX model {}", path.as_ref().display()))?;
            let input_name = session
                .inputs()
                .first()
                .map(|input| input.name().to_string())
                .context("ONNX model has no inputs")?;
            let policy_name = session
                .outputs()
                .first()
                .map(|output| output.name().to_string())
                .context("ONNX model has no policy output")?;
            let value_name = session
                .outputs()
                .get(1)
                .map(|output| output.name().to_string())
                .context("ONNX model has no value output")?;
            Ok(Self {
                session,
                input_name,
                policy_name,
                value_name,
            })
        }
    }

    impl Evaluator for OnnxCpuEvaluator {
        fn evaluate(&mut self, positions: &[Position]) -> Vec<Evaluation> {
            if positions.is_empty() {
                return Vec::new();
            }
            self.evaluate_checked(positions)
                .expect("ONNX evaluator failed")
        }
    }

    impl OnnxCpuEvaluator {
        pub fn evaluate_checked(&mut self, positions: &[Position]) -> Result<Vec<Evaluation>> {
            if positions.is_empty() {
                return Ok(Vec::new());
            }
            let input = encode_positions(positions);
            let tensor =
                Tensor::from_array(([positions.len(), 2, 4, 4, 4], input.into_boxed_slice()))
                    .map_err(|error| anyhow!("{error}"))
                    .context("create ONNX input tensor")?;
            let outputs = self
                .session
                .run(ort::inputs![self.input_name.as_str() => tensor])
                .map_err(|error| anyhow!("{error}"))
                .context("run ONNX inference")?;
            let (_, policy_data) = outputs[self.policy_name.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|error| anyhow!("{error}"))
                .context("extract policy output")?;
            let (_, value_data) = outputs[self.value_name.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|error| anyhow!("{error}"))
                .context("extract value output")?;
            if policy_data.len() != positions.len() * ACTION_COUNT {
                bail!(
                    "policy output has {} values for {} positions",
                    policy_data.len(),
                    positions.len()
                );
            }
            if value_data.len() != positions.len() {
                bail!(
                    "value output has {} values for {} positions",
                    value_data.len(),
                    positions.len()
                );
            }

            let mut evaluations = Vec::with_capacity(positions.len());
            for (index, position) in positions.iter().enumerate() {
                let mut logits = [0.0_f32; ACTION_COUNT];
                logits.copy_from_slice(
                    &policy_data[index * ACTION_COUNT..(index + 1) * ACTION_COUNT],
                );
                evaluations.push(Evaluation {
                    priors: masked_softmax(&logits, position.legal_mask()),
                    value: value_data[index].clamp(-1.0, 1.0),
                });
            }
            Ok(evaluations)
        }
    }
}

#[cfg(test)]
mod tests {
    use c4_core::{Action, Position};

    use super::*;

    #[test]
    fn uniform_evaluator_masks_full_columns() {
        let mut position = Position::new();
        let action = Action::from_index(0).unwrap();
        for _ in 0..4 {
            position = position.play(action).unwrap().position;
        }
        let mut evaluator = UniformEvaluator;
        let evaluation = evaluator.evaluate(&[position])[0].clone();
        assert_eq!(evaluation.priors[0], 0.0);
        assert!((evaluation.priors.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn encoder_uses_current_and_opponent_planes() {
        let position = Position::new()
            .play(Action::from_index(0).unwrap())
            .unwrap()
            .position;
        let planes = encode_positions(&[position]);
        assert_eq!(planes.len(), 128);
        assert_eq!(planes[64], 1.0);
    }
}
