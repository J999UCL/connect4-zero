use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PuctConfig {
    pub simulations_per_move: usize,
    pub max_leaf_batch_size: usize,
    pub c_puct: f32,
    pub policy_temperature: f32,
    pub root_dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    pub add_root_noise: bool,
    pub seed: u64,
}

impl Default for PuctConfig {
    fn default() -> Self {
        Self {
            simulations_per_move: 128,
            max_leaf_batch_size: 1,
            c_puct: 1.5,
            policy_temperature: 1.0,
            root_dirichlet_alpha: 0.3,
            root_exploration_fraction: 0.25,
            add_root_noise: false,
            seed: 0,
        }
    }
}
