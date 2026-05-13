use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArenaConfig {
    pub games: usize,
    pub batch_size: usize,
    pub opening_plies: usize,
    pub paired_openings: bool,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            games: 256,
            batch_size: 32,
            opening_plies: 6,
            paired_openings: true,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ArenaSummary {
    pub candidate_wins: usize,
    pub baseline_wins: usize,
    pub draws: usize,
    pub unique_openings: usize,
}
