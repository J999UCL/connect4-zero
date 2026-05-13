use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SearchDiagnostics {
    pub nodes: usize,
    pub max_depth: usize,
    pub leaf_evals: usize,
    pub terminal_evals: usize,
    pub root_visits: u32,
    pub reused_tree: bool,
}
