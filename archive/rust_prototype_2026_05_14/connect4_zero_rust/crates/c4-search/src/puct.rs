use c4_core::{Action, Position, constants::ACTION_COUNT};
use c4_infer::{Evaluation, Evaluator};
use rand::{SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Gamma};

use crate::{PuctConfig, SearchDiagnostics, SearchTree, tree::NodeId};

#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub policy: [f32; ACTION_COUNT],
    pub visits: [u32; ACTION_COUNT],
    pub q_values: [f32; ACTION_COUNT],
    pub root_value: f32,
    pub diagnostics: SearchDiagnostics,
}

pub struct PuctSearch<E> {
    pub evaluator: E,
    pub config: PuctConfig,
    rng: SmallRng,
    leaf_evals: usize,
    terminal_evals: usize,
}

impl<E: Evaluator> PuctSearch<E> {
    pub fn new(evaluator: E, config: PuctConfig) -> Self {
        let seed = config.seed;
        Self {
            evaluator,
            config,
            rng: SmallRng::seed_from_u64(seed),
            leaf_evals: 0,
            terminal_evals: 0,
        }
    }

    pub fn search_position(&mut self, position: Position) -> (SearchTree, SearchResult) {
        let mut tree = SearchTree::new(position);
        let result = self.search_tree(&mut tree);
        (tree, result)
    }

    pub fn search_tree(&mut self, tree: &mut SearchTree) -> SearchResult {
        self.leaf_evals = 0;
        self.terminal_evals = 0;
        if tree.root().terminal_value.is_none() {
            self.ensure_expanded(tree, tree.root);
            if self.config.add_root_noise {
                self.add_root_noise(tree);
            }
        }

        for _ in 0..self.config.simulations_per_move {
            if tree.root().terminal_value.is_some() {
                break;
            }
            let (path, value) = self.select_expand_evaluate(tree);
            self.backpropagate(tree, &path, value);
        }

        self.build_result(tree)
    }

    fn select_expand_evaluate(&mut self, tree: &mut SearchTree) -> (Vec<NodeId>, f32) {
        let mut path = vec![tree.root];
        let mut node_id = tree.root;

        loop {
            if let Some(value) = tree.nodes[node_id].terminal_value {
                self.terminal_evals += 1;
                return (path, value);
            }
            if !tree.nodes[node_id].expanded {
                let value = self.evaluate_node(tree, node_id);
                return (path, value);
            }

            let action = self.select_action(tree, node_id);
            let action = Action::from_index(action).expect("PUCT selected valid action");
            if let Some(child) = tree.nodes[node_id].children[action.index()] {
                node_id = child;
                path.push(node_id);
                continue;
            }

            let child_position = tree.nodes[node_id]
                .position
                .play(action)
                .expect("PUCT expanded legal action")
                .position;
            let child = tree.add_child(node_id, action, child_position);
            path.push(child);
            if let Some(value) = tree.nodes[child].terminal_value {
                self.terminal_evals += 1;
                return (path, value);
            }
            let value = self.evaluate_node(tree, child);
            return (path, value);
        }
    }

    fn ensure_expanded(&mut self, tree: &mut SearchTree, node_id: NodeId) {
        if !tree.nodes[node_id].expanded && tree.nodes[node_id].terminal_value.is_none() {
            self.evaluate_node(tree, node_id);
        }
    }

    fn evaluate_node(&mut self, tree: &mut SearchTree, node_id: NodeId) -> f32 {
        let position = tree.nodes[node_id].position;
        let Evaluation { priors, value } = self
            .evaluator
            .evaluate(std::slice::from_ref(&position))
            .into_iter()
            .next()
            .expect("evaluator returned no result");
        let legal_mask = tree.nodes[node_id].legal_mask;
        let mut masked = [0.0_f32; ACTION_COUNT];
        let mut total = 0.0_f32;
        for (action, masked_prior) in masked.iter_mut().enumerate() {
            if legal_mask & (1_u16 << action) == 0 {
                continue;
            }
            *masked_prior = priors[action].max(0.0);
            total += *masked_prior;
        }
        if total <= 0.0 {
            let count = legal_mask.count_ones().max(1) as f32;
            for (action, masked_prior) in masked.iter_mut().enumerate() {
                if legal_mask & (1_u16 << action) != 0 {
                    *masked_prior = 1.0 / count;
                }
            }
        } else {
            for prior in &mut masked {
                *prior /= total;
            }
        }
        tree.nodes[node_id].priors = masked;
        tree.nodes[node_id].expanded = true;
        self.leaf_evals += 1;
        value.clamp(-1.0, 1.0)
    }

    fn select_action(&self, tree: &SearchTree, node_id: NodeId) -> usize {
        let node = &tree.nodes[node_id];
        let parent_visits = node.visits as f32;
        let mut best_action = 0_usize;
        let mut best_score = f32::NEG_INFINITY;
        for action in 0..ACTION_COUNT {
            if node.legal_mask & (1_u16 << action) == 0 {
                continue;
            }
            let prior = node.priors[action];
            let (child_visits, q_value) = match node.children[action] {
                Some(child) => {
                    let child_node = &tree.nodes[child];
                    (child_node.visits as f32, -child_node.mean_value())
                }
                None => (0.0, 0.0),
            };
            let exploration =
                self.config.c_puct * prior * (parent_visits + 1.0).sqrt() / (1.0 + child_visits);
            let score = q_value + exploration;
            if score > best_score {
                best_score = score;
                best_action = action;
            }
        }
        best_action
    }

    fn backpropagate(&self, tree: &mut SearchTree, path: &[NodeId], leaf_value: f32) {
        let mut value = leaf_value;
        for &node_id in path.iter().rev() {
            let node = &mut tree.nodes[node_id];
            node.visits += 1;
            node.value_sum += value;
            value = -value;
        }
    }

    fn add_root_noise(&mut self, tree: &mut SearchTree) {
        let root = tree.root;
        let legal: Vec<usize> = (0..ACTION_COUNT)
            .filter(|&action| tree.nodes[root].legal_mask & (1_u16 << action) != 0)
            .collect();
        if legal.is_empty() {
            return;
        }
        let gamma = Gamma::new(self.config.root_dirichlet_alpha, 1.0)
            .expect("positive root_dirichlet_alpha");
        let mut samples = Vec::with_capacity(legal.len());
        let mut total = 0.0_f32;
        for _ in &legal {
            let sample: f32 = gamma.sample(&mut self.rng);
            samples.push(sample);
            total += sample;
        }
        if total <= 0.0 {
            return;
        }
        let fraction = self.config.root_exploration_fraction;
        for (&action, &sample) in legal.iter().zip(samples.iter()) {
            let noise = sample / total;
            let current = tree.nodes[root].priors[action];
            tree.nodes[root].priors[action] = (1.0 - fraction) * current + fraction * noise;
        }
    }

    fn build_result(&self, tree: &SearchTree) -> SearchResult {
        let root = tree.root();
        let mut visits = [0_u32; ACTION_COUNT];
        let mut q_values = [0.0_f32; ACTION_COUNT];
        for action in 0..ACTION_COUNT {
            if root.legal_mask & (1_u16 << action) == 0 {
                continue;
            }
            if let Some(child) = root.children[action] {
                visits[action] = tree.nodes[child].visits;
                q_values[action] = -tree.nodes[child].mean_value();
            }
        }
        let policy = policy_from_visits(visits, root.legal_mask, self.config.policy_temperature);
        SearchResult {
            policy,
            visits,
            q_values,
            root_value: root.mean_value(),
            diagnostics: SearchDiagnostics {
                nodes: tree.nodes.len(),
                max_depth: tree.max_depth_from_root(),
                leaf_evals: self.leaf_evals,
                terminal_evals: self.terminal_evals,
                root_visits: root.visits,
                reused_tree: tree.reused,
            },
        }
    }
}

pub fn policy_from_visits(
    visits: [u32; ACTION_COUNT],
    legal_mask: u16,
    temperature: f32,
) -> [f32; ACTION_COUNT] {
    let mut policy = [0.0_f32; ACTION_COUNT];
    if legal_mask == 0 {
        return policy;
    }
    if temperature <= 0.0 {
        let mut best_action = None;
        let mut best_visits = 0_u32;
        for (action, &visit_count) in visits.iter().enumerate() {
            if legal_mask & (1_u16 << action) == 0 {
                continue;
            }
            if best_action.is_none() || visit_count > best_visits {
                best_action = Some(action);
                best_visits = visit_count;
            }
        }
        if let Some(action) = best_action {
            policy[action] = 1.0;
        }
        return policy;
    }

    let exponent = 1.0 / temperature;
    let mut total = 0.0_f32;
    for (action, policy_value) in policy.iter_mut().enumerate() {
        if legal_mask & (1_u16 << action) == 0 {
            continue;
        }
        let weight = (visits[action] as f32).powf(exponent);
        *policy_value = weight;
        total += weight;
    }
    if total <= 0.0 {
        let count = legal_mask.count_ones() as f32;
        for (action, policy_value) in policy.iter_mut().enumerate() {
            if legal_mask & (1_u16 << action) != 0 {
                *policy_value = 1.0 / count;
            }
        }
    } else {
        for value in &mut policy {
            *value /= total;
        }
    }
    policy
}

#[cfg(test)]
mod tests {
    use c4_core::{Action, Position};
    use c4_infer::UniformEvaluator;

    use super::*;

    fn searched(config: PuctConfig) -> (SearchTree, SearchResult) {
        let mut search = PuctSearch::new(UniformEvaluator, config);
        search.search_position(Position::new())
    }

    #[test]
    fn root_visits_sum_to_simulations() {
        let (_, result) = searched(PuctConfig {
            simulations_per_move: 32,
            ..PuctConfig::default()
        });
        assert_eq!(result.visits.iter().sum::<u32>(), 32);
        assert_eq!(result.diagnostics.root_visits, 32);
    }

    #[test]
    fn illegal_actions_have_zero_policy() {
        let mut position = Position::new();
        let action = Action::from_index(0).unwrap();
        for _ in 0..4 {
            position = position.play(action).unwrap().position;
        }
        let mut search = PuctSearch::new(
            UniformEvaluator,
            PuctConfig {
                simulations_per_move: 32,
                ..PuctConfig::default()
            },
        );
        let (_, result) = search.search_position(position);
        assert_eq!(result.policy[0], 0.0);
        assert_eq!(result.visits[0], 0);
    }

    #[test]
    fn immediate_win_receives_dominant_visits() {
        let mut position = Position::new();
        for action in [0, 1, 4, 2, 8, 3] {
            position = position
                .play(Action::from_index(action).unwrap())
                .unwrap()
                .position;
        }
        let mut search = PuctSearch::new(
            UniformEvaluator,
            PuctConfig {
                simulations_per_move: 64,
                policy_temperature: 0.0,
                ..PuctConfig::default()
            },
        );
        let (_, result) = search.search_position(position);
        assert_eq!(
            result
                .visits
                .iter()
                .enumerate()
                .max_by_key(|(_, visits)| *visits)
                .map(|(action, _)| action),
            Some(12)
        );
        assert_eq!(result.policy[12], 1.0);
    }

    #[test]
    fn tree_reuse_keeps_existing_child_visits() {
        let mut search = PuctSearch::new(
            UniformEvaluator,
            PuctConfig {
                simulations_per_move: 32,
                ..PuctConfig::default()
            },
        );
        let (mut tree, result) = search.search_position(Position::new());
        let action_index = result
            .visits
            .iter()
            .enumerate()
            .max_by_key(|(_, visits)| *visits)
            .map(|(action, _)| action)
            .unwrap();
        let action = Action::from_index(action_index).unwrap();
        assert!(tree.advance_to_child(action));
        let before = tree.root().visits;
        let result = search.search_tree(&mut tree);
        assert!(result.diagnostics.reused_tree);
        assert!(tree.root().visits > before);
    }
}
