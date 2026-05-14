use c4_core::{Action, Position, constants::ACTION_COUNT};

pub type NodeId = usize;

#[derive(Clone, Debug)]
pub struct Node {
    pub position: Position,
    pub parent: Option<NodeId>,
    pub parent_action: Option<Action>,
    pub children: [Option<NodeId>; ACTION_COUNT],
    pub legal_mask: u16,
    pub visits: u32,
    pub value_sum: f32,
    pub prior: f32,
    pub terminal_value: Option<f32>,
    pub priors: [f32; ACTION_COUNT],
    pub expanded: bool,
    pub depth: usize,
}

impl Node {
    pub fn root(position: Position) -> Self {
        Self {
            position,
            parent: None,
            parent_action: None,
            children: [None; ACTION_COUNT],
            legal_mask: position.legal_mask(),
            visits: 0,
            value_sum: 0.0,
            prior: 1.0,
            terminal_value: position.terminal_value(),
            priors: [0.0; ACTION_COUNT],
            expanded: false,
            depth: 0,
        }
    }

    pub fn mean_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

#[derive(Clone, Debug)]
pub struct SearchTree {
    pub nodes: Vec<Node>,
    pub root: NodeId,
    pub reused: bool,
}

impl SearchTree {
    pub fn new(position: Position) -> Self {
        Self {
            nodes: vec![Node::root(position)],
            root: 0,
            reused: false,
        }
    }

    pub fn root(&self) -> &Node {
        &self.nodes[self.root]
    }

    pub fn root_mut(&mut self) -> &mut Node {
        &mut self.nodes[self.root]
    }

    pub fn add_child(&mut self, parent: NodeId, action: Action, position: Position) -> NodeId {
        if let Some(child) = self.nodes[parent].children[action.index()] {
            return child;
        }
        let depth = self.nodes[parent].depth + 1;
        let child = Node {
            position,
            parent: Some(parent),
            parent_action: Some(action),
            children: [None; ACTION_COUNT],
            legal_mask: position.legal_mask(),
            visits: 0,
            value_sum: 0.0,
            prior: self.nodes[parent].priors[action.index()],
            terminal_value: position.terminal_value(),
            priors: [0.0; ACTION_COUNT],
            expanded: false,
            depth,
        };
        let child_id = self.nodes.len();
        self.nodes.push(child);
        self.nodes[parent].children[action.index()] = Some(child_id);
        child_id
    }

    pub fn advance_to_child(&mut self, action: Action) -> bool {
        match self.nodes[self.root].children[action.index()] {
            Some(child) => {
                self.root = child;
                self.reused = true;
                true
            }
            None => false,
        }
    }

    pub fn max_depth_from_root(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| self.is_descendant_of_root(node))
            .map(|node| node.depth.saturating_sub(self.nodes[self.root].depth))
            .max()
            .unwrap_or(0)
    }

    fn is_descendant_of_root(&self, node: &Node) -> bool {
        let mut current = node.parent;
        if std::ptr::eq(node, &self.nodes[self.root]) {
            return true;
        }
        while let Some(id) = current {
            if id == self.root {
                return true;
            }
            current = self.nodes[id].parent;
        }
        false
    }
}
