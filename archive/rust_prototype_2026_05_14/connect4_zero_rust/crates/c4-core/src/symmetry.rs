use crate::{
    action::Action,
    constants::{ACTION_COUNT, BOARD_SIZE},
    geometry::bit_mask,
    position::Position,
};

pub type ActionPermutation = [usize; ACTION_COUNT];

pub fn identity_action_permutation() -> ActionPermutation {
    let mut permutation = [0; ACTION_COUNT];
    for y in 0..BOARD_SIZE {
        for x in 0..BOARD_SIZE {
            let index = y * BOARD_SIZE + x;
            permutation[index] = index;
        }
    }
    permutation
}

pub const SYMMETRY_COUNT: usize = 8;

pub fn action_permutations() -> [ActionPermutation; SYMMETRY_COUNT] {
    let mut permutations = [[0; ACTION_COUNT]; SYMMETRY_COUNT];
    for (symmetry, permutation) in permutations.iter_mut().enumerate() {
        for (action_index, destination) in permutation.iter_mut().enumerate() {
            let action = Action::from_index(action_index).expect("valid action index");
            let (x, y) = transform_xy(symmetry, action.x(), action.y());
            *destination = Action::from_xy(x, y)
                .expect("valid transformed action")
                .index();
        }
    }
    permutations
}

pub fn transform_position(position: &Position, symmetry: usize) -> Position {
    assert!(symmetry < SYMMETRY_COUNT);
    let mut heights = [0_u8; ACTION_COUNT];
    for action_index in 0..ACTION_COUNT {
        let action = Action::from_index(action_index).expect("valid action index");
        let (x, y) = transform_xy(symmetry, action.x(), action.y());
        let transformed = Action::from_xy(x, y).expect("valid transformed action");
        heights[transformed.index()] = position.heights[action_index];
    }
    Position {
        current: transform_bits(position.current, symmetry),
        opponent: transform_bits(position.opponent, symmetry),
        heights,
        ply: position.ply,
        outcome: position.outcome,
    }
}

fn transform_bits(bits: u64, symmetry: usize) -> u64 {
    let mut transformed = 0_u64;
    for z in 0..BOARD_SIZE {
        for y in 0..BOARD_SIZE {
            for x in 0..BOARD_SIZE {
                let source = bit_mask(x, y, z);
                if bits & source == 0 {
                    continue;
                }
                let (tx, ty) = transform_xy(symmetry, x, y);
                transformed |= bit_mask(tx, ty, z);
            }
        }
    }
    transformed
}

fn transform_xy(symmetry: usize, x: usize, y: usize) -> (usize, usize) {
    let n = BOARD_SIZE - 1;
    match symmetry {
        0 => (x, y),
        1 => (n - y, x),
        2 => (n - x, n - y),
        3 => (y, n - x),
        4 => (n - x, y),
        5 => (x, n - y),
        6 => (y, x),
        7 => (n - y, n - x),
        _ => panic!("invalid symmetry index"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_keeps_all_actions_fixed() {
        let permutation = identity_action_permutation();
        assert_eq!(permutation[0], 0);
        assert_eq!(permutation[15], 15);
    }

    #[test]
    fn all_symmetry_permutations_are_bijections() {
        for permutation in action_permutations() {
            let mut sorted = permutation;
            sorted.sort_unstable();
            assert_eq!(
                sorted,
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            );
        }
    }

    #[test]
    fn transform_preserves_legal_action_count() {
        let position = Position::new()
            .play(Action::from_index(0).unwrap())
            .unwrap()
            .position;
        for symmetry in 0..SYMMETRY_COUNT {
            let transformed = transform_position(&position, symmetry);
            assert_eq!(
                transformed.legal_actions().len(),
                position.legal_actions().len()
            );
            assert_eq!(
                transformed.current.count_ones(),
                position.current.count_ones()
            );
            assert_eq!(
                transformed.opponent.count_ones(),
                position.opponent.count_ones()
            );
        }
    }
}
