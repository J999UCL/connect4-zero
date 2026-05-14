use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    action::Action,
    constants::{ACTION_COUNT, BOARD_SIZE, CELL_COUNT, COLUMN_COUNT},
    geometry::{WIN_MASKS, bit_mask, cell_index, column_index},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Cell {
    Empty,
    Current,
    Opponent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum GameOutcome {
    CurrentPlayerLoss,
    Draw,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub current: u64,
    pub opponent: u64,
    pub heights: [u8; COLUMN_COUNT],
    pub ply: u8,
    pub outcome: Option<GameOutcome>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlayResult {
    pub position: Position,
    pub outcome: Option<GameOutcome>,
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum PlayError {
    #[error("cannot play into a terminal position")]
    Terminal,
    #[error("column is full")]
    FullColumn,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            current: 0,
            opponent: 0,
            heights: [0; COLUMN_COUNT],
            ply: 0,
            outcome: None,
        }
    }
}

impl Position {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_bits(
        current: u64,
        opponent: u64,
        heights: [u8; COLUMN_COUNT],
        ply: u8,
        outcome: Option<GameOutcome>,
    ) -> Self {
        debug_assert_eq!(current & opponent, 0);
        Self {
            current,
            opponent,
            heights,
            ply,
            outcome,
        }
    }

    pub fn occupancy(&self) -> u64 {
        self.current | self.opponent
    }

    pub fn is_terminal(&self) -> bool {
        self.outcome.is_some()
    }

    pub fn terminal_value(&self) -> Option<f32> {
        match self.outcome {
            Some(GameOutcome::CurrentPlayerLoss) => Some(-1.0),
            Some(GameOutcome::Draw) => Some(0.0),
            None => None,
        }
    }

    pub fn height(&self, action: Action) -> u8 {
        self.heights[action.index()]
    }

    pub fn cell(&self, x: usize, y: usize, z: usize) -> Cell {
        let mask = bit_mask(x, y, z);
        if self.current & mask != 0 {
            Cell::Current
        } else if self.opponent & mask != 0 {
            Cell::Opponent
        } else {
            Cell::Empty
        }
    }

    pub fn board_array(&self) -> [i8; CELL_COUNT] {
        let mut board = [0_i8; CELL_COUNT];
        for z in 0..BOARD_SIZE {
            for y in 0..BOARD_SIZE {
                for x in 0..BOARD_SIZE {
                    board[cell_index(x, y, z)] = match self.cell(x, y, z) {
                        Cell::Empty => 0,
                        Cell::Current => 1,
                        Cell::Opponent => -1,
                    };
                }
            }
        }
        board
    }

    pub fn legal_mask(&self) -> u16 {
        if self.is_terminal() {
            return 0;
        }
        let mut mask = 0_u16;
        for action in 0..ACTION_COUNT {
            if self.heights[action] < BOARD_SIZE as u8 {
                mask |= 1_u16 << action;
            }
        }
        mask
    }

    pub fn legal_actions(&self) -> Vec<Action> {
        (0..ACTION_COUNT)
            .filter_map(Action::from_index)
            .filter(|&action| self.is_legal(action))
            .collect()
    }

    pub fn is_legal(&self, action: Action) -> bool {
        !self.is_terminal() && self.heights[action.index()] < BOARD_SIZE as u8
    }

    pub fn play(&self, action: Action) -> Result<PlayResult, PlayError> {
        if self.is_terminal() {
            return Err(PlayError::Terminal);
        }
        if !self.is_legal(action) {
            return Err(PlayError::FullColumn);
        }

        let mut heights = self.heights;
        let x = action.x();
        let y = action.y();
        let z = heights[column_index(x, y)] as usize;
        let placed = bit_mask(x, y, z);
        heights[action.index()] += 1;

        let mover_bits = self.current | placed;
        let ply = self.ply + 1;
        let outcome = if has_won(mover_bits) {
            Some(GameOutcome::CurrentPlayerLoss)
        } else if ply as usize == CELL_COUNT {
            Some(GameOutcome::Draw)
        } else {
            None
        };
        let next = Position {
            current: self.opponent,
            opponent: mover_bits,
            heights,
            ply,
            outcome,
        };
        Ok(PlayResult {
            position: next,
            outcome,
        })
    }
}

pub fn has_won(bits: u64) -> bool {
    for &mask in &WIN_MASKS {
        if bits & mask == mask {
            return true;
        }
    }
    false
}

pub fn legal_mask_to_actions(mask: u16) -> Vec<Action> {
    (0..ACTION_COUNT)
        .filter(|&action| mask & (1_u16 << action) != 0)
        .filter_map(Action::from_index)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn play_sequence(actions: &[usize]) -> Position {
        let mut position = Position::new();
        for &action in actions {
            position = position
                .play(Action::from_index(action).unwrap())
                .unwrap()
                .position;
        }
        position
    }

    #[test]
    fn empty_position_has_all_actions_legal() {
        let position = Position::new();
        assert_eq!(position.legal_actions().len(), COLUMN_COUNT);
        assert_eq!(position.legal_mask(), 0xffff);
    }

    #[test]
    fn play_places_piece_at_column_height_and_flips_perspective() {
        let position = Position::new();
        let action = Action::from_xy(2, 1).unwrap();
        let next = position.play(action).unwrap().position;

        assert_eq!(next.height(action), 1);
        assert_eq!(next.cell(2, 1, 0), Cell::Opponent);
        assert_eq!(next.ply, 1);
    }

    #[test]
    fn full_column_is_illegal() {
        let mut position = Position::new();
        let action = Action::from_index(0).unwrap();
        for _ in 0..4 {
            position = position.play(action).unwrap().position;
        }
        assert!(!position.is_legal(action));
        assert_eq!(position.play(action).unwrap_err(), PlayError::FullColumn);
    }

    #[test]
    fn detects_horizontal_win() {
        let position = play_sequence(&[0, 1, 4, 2, 8, 3, 12]);
        assert_eq!(position.outcome, Some(GameOutcome::CurrentPlayerLoss));
        assert_eq!(position.terminal_value(), Some(-1.0));
    }

    #[test]
    fn detects_vertical_win() {
        let position = play_sequence(&[0, 1, 0, 1, 0, 1, 0]);
        assert_eq!(position.outcome, Some(GameOutcome::CurrentPlayerLoss));
    }

    #[test]
    fn detects_3d_diagonal_win() {
        let mut position = Position::new();
        let mut heights = [0_u8; COLUMN_COUNT];
        heights[0] = 1;
        heights[5] = 2;
        heights[10] = 3;
        heights[15] = 4;
        let current = bit_mask(0, 0, 0) | bit_mask(1, 1, 1) | bit_mask(2, 2, 2) | bit_mask(3, 3, 3);
        position.current = current;
        position.heights = heights;
        assert!(has_won(position.current));
    }
}
