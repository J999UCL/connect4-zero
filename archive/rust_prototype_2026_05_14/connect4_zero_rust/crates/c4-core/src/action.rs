use serde::{Deserialize, Serialize};

use crate::constants::{ACTION_COUNT, BOARD_SIZE};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct Action(u8);

impl Action {
    pub fn from_index(index: usize) -> Option<Self> {
        (index < ACTION_COUNT).then_some(Self(index as u8))
    }

    pub fn from_xy(x: usize, y: usize) -> Option<Self> {
        (x < BOARD_SIZE && y < BOARD_SIZE).then_some(Self((x * BOARD_SIZE + y) as u8))
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }

    pub fn x(self) -> usize {
        self.index() / BOARD_SIZE
    }

    pub fn y(self) -> usize {
        self.index() % BOARD_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_indices_to_xy() {
        assert_eq!(Action::from_index(0).unwrap().x(), 0);
        assert_eq!(Action::from_index(0).unwrap().y(), 0);
        assert_eq!(Action::from_index(15).unwrap().x(), 3);
        assert_eq!(Action::from_index(15).unwrap().y(), 3);
        assert!(Action::from_index(16).is_none());
    }

    #[test]
    fn maps_xy_to_indices() {
        assert_eq!(Action::from_xy(0, 0).unwrap().index(), 0);
        assert_eq!(Action::from_xy(1, 0).unwrap().index(), 4);
        assert_eq!(Action::from_xy(3, 3).unwrap().index(), 15);
        assert!(Action::from_xy(4, 0).is_none());
        assert!(Action::from_xy(0, 4).is_none());
    }
}
