pub mod action;
pub mod constants;
pub mod geometry;
pub mod position;
pub mod symmetry;

pub use action::Action;
pub use position::{Cell, GameOutcome, PlayError, PlayResult, Position};

pub type Board = Position;
