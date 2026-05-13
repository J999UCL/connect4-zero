use crate::constants::{BOARD_SIZE, WIN_LENGTH, WIN_MASK_COUNT};

pub fn cell_index(x: usize, y: usize, z: usize) -> usize {
    debug_assert!(x < BOARD_SIZE);
    debug_assert!(y < BOARD_SIZE);
    debug_assert!(z < BOARD_SIZE);
    x * BOARD_SIZE * BOARD_SIZE + y * BOARD_SIZE + z
}

pub fn column_index(x: usize, y: usize) -> usize {
    debug_assert!(x < BOARD_SIZE);
    debug_assert!(y < BOARD_SIZE);
    x * BOARD_SIZE + y
}

pub const fn bit_mask(x: usize, y: usize, z: usize) -> u64 {
    1_u64 << (x * BOARD_SIZE * BOARD_SIZE + y * BOARD_SIZE + z)
}

pub const WIN_MASKS: [u64; WIN_MASK_COUNT] = make_win_masks();

const DIRECTIONS: [(i8, i8, i8); 13] = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, -1, 0),
    (1, 0, 1),
    (1, 0, -1),
    (0, 1, 1),
    (0, 1, -1),
    (1, 1, 1),
    (1, 1, -1),
    (1, -1, 1),
    (1, -1, -1),
];

const fn make_win_masks() -> [u64; WIN_MASK_COUNT] {
    let mut masks = [0_u64; WIN_MASK_COUNT];
    let mut count = 0;
    let mut direction_index = 0;
    while direction_index < DIRECTIONS.len() {
        let (dx, dy, dz) = DIRECTIONS[direction_index];
        let mut z = 0;
        while z < BOARD_SIZE {
            let mut y = 0;
            while y < BOARD_SIZE {
                let mut x = 0;
                while x < BOARD_SIZE {
                    if line_fits(x as i8, y as i8, z as i8, dx, dy, dz) {
                        masks[count] = line_mask(x as i8, y as i8, z as i8, dx, dy, dz);
                        count += 1;
                    }
                    x += 1;
                }
                y += 1;
            }
            z += 1;
        }
        direction_index += 1;
    }
    masks
}

const fn line_fits(x: i8, y: i8, z: i8, dx: i8, dy: i8, dz: i8) -> bool {
    let end_x = x + dx * (WIN_LENGTH as i8 - 1);
    let end_y = y + dy * (WIN_LENGTH as i8 - 1);
    let end_z = z + dz * (WIN_LENGTH as i8 - 1);
    in_bounds(x)
        && in_bounds(y)
        && in_bounds(z)
        && in_bounds(end_x)
        && in_bounds(end_y)
        && in_bounds(end_z)
}

const fn in_bounds(value: i8) -> bool {
    value >= 0 && value < BOARD_SIZE as i8
}

const fn line_mask(x: i8, y: i8, z: i8, dx: i8, dy: i8, dz: i8) -> u64 {
    let mut mask = 0_u64;
    let mut offset = 0;
    while offset < WIN_LENGTH {
        let cx = (x + dx * offset as i8) as usize;
        let cy = (y + dy * offset as i8) as usize;
        let cz = (z + dz * offset as i8) as usize;
        mask |= bit_mask(cx, cy, cz);
        offset += 1;
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes_bottom_and_top_corners() {
        assert_eq!(cell_index(0, 0, 0), 0);
        assert_eq!(cell_index(3, 3, 3), 63);
        assert_eq!(column_index(3, 3), 15);
    }

    #[test]
    fn enumerates_exactly_76_winning_lines() {
        assert_eq!(WIN_MASKS.len(), 76);
        assert!(WIN_MASKS.iter().all(|&mask| mask.count_ones() == 4));
        let mut unique = WIN_MASKS.to_vec();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 76);
    }
}
