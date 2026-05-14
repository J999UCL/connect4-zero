use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use c4_core::{Position, constants::ACTION_COUNT};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelfPlaySample {
    pub position: Position,
    pub policy: [f32; ACTION_COUNT],
    pub value: i8,
    pub visit_counts: [u32; ACTION_COUNT],
    pub q_values: [f32; ACTION_COUNT],
    pub legal_mask: [bool; ACTION_COUNT],
    pub action: u8,
    pub ply: u8,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShardManifest {
    pub format: String,
    pub format_version: u32,
    pub samples: usize,
    pub generator: String,
    pub model_path: Option<String>,
    pub shards: Vec<ShardRecord>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShardRecord {
    pub path: String,
    pub samples: usize,
    pub compression: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub generator: String,
    pub model_path: Option<String>,
}

impl Default for DatasetMetadata {
    fn default() -> Self {
        Self {
            generator: "c4-selfplay".to_string(),
            model_path: None,
        }
    }
}

const MAGIC: &[u8; 8] = b"C4ZRS001";
const FORMAT: &str = "connect4_zero.rust.selfplay.v1";

pub fn write_dataset(
    output_dir: impl AsRef<Path>,
    samples: &[SelfPlaySample],
    samples_per_shard: usize,
    metadata: DatasetMetadata,
) -> Result<ShardManifest> {
    if samples_per_shard == 0 {
        bail!("samples_per_shard must be positive");
    }
    let output_dir = output_dir.as_ref();
    let shard_dir = output_dir.join("shards");
    fs::create_dir_all(&shard_dir).with_context(|| format!("create {}", shard_dir.display()))?;

    let mut records = Vec::new();
    for (shard_index, chunk) in samples.chunks(samples_per_shard).enumerate() {
        let name = format!("shard-{shard_index:06}.c4zst");
        let relative = PathBuf::from("shards").join(&name);
        let path = output_dir.join(&relative);
        write_shard(&path, chunk).with_context(|| format!("write shard {}", path.display()))?;
        records.push(ShardRecord {
            path: relative.to_string_lossy().to_string(),
            samples: chunk.len(),
            compression: "zstd".to_string(),
        });
    }

    let manifest = ShardManifest {
        format: FORMAT.to_string(),
        format_version: 1,
        samples: samples.len(),
        generator: metadata.generator,
        model_path: metadata.model_path,
        shards: records,
    };
    let manifest_path = output_dir.join("manifest.json");
    let manifest_json = serde_json::to_vec_pretty(&manifest)?;
    fs::write(&manifest_path, manifest_json)
        .with_context(|| format!("write manifest {}", manifest_path.display()))?;
    Ok(manifest)
}

pub fn write_shard(path: impl AsRef<Path>, samples: &[SelfPlaySample]) -> Result<()> {
    let mut payload = Vec::new();
    payload.extend_from_slice(MAGIC);
    write_u64(&mut payload, samples.len() as u64);

    for sample in samples {
        write_u64(&mut payload, sample.position.current);
    }
    for sample in samples {
        write_u64(&mut payload, sample.position.opponent);
    }
    for sample in samples {
        payload.extend_from_slice(&sample.position.heights);
    }
    for sample in samples {
        for value in sample.policy {
            write_f32(&mut payload, value);
        }
    }
    for sample in samples {
        payload.push(sample.value as u8);
    }
    for sample in samples {
        for value in sample.visit_counts {
            write_u32(&mut payload, value);
        }
    }
    for sample in samples {
        for value in sample.q_values {
            write_f32(&mut payload, value);
        }
    }
    for sample in samples {
        write_u16(&mut payload, bool_mask_to_u16(sample.legal_mask));
    }
    for sample in samples {
        payload.push(sample.action);
    }
    for sample in samples {
        payload.push(sample.ply);
    }

    let compressed = zstd::stream::encode_all(payload.as_slice(), 3)?;
    let mut file = File::create(path.as_ref())?;
    file.write_all(&compressed)?;
    Ok(())
}

pub fn read_shard(path: impl AsRef<Path>) -> Result<Vec<SelfPlaySample>> {
    let mut compressed = Vec::new();
    File::open(path.as_ref())?.read_to_end(&mut compressed)?;
    let payload = zstd::stream::decode_all(compressed.as_slice())?;
    parse_payload(&payload)
}

fn parse_payload(payload: &[u8]) -> Result<Vec<SelfPlaySample>> {
    let mut cursor = Cursor::new(payload);
    let magic = cursor.read_exact(8)?;
    if magic != MAGIC {
        bail!("invalid shard magic");
    }
    let n = cursor.read_u64()? as usize;

    let mut current = Vec::with_capacity(n);
    let mut opponent = Vec::with_capacity(n);
    let mut heights = Vec::with_capacity(n);
    let mut policies = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    let mut visit_counts = Vec::with_capacity(n);
    let mut q_values = Vec::with_capacity(n);
    let mut legal_masks = Vec::with_capacity(n);
    let mut actions = Vec::with_capacity(n);
    let mut plies = Vec::with_capacity(n);

    for _ in 0..n {
        current.push(cursor.read_u64()?);
    }
    for _ in 0..n {
        opponent.push(cursor.read_u64()?);
    }
    for _ in 0..n {
        let mut item = [0_u8; ACTION_COUNT];
        item.copy_from_slice(cursor.read_exact(ACTION_COUNT)?);
        heights.push(item);
    }
    for _ in 0..n {
        let mut item = [0.0_f32; ACTION_COUNT];
        for value in &mut item {
            *value = cursor.read_f32()?;
        }
        policies.push(item);
    }
    for _ in 0..n {
        values.push(cursor.read_i8()?);
    }
    for _ in 0..n {
        let mut item = [0_u32; ACTION_COUNT];
        for value in &mut item {
            *value = cursor.read_u32()?;
        }
        visit_counts.push(item);
    }
    for _ in 0..n {
        let mut item = [0.0_f32; ACTION_COUNT];
        for value in &mut item {
            *value = cursor.read_f32()?;
        }
        q_values.push(item);
    }
    for _ in 0..n {
        legal_masks.push(u16_to_bool_mask(cursor.read_u16()?));
    }
    for _ in 0..n {
        actions.push(cursor.read_u8()?);
    }
    for _ in 0..n {
        plies.push(cursor.read_u8()?);
    }

    let mut samples = Vec::with_capacity(n);
    for index in 0..n {
        samples.push(SelfPlaySample {
            position: Position::from_bits(
                current[index],
                opponent[index],
                heights[index],
                plies[index],
                None,
            ),
            policy: policies[index],
            value: values[index],
            visit_counts: visit_counts[index],
            q_values: q_values[index],
            legal_mask: legal_masks[index],
            action: actions[index],
            ply: plies[index],
        });
    }
    Ok(samples)
}

struct Cursor<'a> {
    payload: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(payload: &'a [u8]) -> Self {
        Self { payload, offset: 0 }
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8]> {
        if self.offset + len > self.payload.len() {
            bail!("unexpected end of shard payload");
        }
        let start = self.offset;
        self.offset += len;
        Ok(&self.payload[start..self.offset])
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let mut bytes = [0_u8; 2];
        bytes.copy_from_slice(self.read_exact(2)?);
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let mut bytes = [0_u8; 4];
        bytes.copy_from_slice(self.read_exact(4)?);
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(self.read_exact(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }
}

fn write_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn write_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn write_f32(output: &mut Vec<u8>, value: f32) {
    write_u32(output, value.to_bits());
}

pub fn bool_mask_to_u16(mask: [bool; ACTION_COUNT]) -> u16 {
    let mut packed = 0_u16;
    for (action, &legal) in mask.iter().enumerate() {
        if legal {
            packed |= 1_u16 << action;
        }
    }
    packed
}

pub fn u16_to_bool_mask(mask: u16) -> [bool; ACTION_COUNT] {
    let mut unpacked = [false; ACTION_COUNT];
    for (action, legal) in unpacked.iter_mut().enumerate() {
        *legal = mask & (1_u16 << action) != 0;
    }
    unpacked
}

pub fn legal_mask_from_position(position: &Position) -> [bool; ACTION_COUNT] {
    u16_to_bool_mask(position.legal_mask())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use c4_core::{Action, Position};

    use super::*;

    #[test]
    fn shard_round_trip_preserves_samples() {
        let temp = std::env::temp_dir().join(format!(
            "c4-data-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&temp).unwrap();
        let position = Position::new()
            .play(Action::from_index(0).unwrap())
            .unwrap()
            .position;
        let mut policy = [0.0_f32; ACTION_COUNT];
        policy[1] = 1.0;
        let mut visit_counts = [0_u32; ACTION_COUNT];
        visit_counts[1] = 7;
        let sample = SelfPlaySample {
            position,
            policy,
            value: 1,
            visit_counts,
            q_values: [0.0; ACTION_COUNT],
            legal_mask: legal_mask_from_position(&position),
            action: 1,
            ply: position.ply,
        };
        let manifest = write_dataset(
            &temp,
            std::slice::from_ref(&sample),
            1,
            DatasetMetadata::default(),
        )
        .unwrap();
        assert_eq!(manifest.samples, 1);
        let loaded = read_shard(temp.join(&manifest.shards[0].path)).unwrap();
        assert_eq!(loaded, vec![sample]);
        fs::remove_dir_all(temp).unwrap();
    }
}
