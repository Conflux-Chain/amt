//! Precomputed slot powers of the AMT quotients.
//!
//! Bumping slot `j` of leaf `idx` by one adds `quotients[d][idx] * 2^{40 j}`
//! to the proof of the level-`d` node above `idx`. [`QuotientPower`] stores
//! these points for `j` in `1..SLOT_SIZE` (slot 0 is `quotients[d][idx]`
//! itself), so a proof tree update is point additions only, like
//! `basis_power` for commitments.
//!
//! At depth 16 this is `16 * 2^16 * 5` points (~500 MB uncompressed on
//! BLS12-381), so it lives outside [`AmtParams`] and is only built or loaded
//! by nodes that maintain proofs.

use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use ark_ec::pairing::Pairing;
use ark_ff::AdditiveGroup;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::cfg_iter;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    ec_algebra::{CurveGroup, G1Aff, G1},
    error,
    proofs::SLOT_BITS,
    prove_params::{SLOT_SIZE, SLOT_SIZE_MINUS_1},
    utils::qpower_file_name,
    AmtParams,
};

/// `quotients[d][idx] * 2^{40 j}` for `j` in `1..SLOT_SIZE`, for every
/// level `d` and leaf `idx` of an [`AmtParams`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuotientPower<PE: Pairing> {
    powers: Vec<Vec<[G1Aff<PE>; SLOT_SIZE_MINUS_1]>>,
}

impl<PE: Pairing> QuotientPower<PE> {
    /// Computes the powers from `params` (40 doublings per point).
    pub fn new(params: &AmtParams<PE>) -> Self {
        let powers = params
            .quotients
            .iter()
            .map(|level| {
                let projective: Vec<[G1<PE>; SLOT_SIZE_MINUS_1]> =
                    cfg_iter!(level, 256).map(slot_powers::<PE>).collect();
                G1::<PE>::normalize_batch(projective.as_flattened())
                    .chunks_exact(SLOT_SIZE_MINUS_1)
                    .map(|c| c.try_into().unwrap())
                    .collect()
            })
            .collect();
        Self { powers }
    }

    /// Loads the powers for `params` from `dir`, or computes them and saves
    /// them there when the file is missing or does not match `params`.
    ///
    /// # Panics
    ///
    /// If the file cannot be used and `create` is false, or saving fails.
    pub fn from_dir(dir: impl AsRef<Path>, params: &AmtParams<PE>, create: bool) -> Self {
        let depth = ark_std::log2(params.len()) as usize;
        let prove_depth = params.quotients.len();
        let path = dir
            .as_ref()
            .join(qpower_file_name::<PE>(depth, prove_depth));

        match Self::load(&path) {
            Ok(loaded) if loaded.matches(params) => return loaded,
            Ok(_) => info!(?path, "Quotient powers do not match the AMT params"),
            Err(e) => info!(?path, error = ?e, "Fail to load quotient powers"),
        }

        if !create {
            panic!("Fail to load quotient powers from {:?}", path);
        }

        let generated = Self::new(params);
        info!(file = ?path, "Save generated quotient powers");
        let writer = BufWriter::new(File::create(&path).unwrap());
        generated.serialize_uncompressed(writer).unwrap();
        generated
    }

    fn load(path: &Path) -> Result<Self, error::Error> {
        let reader = BufReader::new(File::open(path)?);
        Ok(Self::deserialize_uncompressed_unchecked(reader)?)
    }

    /// Cheap consistency check against `params`: same shape, and the first
    /// and last points of every level are the right multiples.
    pub fn matches(&self, params: &AmtParams<PE>) -> bool {
        if self.powers.len() != params.quotients.len() {
            return false;
        }
        self.powers
            .iter()
            .zip(&params.quotients)
            .all(|(powers, quotients)| {
                powers.len() == quotients.len()
                    && [0, quotients.len() - 1].into_iter().all(|idx| {
                        let expected =
                            G1::<PE>::normalize_batch(&slot_powers::<PE>(&quotients[idx]));
                        powers[idx][..] == expected[..]
                    })
            })
    }

    /// Number of levels (the prove depth of the params they were built from).
    pub fn levels(&self) -> usize {
        self.powers.len()
    }

    /// `quotients[level][idx] * 2^{40 j}` for `j` in `1..SLOT_SIZE`.
    #[inline]
    pub fn at(&self, level: usize, idx: usize) -> &[G1Aff<PE>; SLOT_SIZE_MINUS_1] {
        &self.powers[level][idx]
    }

    /// Keeps the first `depth` levels, matching
    /// [`AmtParams::reduce_prove_depth`].
    pub fn reduce_prove_depth(&self, depth: usize) -> Self {
        Self {
            powers: self.powers[..depth].to_vec(),
        }
    }
}

impl<PE: Pairing> AmtParams<PE> {
    /// Quotient counterpart of [`Self::basis_power_at`]: returns
    /// `quotients[level][idx]` and its `2^{40 j}` multiples for `j` in
    /// `1..SLOT_SIZE`.
    #[inline]
    pub fn quotient_power_at<'a>(
        &'a self,
        powers: &'a QuotientPower<PE>,
        level: usize,
        idx: usize,
    ) -> (&'a G1Aff<PE>, &'a [G1Aff<PE>; SLOT_SIZE_MINUS_1]) {
        (&self.quotients[level][idx], powers.at(level, idx))
    }
}

/// `q * 2^{40 j}` for `j` in `1..SLOT_SIZE`, by repeated doubling.
fn slot_powers<PE: Pairing>(q: &G1Aff<PE>) -> [G1<PE>; SLOT_SIZE_MINUS_1] {
    let mut acc: G1<PE> = (*q).into();
    std::array::from_fn(|_| {
        for _ in 0..SLOT_BITS {
            acc.double_in_place();
        }
        acc
    })
}

const _: () = assert!(SLOT_SIZE_MINUS_1 + 1 == SLOT_SIZE);

// Encoded as the level count (u64), then per level the leaf count (u64)
// followed by the points, `SLOT_SIZE - 1` per leaf.
impl<PE: Pairing> CanonicalSerialize for QuotientPower<PE> {
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        (self.powers.len() as u64).serialize_with_mode(&mut writer, compress)?;
        for level in &self.powers {
            (level.len() as u64).serialize_with_mode(&mut writer, compress)?;
            for point in level.iter().flatten() {
                point.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let point = G1Aff::<PE>::default().serialized_size(compress);
        8 + self
            .powers
            .iter()
            .map(|level| 8 + level.len() * SLOT_SIZE_MINUS_1 * point)
            .sum::<usize>()
    }
}

impl<PE: Pairing> CanonicalDeserialize for QuotientPower<PE> {
    fn deserialize_with_mode<R: ark_serialize::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let levels = u64::deserialize_with_mode(&mut reader, compress, validate)?;
        let mut powers = Vec::new();
        for _ in 0..levels {
            let len = u64::deserialize_with_mode(&mut reader, compress, validate)? as usize;
            // Do not trust `len` for the allocation size.
            let mut level = Vec::with_capacity(len.min(1 << 20));
            for _ in 0..len {
                let mut leaf = [G1Aff::<PE>::default(); SLOT_SIZE_MINUS_1];
                for point in leaf.iter_mut() {
                    // Points are checked in one batch below.
                    *point =
                        G1Aff::<PE>::deserialize_with_mode(&mut reader, compress, Validate::No)?;
                }
                level.push(leaf);
            }
            powers.push(level);
        }
        let result = Self { powers };
        if let Validate::Yes = validate {
            result.check()?;
        }
        Ok(result)
    }
}

impl<PE: Pairing> Valid for QuotientPower<PE> {
    fn check(&self) -> Result<(), SerializationError> {
        Valid::batch_check(self.powers.iter().flatten().flatten())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ec_algebra::{Fr, FrInt},
        prove_params::tests::{AMT, PE, TEST_LENGTH, TEST_LEVEL},
    };

    static POWERS: once_cell::sync::Lazy<QuotientPower<PE>> =
        once_cell::sync::Lazy::new(|| QuotientPower::new(&AMT));

    #[test]
    fn test_powers_match_scalar_mul() {
        assert_eq!(POWERS.levels(), TEST_LEVEL);
        for d in 0..TEST_LEVEL {
            for idx in 0..TEST_LENGTH {
                let (q, powers) = AMT.quotient_power_at(&POWERS, d, idx);
                for (j, power) in powers.iter().enumerate() {
                    let mut scalar = FrInt::<PE>::from(1u64);
                    scalar <<= (SLOT_BITS * (j + 1)) as u32;
                    let expected = (*q * Fr::<PE>::from(scalar)).into_affine();
                    assert_eq!(*power, expected, "d = {d}, idx = {idx}, j = {}", j + 1);
                }
            }
        }
    }

    #[test]
    fn test_serde_round_trip() {
        for compress in [Compress::Yes, Compress::No] {
            let mut bytes = Vec::new();
            POWERS.serialize_with_mode(&mut bytes, compress).unwrap();
            assert_eq!(bytes.len(), POWERS.serialized_size(compress));
            for validate in [Validate::Yes, Validate::No] {
                let back =
                    QuotientPower::<PE>::deserialize_with_mode(&bytes[..], compress, validate)
                        .unwrap();
                assert_eq!(back, *POWERS);
            }
            // Truncated input is an error, not a panic.
            assert!(QuotientPower::<PE>::deserialize_with_mode(
                &bytes[..bytes.len() - 1],
                compress,
                Validate::Yes
            )
            .is_err());
        }
    }

    #[test]
    fn test_reduce_prove_depth() {
        for depth in 1..=TEST_LEVEL {
            let reduced_params = AMT.reduce_prove_depth(depth);
            let reduced = POWERS.reduce_prove_depth(depth);
            assert_eq!(reduced, QuotientPower::new(&reduced_params));
            assert!(reduced.matches(&reduced_params));
        }
    }

    #[test]
    fn test_matches_rejects_wrong_powers() {
        assert!(POWERS.matches(&AMT));
        assert!(!POWERS.matches(&AMT.reduce_prove_depth(TEST_LEVEL - 1)));

        let mut tampered = POWERS.clone();
        let last = TEST_LENGTH - 1;
        tampered.powers[TEST_LEVEL - 1][last][0] = tampered.powers[0][0][1];
        assert!(!tampered.matches(&AMT));
    }

    #[test]
    fn test_from_dir_creates_then_loads() {
        let dir = std::env::temp_dir().join(format!(
            "amt-qpower-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let created = QuotientPower::from_dir(&dir, &AMT, true);
        assert_eq!(created, *POWERS);
        let loaded = QuotientPower::from_dir(&dir, &AMT, false);
        assert_eq!(loaded, *POWERS);

        // A file for other params is regenerated instead of being used.
        let path = dir.join(qpower_file_name::<PE>(TEST_LEVEL, TEST_LEVEL));
        let mut wrong = POWERS.clone();
        wrong.powers[0][0][0] = wrong.powers[0][1][0];
        wrong
            .serialize_uncompressed(File::create(&path).unwrap())
            .unwrap();
        assert_eq!(QuotientPower::from_dir(&dir, &AMT, true), *POWERS);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
