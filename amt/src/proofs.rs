//! AMT opening proofs.
//!
//! A proof for leaf `idx` of an AMT of depth `D` lists, for each tree level
//! `d` in `0..D` (level `d` has `2^{d+1}` nodes), the commitment and quotient
//! proof of the sibling of the node that contains `idx`. With the parameter
//! layout of this crate (bit-reversed), that node is `idx >> (D - 1 - d)`, so
//! the subtree of a node is a contiguous range of leaf indices.
//!
//! Verification checks that
//!   1. `value * basis[idx]` plus the sibling commitments sums to the AMT
//!      commitment, and
//!   2. each sibling commitment `C` and proof `π` at level `d` satisfy
//!      `e(C, g2) == e(π, vanishes[d][sibling])`.

use std::ops::Deref;

use ark_ec::pairing::Pairing;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::{
    ec_algebra::{AffineRepr, BigInteger, CurveGroup, Fr, G1Aff, G2Aff, PrimeField, Zero, G1},
    prove_params::SLOT_SIZE,
    AmtParams,
};

/// Bit width of one version slot in an AMT leaf value.
pub const SLOT_BITS: usize = 40;

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct AllProofs<PE: Pairing> {
    pub(crate) commitments: Vec<Vec<G1Aff<PE>>>,
    pub(crate) proofs: Vec<Vec<G1Aff<PE>>>,
    pub(crate) input_len: usize,
    pub(crate) batch_size: usize,
}

impl<PE: Pairing> AllProofs<PE> {
    #[allow(unused)]
    pub fn get_proof(&self, reversed_index: usize) -> Proof<PE> {
        assert!(reversed_index * self.batch_size < self.input_len);
        let index_bits = self.commitments.len();

        let mut answer = vec![];

        for d in 0..index_bits {
            let lv_index = reversed_index >> (index_bits - 1 - d);
            let commitment = self.commitments[d][lv_index ^ 1];
            let proof = self.proofs[d][lv_index ^ 1];
            answer.push((commitment, proof));
        }
        Proof(answer)
    }
}

/// Sibling `(commitment, proof)` pairs from the top level (2 nodes) down to
/// the leaf level.
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct Proof<PE: Pairing>(Vec<(G1Aff<PE>, G1Aff<PE>)>);

impl<PE: Pairing> Proof<PE> {
    /// Builds a proof from sibling `(commitment, proof)` pairs, ordered from
    /// the top level down to the leaf level. Use this when the proof tree
    /// nodes are read from external storage.
    pub fn new(levels: Vec<(G1Aff<PE>, G1Aff<PE>)>) -> Self {
        Self(levels)
    }

    pub fn into_inner(self) -> Vec<(G1Aff<PE>, G1Aff<PE>)> {
        self.0
    }
}

impl<PE: Pairing> PartialEq for Proof<PE> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<PE: Pairing> Deref for Proof<PE> {
    type Target = Vec<(G1Aff<PE>, G1Aff<PE>)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum AmtProofError {
    /// The proof does not have one entry per tree level.
    UnexpectedDataLength,
    /// The leaf index is out of range.
    IncorrectPosition,
    /// The pairing check failed at the given level.
    KzgError(usize),
    /// The sibling commitments plus the leaf do not sum to the commitment.
    InconsistentCommitment,
}

impl<PE: Pairing> AmtParams<PE> {
    /// Verifies that leaf `idx` of the AMT committed to by `commitment` holds
    /// `value`.
    ///
    /// `idx` is the same leaf index used for commitment updates
    /// ([`Self::basis_power_at`]). For LVMT, `value` is the packed leaf
    /// value, see [`pack_slots`].
    ///
    /// # Panics
    ///
    /// If these parameters were reduced to a prove depth smaller than the
    /// full tree depth (single-value leaves need the full depth).
    pub fn verify(
        &self,
        commitment: &G1Aff<PE>,
        idx: usize,
        value: Fr<PE>,
        proof: &Proof<PE>,
    ) -> Result<(), AmtProofError> {
        verify_amt_proof(
            &self.basis,
            &self.vanishes,
            &self.g2,
            commitment,
            idx,
            value,
            proof,
        )
    }
}

pub(crate) fn verify_amt_proof<PE: Pairing>(
    basis: &[G1Aff<PE>],
    vanishes: &[Vec<G2Aff<PE>>],
    g2: &G2Aff<PE>,
    commitment: &G1Aff<PE>,
    idx: usize,
    value: Fr<PE>,
    proof: &Proof<PE>,
) -> Result<(), AmtProofError> {
    use AmtProofError::*;

    let depth = ark_std::log2(basis.len()) as usize;
    assert!(
        vanishes.len() >= depth,
        "verification needs {depth} levels of vanishes, got {}",
        vanishes.len()
    );

    if proof.len() != depth {
        return Err(UnexpectedDataLength);
    }
    if idx >= basis.len() {
        return Err(IncorrectPosition);
    }

    // Commitment consistency first: it is far cheaper than the pairings.
    let mut sum: G1<PE> = basis[idx] * value;
    for (sibling_commitment, _) in proof.iter() {
        sum += sibling_commitment;
    }
    if sum.into_affine() != *commitment {
        return Err(InconsistentCommitment);
    }

    for (d, (sibling_commitment, quotient)) in proof.iter().enumerate() {
        let sibling = (idx >> (depth - 1 - d)) ^ 1;
        let vanish = vanishes[d][sibling];
        // e(C, g2) == e(π, vanish)  <=>  e(C, g2) * e(-π, vanish) == 1
        let neg_quotient = -quotient.into_group();
        let product = PE::multi_pairing(
            [sibling_commitment.into_group(), neg_quotient],
            [g2.into_group(), vanish.into_group()],
        );
        if !product.is_zero() {
            return Err(KzgError(d));
        }
    }

    Ok(())
}

/// Packs `SLOT_SIZE` version numbers into one AMT leaf value:
/// `Σ vers[j] * 2^{40 j}`.
///
/// Returns `None` if a version does not fit in 40 bits.
pub fn pack_slots<F: PrimeField>(vers: &[u64; SLOT_SIZE]) -> Option<F> {
    debug_assert!(F::MODULUS_BIT_SIZE as usize > SLOT_BITS * SLOT_SIZE);
    let mut bytes = [0u8; SLOT_BITS * SLOT_SIZE / 8];
    for (j, ver) in vers.iter().enumerate() {
        if *ver >> SLOT_BITS != 0 {
            return None;
        }
        let start = j * SLOT_BITS / 8;
        bytes[start..start + SLOT_BITS / 8].copy_from_slice(&ver.to_le_bytes()[..SLOT_BITS / 8]);
    }
    Some(F::from_le_bytes_mod_order(&bytes))
}

/// Inverse of [`pack_slots`].
///
/// Returns `None` if `value` is not a packed leaf value, i.e. it has bits set
/// above the `SLOT_SIZE` slots. A verifier should reject such values rather
/// than read slots out of them.
pub fn unpack_slots<F: PrimeField>(value: &F) -> Option<[u64; SLOT_SIZE]> {
    const PACKED_BYTES: usize = SLOT_BITS * SLOT_SIZE / 8;
    let bytes = value.into_bigint().to_bytes_le();
    if bytes[PACKED_BYTES..].iter().any(|b| *b != 0) {
        return None;
    }
    let mut vers = [0u64; SLOT_SIZE];
    for (j, ver) in vers.iter_mut().enumerate() {
        let start = j * SLOT_BITS / 8;
        let mut buf = [0u8; 8];
        buf[..SLOT_BITS / 8].copy_from_slice(&bytes[start..start + SLOT_BITS / 8]);
        *ver = u64::from_le_bytes(buf);
    }
    Some(vers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        commitment::SlotChanges,
        ec_algebra::{Field, UniformRand, VariableBaseMSM},
        prove_params::tests::{AMT, PE, TEST_LENGTH, TEST_LEVEL},
    };
    use rand::Rng;

    type F = Fr<PE>;

    /// Builds every (commitment, proof) node of the tree with one MSM per
    /// node, straight from the definition: the node `k` at level `d` covers
    /// the contiguous leaves `k * len / 2^{d+1} ..`.
    fn naive_tree(values: &[F]) -> Vec<Vec<(G1Aff<PE>, G1Aff<PE>)>> {
        (0..TEST_LEVEL)
            .map(|d| {
                let chunk = TEST_LENGTH >> (d + 1);
                (0..1 << (d + 1))
                    .map(|k| {
                        let range = k * chunk..(k + 1) * chunk;
                        let c = G1::<PE>::msm(&AMT.basis[range.clone()], &values[range.clone()])
                            .unwrap();
                        let p = G1::<PE>::msm(&AMT.quotients[d][range.clone()], &values[range])
                            .unwrap();
                        (c.into_affine(), p.into_affine())
                    })
                    .collect()
            })
            .collect()
    }

    fn naive_proof(tree: &[Vec<(G1Aff<PE>, G1Aff<PE>)>], idx: usize) -> Proof<PE> {
        Proof::new(
            (0..TEST_LEVEL)
                .map(|d| tree[d][(idx >> (TEST_LEVEL - 1 - d)) ^ 1])
                .collect(),
        )
    }

    fn commit(values: &[F]) -> G1Aff<PE> {
        G1::<PE>::msm(&AMT.basis, values).unwrap().into_affine()
    }

    fn random_values(rng: &mut impl Rng) -> Vec<F> {
        (0..TEST_LENGTH).map(|_| F::rand(rng)).collect()
    }

    #[test]
    fn test_naive_tree_matches_pairing_relation() {
        let mut rng = rand::thread_rng();
        let values = random_values(&mut rng);
        let tree = naive_tree(&values);
        for (d, level) in tree.iter().enumerate() {
            for (k, (c, p)) in level.iter().enumerate() {
                assert_eq!(
                    PE::pairing(c, AMT.g2),
                    PE::pairing(p, AMT.vanishes[d][k]),
                    "d = {d}, k = {k}"
                );
            }
        }
    }

    #[test]
    fn test_verify_all_leaves() {
        let mut rng = rand::thread_rng();
        let values = random_values(&mut rng);
        let commitment = commit(&values);
        let tree = naive_tree(&values);
        for (idx, value) in values.iter().enumerate() {
            let proof = naive_proof(&tree, idx);
            assert_eq!(AMT.verify(&commitment, idx, *value, &proof), Ok(()));
        }
    }

    #[test]
    fn test_verify_matches_all_proofs_get_proof() {
        let mut rng = rand::thread_rng();
        let values = random_values(&mut rng);
        let tree = naive_tree(&values);
        let all = AllProofs::<PE> {
            commitments: tree
                .iter()
                .map(|l| l.iter().map(|x| x.0).collect())
                .collect(),
            proofs: tree
                .iter()
                .map(|l| l.iter().map(|x| x.1).collect())
                .collect(),
            input_len: TEST_LENGTH,
            batch_size: 1,
        };
        for idx in 0..TEST_LENGTH {
            assert_eq!(all.get_proof(idx), naive_proof(&tree, idx));
        }
    }

    #[test]
    fn test_verify_commitment_from_slot_changes() {
        // Commitments maintained by `apply_diffs` are what LVMT verifies
        // against; leaf values are the packed slot versions.
        let mut rng = rand::thread_rng();
        let mut versions = vec![[0u64; SLOT_SIZE]; TEST_LENGTH];
        let mut commitment = G1::<PE>::zero();
        for _ in 0..4 {
            let changes: Vec<(usize, SlotChanges)> = (0..8)
                .map(|_| {
                    let mut slots = [false; SLOT_SIZE];
                    slots.iter_mut().for_each(|s| *s = rng.gen_bool(0.4));
                    (rng.gen_range(0..TEST_LENGTH), slots)
                })
                .collect();
            for (idx, slots) in &changes {
                for (j, changed) in slots.iter().enumerate() {
                    versions[*idx][j] += *changed as u64;
                }
            }
            let updates = vec![(commitment, changes.iter().map(|(i, s)| (*i, s)))];
            commitment = AMT.apply_diffs(updates)[0].into_group();
        }

        let values: Vec<F> = versions.iter().map(|v| pack_slots(v).unwrap()).collect();
        let commitment = commitment.into_affine();
        assert_eq!(commitment, commit(&values));

        let tree = naive_tree(&values);
        for idx in 0..TEST_LENGTH {
            let proof = naive_proof(&tree, idx);
            assert_eq!(AMT.verify(&commitment, idx, values[idx], &proof), Ok(()));
            assert_eq!(unpack_slots(&values[idx]), Some(versions[idx]));
        }
    }

    #[test]
    fn test_verify_rejects_tampering() {
        use AmtProofError::*;
        let mut rng = rand::thread_rng();
        let values = random_values(&mut rng);
        let commitment = commit(&values);
        let tree = naive_tree(&values);
        let idx = rng.gen_range(0..TEST_LENGTH);
        let proof = naive_proof(&tree, idx);
        let value = values[idx];
        let junk = G1::<PE>::rand(&mut rng).into_affine();

        // Wrong value or wrong position break the commitment sum.
        assert_eq!(
            AMT.verify(&commitment, idx, value + F::from(1u64), &proof),
            Err(InconsistentCommitment)
        );
        assert_eq!(
            AMT.verify(&commitment, idx ^ 1, value, &proof),
            Err(InconsistentCommitment)
        );
        assert_eq!(
            AMT.verify(&junk, idx, value, &proof),
            Err(InconsistentCommitment)
        );
        assert_eq!(
            AMT.verify(&commitment, TEST_LENGTH, value, &proof),
            Err(IncorrectPosition)
        );

        // Wrong proof length.
        let mut short = proof.clone().into_inner();
        short.pop();
        assert_eq!(
            AMT.verify(&commitment, idx, value, &Proof::new(short)),
            Err(UnexpectedDataLength)
        );
        let mut long = proof.clone().into_inner();
        long.push(long[0]);
        assert_eq!(
            AMT.verify(&commitment, idx, value, &Proof::new(long)),
            Err(UnexpectedDataLength)
        );

        for d in 0..TEST_LEVEL {
            // A tampered quotient fails the pairing at its level.
            let mut levels = proof.clone().into_inner();
            levels[d].1 = junk;
            assert_eq!(
                AMT.verify(&commitment, idx, value, &Proof::new(levels)),
                Err(KzgError(d))
            );

            // Moving commitment mass between two levels keeps the sum, so
            // only the pairing check can catch it.
            let other = (d + 1) % TEST_LEVEL;
            let mut levels = proof.clone().into_inner();
            let delta = G1::<PE>::rand(&mut rng);
            levels[d].0 = (levels[d].0 + delta).into_affine();
            levels[other].0 = (levels[other].0.into_group() - delta).into_affine();
            assert_eq!(
                AMT.verify(&commitment, idx, value, &Proof::new(levels)),
                Err(KzgError(d.min(other)))
            );
        }
    }

    #[test]
    fn test_pack_unpack_slots() {
        let mut rng = rand::thread_rng();
        const MAX: u64 = (1 << SLOT_BITS) - 1;
        for _ in 0..1000 {
            let mut vers = [0u64; SLOT_SIZE];
            vers.iter_mut().for_each(|v| *v = rng.gen_range(0..=MAX));
            let value: F = pack_slots(&vers).unwrap();

            let mut expected = F::zero();
            for (j, v) in vers.iter().enumerate() {
                expected += F::from(*v) * F::from(2u64).pow([(SLOT_BITS * j) as u64]);
            }
            assert_eq!(value, expected);
            assert_eq!(unpack_slots(&value), Some(vers));
        }

        let mut vers = [MAX; SLOT_SIZE];
        assert!(pack_slots::<F>(&vers).is_some());
        vers[3] = MAX + 1;
        assert_eq!(pack_slots::<F>(&vers), None);

        // 2^240 is above every slot.
        let above = F::from(2u64).pow([(SLOT_BITS * SLOT_SIZE) as u64]);
        assert_eq!(unpack_slots(&above), None);
        assert_eq!(unpack_slots(&-F::from(1u64)), None);
    }
}
