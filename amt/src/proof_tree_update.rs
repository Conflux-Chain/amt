//! Incremental maintenance of the AMT proof tree.
//!
//! Bumping slot `j` of leaf `idx` by one adds, at every level `d` of the
//! proof tree (see [`crate::proof_tree`] for the layout), to the node
//! containing `idx`:
//!   - `basis[idx] * 2^{40 j}` to its commitment, and
//!   - `quotients[d][idx] * 2^{40 j}` to its proof.
//!
//! [`AmtParams::proof_tree_diffs`] turns a batch of slot changes into one
//! [`NodeDiff`] per touched node. It does not know how nodes are stored, so
//! callers can keep them in their own (versioned) storage; [`ProofTree`] is a
//! sparse in-memory store built on it.

use std::collections::HashMap;

use ark_ec::{
    pairing::Pairing,
    short_weierstrass::{Affine, Projective, SWCurveConfig},
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    commitment::{batch_normalize, SlotChanges},
    ec_algebra::{AffineRepr, CurveGroup, Fr, G1Aff, Zero, G1},
    proofs::{pack_slots, AllProofs, Proof},
    AmtParams, QuotientPower, SLOT_SIZE,
};

/// With the `parallel` feature, [`AmtParams::proof_tree_diffs`] spreads
/// levels over threads only from this many changed leaves on (when
/// quotient powers are given; scalar multiplications always go parallel).
pub const PROOF_DIFFS_PARALLEL_THRESHOLD: usize = 64;

/// Change of one proof tree node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeDiff<PE: Pairing> {
    /// Tree level, `0..prove_depth`; level `d` has `2^{d+1}` nodes.
    pub level: usize,
    /// Node index within the level.
    pub index: usize,
    /// Added to the node commitment.
    pub commitment: G1<PE>,
    /// Added to the node proof.
    pub proof: G1<PE>,
}

impl<PE: Pairing> AmtParams<PE> {
    /// Node changes caused by `changes`, a list of `(leaf index, slot
    /// changes)`, aggregated so that each touched node appears once. The
    /// result is sorted by `(level, index)`.
    ///
    /// With `powers`, every proof update is point additions; without it,
    /// each changed leaf costs one scalar multiplication per level.
    ///
    /// A leaf listed several times is bumped once per listing, as in
    /// [`Self::commitment_diff`].
    pub fn proof_tree_diffs<'a>(
        &self,
        changes: impl IntoIterator<Item = (usize, &'a SlotChanges)>,
        powers: Option<&QuotientPower<PE>>,
    ) -> Vec<NodeDiff<PE>> {
        self.proof_tree_diffs_with(changes, powers, |_, _| true)
    }

    /// Like [`Self::proof_tree_diffs`], but only for nodes where
    /// `keep(level, index)` is true, e.g. the nodes of a proof shard. Proof
    /// changes of other nodes are not computed.
    pub fn proof_tree_diffs_with<'a>(
        &self,
        changes: impl IntoIterator<Item = (usize, &'a SlotChanges)>,
        powers: Option<&QuotientPower<PE>>,
        keep: impl Fn(usize, usize) -> bool + Sync,
    ) -> Vec<NodeDiff<PE>> {
        let levels = self.quotients.len();
        if let Some(powers) = powers {
            assert_eq!(
                powers.levels(),
                levels,
                "quotient powers do not match the prove depth"
            );
        }
        let depth = ark_std::log2(self.len()) as usize;

        let mut changes: Vec<(usize, &SlotChanges)> = changes
            .into_iter()
            .filter(|(_, slots)| slots.iter().any(|s| *s))
            .collect();
        if changes.is_empty() {
            return Vec::new();
        }
        for (idx, _) in &changes {
            assert!(*idx < self.len(), "leaf index {idx} out of range");
        }
        // Sorted leaves give non-decreasing node indices at every level.
        changes.sort_by_key(|(idx, _)| *idx);

        // Commitments: per-leaf diffs merged into bottom nodes, then summed
        // pairwise up the tree.
        let shift = |d: usize| depth - 1 - d;
        let mut commitments: Vec<Vec<(usize, G1<PE>)>> = vec![Vec::new(); levels];
        commitments[levels - 1] = merge_sorted(changes.iter().map(|(idx, slots)| {
            (
                idx >> shift(levels - 1),
                self.commitment_diff([(*idx, *slots)]),
            )
        }));
        for d in (0..levels - 1).rev() {
            commitments[d] = merge_sorted(commitments[d + 1].iter().map(|(k, c)| (k >> 1, *c)));
        }

        // Proofs: every level needs its own quotients, so levels are
        // independent.
        let scalars: Option<Vec<Fr<PE>>> = powers
            .is_none()
            .then(|| changes.iter().map(|(_, s)| slot_scalar::<PE>(s)).collect());
        let level_proofs = |d: usize| -> Vec<(usize, G1<PE>)> {
            merge_sorted(
                changes
                    .iter()
                    .enumerate()
                    .map(|(i, (idx, slots))| (i, idx >> shift(d), *idx, *slots))
                    .filter(|(_, node, _, _)| keep(d, *node))
                    .map(|(i, node, idx, slots)| {
                        let diff = match (powers, &scalars) {
                            (Some(powers), _) => {
                                let (q, qp) = self.quotient_power_at(powers, d, idx);
                                let mut acc = G1::<PE>::zero();
                                add_slot_points::<PE>(&mut acc, q, qp, slots);
                                acc
                            }
                            (None, Some(scalars)) => self.quotients[d][idx] * scalars[i],
                            (None, None) => unreachable!(),
                        };
                        (node, diff)
                    }),
            )
        };
        // Waking the rayon pool costs ~20µs, more than a small batch of point
        // additions takes. Scalar multiplications are always worth it.
        #[cfg(feature = "parallel")]
        let proofs: Vec<_> = if powers.is_none() || changes.len() >= PROOF_DIFFS_PARALLEL_THRESHOLD
        {
            (0..levels).into_par_iter().map(level_proofs).collect()
        } else {
            (0..levels).map(level_proofs).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let proofs: Vec<_> = (0..levels).map(level_proofs).collect();

        let mut diffs = Vec::new();
        for (d, (commitments, proofs)) in commitments.into_iter().zip(proofs).enumerate() {
            let kept = commitments.into_iter().filter(|(k, _)| keep(d, *k));
            for ((index, commitment), (proof_index, proof)) in kept.zip(proofs) {
                debug_assert_eq!(index, proof_index);
                diffs.push(NodeDiff {
                    level: d,
                    index,
                    commitment,
                    proof,
                });
            }
        }
        diffs
    }
}

/// Sums values with equal consecutive keys.
fn merge_sorted<G: Zero + std::ops::AddAssign + Copy>(
    items: impl IntoIterator<Item = (usize, G)>,
) -> Vec<(usize, G)> {
    let mut out: Vec<(usize, G)> = Vec::new();
    for (key, value) in items {
        match out.last_mut() {
            Some((last, acc)) if *last == key => *acc += value,
            _ => out.push((key, value)),
        }
    }
    out
}

fn add_slot_points<PE: Pairing>(
    acc: &mut G1<PE>,
    base: &G1Aff<PE>,
    powers: &[G1Aff<PE>; SLOT_SIZE - 1],
    slots: &SlotChanges,
) {
    if slots[0] {
        *acc += base;
    }
    for (changed, power) in slots[1..].iter().zip(powers) {
        if *changed {
            *acc += power;
        }
    }
}

/// `Σ_{changed j} 2^{40 j}`.
fn slot_scalar<PE: Pairing>(slots: &SlotChanges) -> Fr<PE> {
    pack_slots(&slots.map(u64::from)).unwrap()
}

/// `(commitment, proof)` of one proof tree node.
pub type ProofNode<PE> = (G1Aff<PE>, G1Aff<PE>);

/// A sparse in-memory AMT proof tree: nodes whose commitment and proof are
/// both zero are not stored.
#[derive(Clone, Debug)]
pub struct ProofTree<PE: Pairing> {
    /// `log2` of the number of leaves.
    depth: usize,
    levels: Vec<HashMap<usize, ProofNode<PE>>>,
}

impl<PE: Pairing> ProofTree<PE> {
    /// An empty tree (all leaf values zero) shaped for `params`.
    pub fn new(params: &AmtParams<PE>) -> Self {
        Self {
            depth: ark_std::log2(params.len()) as usize,
            levels: vec![HashMap::new(); params.quotients.len()],
        }
    }

    /// Converts a tree from [`AmtParams::build_proof_tree`].
    pub fn from_all_proofs(all: &AllProofs<PE>) -> Self {
        let levels = all
            .commitments
            .iter()
            .zip(&all.proofs)
            .map(|(commitments, proofs)| {
                commitments
                    .iter()
                    .zip(proofs)
                    .enumerate()
                    .filter(|(_, (c, p))| !c.is_zero() || !p.is_zero())
                    .map(|(k, (c, p))| (k, (*c, *p)))
                    .collect()
            })
            .collect();
        Self {
            depth: ark_std::log2(all.input_len) as usize,
            levels,
        }
    }

    /// Number of levels (the prove depth).
    pub fn levels(&self) -> usize {
        self.levels.len()
    }

    /// Number of stored (non-zero) nodes.
    pub fn stored_nodes(&self) -> usize {
        self.levels.iter().map(HashMap::len).sum()
    }

    /// The `(commitment, proof)` of a node; zero for unstored nodes.
    pub fn node(&self, level: usize, index: usize) -> ProofNode<PE> {
        self.levels[level]
            .get(&index)
            .copied()
            .unwrap_or((G1Aff::<PE>::zero(), G1Aff::<PE>::zero()))
    }

    /// The AMT commitment: the sum of the two top-level node commitments.
    pub fn commitment(&self) -> G1Aff<PE> {
        (self.node(0, 0).0 + self.node(0, 1).0).into_affine()
    }

    /// The proof for leaf `idx`, to check with [`AmtParams::verify`].
    pub fn get_proof(&self, idx: usize) -> Proof<PE> {
        assert!(idx >> self.depth == 0, "leaf index {idx} out of range");
        Proof::new(
            (0..self.levels())
                .map(|d| self.node(d, (idx >> (self.depth - 1 - d)) ^ 1))
                .collect(),
        )
    }
}

impl<PE, C> ProofTree<PE>
where
    PE: Pairing<G1 = Projective<C>, G1Affine = Affine<C>>,
    C: SWCurveConfig,
{
    /// Adds `diffs` (from [`AmtParams::proof_tree_diffs`]) to the tree,
    /// normalizing all touched nodes with one batch inversion.
    pub fn apply(&mut self, diffs: &[NodeDiff<PE>]) {
        let mut sums = Vec::with_capacity(2 * diffs.len());
        for diff in diffs {
            let (c, p) = self.node(diff.level, diff.index);
            sums.push(diff.commitment + c);
            sums.push(diff.proof + p);
        }
        let affine = batch_normalize(&sums);
        for (diff, pair) in diffs.iter().zip(affine.chunks_exact(2)) {
            let level = &mut self.levels[diff.level];
            if pair[0].is_zero() && pair[1].is_zero() {
                level.remove(&diff.index);
            } else {
                level.insert(diff.index, (pair[0], pair[1]));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prove_params::tests::{AMT, PE, TEST_LENGTH, TEST_LEVEL};
    use once_cell::sync::Lazy;
    use rand::Rng;

    type F = Fr<PE>;

    static POWERS: Lazy<QuotientPower<PE>> = Lazy::new(|| QuotientPower::new(&AMT));

    fn random_changes(rng: &mut impl Rng, count: usize) -> Vec<(usize, SlotChanges)> {
        (0..count)
            .map(|_| {
                let mut slots = [false; SLOT_SIZE];
                slots.iter_mut().for_each(|s| *s = rng.gen_bool(0.3));
                (rng.gen_range(0..TEST_LENGTH), slots)
            })
            .collect()
    }

    fn as_refs(changes: &[(usize, SlotChanges)]) -> impl Iterator<Item = (usize, &SlotChanges)> {
        changes.iter().map(|(i, s)| (*i, s))
    }

    /// Applies change batches incrementally and checks the result against a
    /// full rebuild after every batch.
    fn check_incremental(params: &AmtParams<PE>, powers: Option<&QuotientPower<PE>>) {
        let mut rng = rand::thread_rng();
        let mut versions = vec![[0u64; SLOT_SIZE]; TEST_LENGTH];
        let mut tree = ProofTree::new(params);
        let mut commitment = G1::<PE>::zero();

        for count in [1, 3, 10, 40] {
            let changes = random_changes(&mut rng, count);
            for (idx, slots) in &changes {
                for (v, s) in versions[*idx].iter_mut().zip(slots) {
                    *v += *s as u64;
                }
            }
            tree.apply(&params.proof_tree_diffs(as_refs(&changes), powers));
            commitment = params.apply_diffs(vec![(commitment, as_refs(&changes))])[0].into();

            let values: Vec<F> = versions.iter().map(|v| pack_slots(v).unwrap()).collect();
            let (root, built) = params.build_proof_tree(&values);
            let built = ProofTree::from_all_proofs(&built);
            assert_eq!(tree.commitment(), root);
            assert_eq!(tree.commitment(), commitment.into_affine());
            assert_eq!(tree.stored_nodes(), built.stored_nodes());
            for d in 0..tree.levels() {
                for k in 0..2 << d {
                    assert_eq!(tree.node(d, k), built.node(d, k), "d = {d}, k = {k}");
                }
            }
            if params.quotients.len() == TEST_LEVEL {
                for (idx, value) in values.iter().enumerate() {
                    assert_eq!(
                        params.verify(&root, idx, *value, &tree.get_proof(idx)),
                        Ok(())
                    );
                }
            }
        }
    }

    #[test]
    fn test_incremental_matches_rebuild_with_powers() {
        check_incremental(&AMT, Some(&POWERS));
    }

    #[test]
    fn test_incremental_matches_rebuild_without_powers() {
        check_incremental(&AMT, None);
    }

    #[test]
    fn test_incremental_reduced_prove_depth() {
        for depth in 1..TEST_LEVEL {
            let params = AMT.reduce_prove_depth(depth);
            check_incremental(&params, Some(&POWERS.reduce_prove_depth(depth)));
            check_incremental(&params, None);
        }
    }

    #[test]
    fn test_diffs_are_aggregated_and_sorted() {
        let mut rng = rand::thread_rng();
        let changes = random_changes(&mut rng, 50);
        let diffs = AMT.proof_tree_diffs(as_refs(&changes), Some(&POWERS));
        assert!(diffs
            .windows(2)
            .all(|w| (w[0].level, w[0].index) < (w[1].level, w[1].index)));
        assert_eq!(
            diffs,
            AMT.proof_tree_diffs(as_refs(&changes), None),
            "with and without quotient powers"
        );
    }

    #[test]
    fn test_duplicate_leaf_counts_twice() {
        let mut slots = [false; SLOT_SIZE];
        slots[2] = true;
        let once = AMT.proof_tree_diffs([(7, &slots)], Some(&POWERS));
        let twice = AMT.proof_tree_diffs([(7, &slots), (7, &slots)], Some(&POWERS));
        let doubled: Vec<_> = once
            .iter()
            .map(|d| NodeDiff {
                commitment: d.commitment + d.commitment,
                proof: d.proof + d.proof,
                ..d.clone()
            })
            .collect();
        assert_eq!(twice, doubled);
    }

    #[test]
    fn test_empty_changes() {
        let none: [(usize, &SlotChanges); 0] = [];
        assert!(AMT.proof_tree_diffs(none, Some(&POWERS)).is_empty());
        let unchanged = [false; SLOT_SIZE];
        assert!(AMT.proof_tree_diffs([(3, &unchanged)], None).is_empty());
    }

    #[test]
    fn test_keep_filter() {
        let mut rng = rand::thread_rng();
        let changes = random_changes(&mut rng, 30);
        let all = AMT.proof_tree_diffs(as_refs(&changes), Some(&POWERS));
        let keep = |level: usize, index: usize| (level + index) % 3 != 0;
        let kept = AMT.proof_tree_diffs_with(as_refs(&changes), Some(&POWERS), keep);
        let expected: Vec<_> = all.into_iter().filter(|d| keep(d.level, d.index)).collect();
        assert_eq!(kept, expected);
    }

    #[test]
    fn test_get_proof_matches_all_proofs() {
        let mut rng = rand::thread_rng();
        let values: Vec<F> = (0..TEST_LENGTH)
            .map(|_| F::from(rng.gen::<u64>()))
            .collect();
        let (_, all) = AMT.build_proof_tree(&values);
        let tree = ProofTree::from_all_proofs(&all);
        for idx in 0..TEST_LENGTH {
            assert_eq!(tree.get_proof(idx), all.get_proof(idx));
        }
    }
}
