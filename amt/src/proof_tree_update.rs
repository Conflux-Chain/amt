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

use std::{collections::HashMap, ops::Range};

use ark_ec::{
    pairing::Pairing,
    short_weierstrass::{Affine, Projective, SWCurveConfig},
};
use ark_std::cfg_into_iter;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    commitment::{batch_normalize, SlotChanges},
    ec_algebra::{AffineRepr, CurveGroup, Fr, G1Aff, VariableBaseMSM, Zero, G1},
    proofs::{pack_slots, AllProofs, Proof},
    AmtParams, QuotientPower, SLOT_SIZE,
};

/// With the `parallel` feature, [`AmtParams::proof_tree_diffs`] with quotient
/// powers spreads levels over threads only from this many changed leaves on.
/// Paths doing scalar multiplications always go parallel.
pub const PROOF_DIFFS_PARALLEL_THRESHOLD: usize = 64;

/// [`AmtParams::proof_tree_diffs_weighted`] splits MSMs into chunks of at
/// most this many points, so that large nodes spread over threads.
const WEIGHTED_MSM_CHUNK: usize = 4096;

/// A touched node in [`AmtParams::proof_tree_diffs_weighted`].
#[derive(Clone)]
struct PlannedNode {
    index: usize,
    /// Range of the changed leaves below the node.
    leaves: Range<usize>,
    /// Range of the touched children in the level below.
    children: Range<usize>,
    keep: bool,
    /// The commitment is the sum of the children commitments.
    summed: bool,
}

impl PlannedNode {
    /// Whether the commitment is computed.
    fn known(&self) -> bool {
        self.keep || self.summed
    }
}

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
    /// this is [`Self::proof_tree_diffs_weighted`] with
    /// `pack_slots(changed slots)` as the deltas, which costs about one
    /// scalar multiplication per changed leaf and level.
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
        let Some(powers) = powers else {
            let weighted = changes
                .into_iter()
                .map(|(idx, slots)| (idx, slot_scalar::<PE>(slots)));
            return self.proof_tree_diffs_weighted(weighted, keep);
        };
        let levels = self.quotients.len();
        assert_eq!(
            powers.levels(),
            levels,
            "quotient powers do not match the prove depth"
        );
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
        let level_proofs = |d: usize| -> Vec<(usize, G1<PE>)> {
            merge_sorted(
                changes
                    .iter()
                    .map(|(idx, slots)| (idx >> shift(d), *idx, *slots))
                    .filter(|(node, _, _)| keep(d, *node))
                    .map(|(node, idx, slots)| {
                        let (q, qp) = self.quotient_power_at(powers, d, idx);
                        let mut acc = G1::<PE>::zero();
                        add_slot_points::<PE>(&mut acc, q, qp, slots);
                        (node, acc)
                    }),
            )
        };
        // Waking the rayon pool costs ~20µs, more than a small batch of point
        // additions takes.
        #[cfg(feature = "parallel")]
        let proofs: Vec<_> = if changes.len() >= PROOF_DIFFS_PARALLEL_THRESHOLD {
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

impl<PE: Pairing> AmtParams<PE> {
    /// Like [`Self::proof_tree_diffs_with`], but leaf `idx` gets an arbitrary
    /// `delta` added to its value instead of one per changed slot: e.g.
    /// [`pack_slots`] of per-slot increments, its negation to roll changes
    /// back, or a whole leaf value to compute nodes from scratch. A leaf
    /// listed several times gets the sum of its deltas.
    ///
    /// Each kept node costs one MSM over the changed leaves below it for its
    /// proof, and one for its commitment unless the commitments of its
    /// changed children are computed anyway. With
    /// [`Self::path_siblings`] as `keep`, every leaf is in exactly one kept
    /// node, so the total is two MSMs worth of work over the changed leaves.
    pub fn proof_tree_diffs_weighted(
        &self,
        changes: impl IntoIterator<Item = (usize, Fr<PE>)>,
        keep: impl Fn(usize, usize) -> bool + Sync,
    ) -> Vec<NodeDiff<PE>> {
        let levels = self.quotients.len();
        let depth = ark_std::log2(self.len()) as usize;

        let mut changes: Vec<(usize, Fr<PE>)> = changes.into_iter().collect();
        for (idx, _) in &changes {
            assert!(*idx < self.len(), "leaf index {idx} out of range");
        }
        changes.sort_unstable_by_key(|(idx, _)| *idx);
        let (leaves, deltas): (Vec<usize>, Vec<Fr<PE>>) = merge_sorted(changes)
            .into_iter()
            .filter(|(_, delta)| !delta.is_zero())
            .unzip();
        if leaves.is_empty() {
            return Vec::new();
        }

        // Plan, bottom-up: the touched nodes of every level with the range
        // of `leaves` below them (sorted leaves give contiguous ranges), and
        // how their commitments are found: summed from the children if all
        // changed children are known, else by an MSM if the node is kept.
        let mut plan: Vec<Vec<PlannedNode>> = vec![Vec::new(); levels];
        for d in (0..levels).rev() {
            let shift = depth - 1 - d;
            let mut nodes: Vec<PlannedNode> = Vec::new();
            for (i, idx) in leaves.iter().enumerate() {
                match nodes.last_mut() {
                    Some(node) if node.index == idx >> shift => node.leaves.end = i + 1,
                    _ => nodes.push(PlannedNode {
                        index: idx >> shift,
                        leaves: i..i + 1,
                        children: 0..0,
                        keep: false,
                        summed: false,
                    }),
                }
            }
            if let Some(below) = plan.get(d + 1) {
                let mut start = 0;
                for node in &mut nodes {
                    let count = below[start..]
                        .iter()
                        .take_while(|child| child.index >> 1 == node.index)
                        .count();
                    node.children = start..start + count;
                    node.summed = below[node.children.clone()].iter().all(PlannedNode::known);
                    start += count;
                }
            }
            for node in &mut nodes {
                node.keep = keep(d, node.index);
            }
            plan[d] = nodes;
        }

        // Every MSM (kept proofs, and commitments that are not summed), split
        // into chunks so that all threads stay busy.
        let mut tasks: Vec<(&[G1Aff<PE>], Range<usize>)> = Vec::new();
        for (d, nodes) in plan.iter().enumerate() {
            for node in nodes.iter().filter(|node| node.keep) {
                tasks.push((&self.quotients[d], node.leaves.clone()));
                if !node.summed {
                    tasks.push((&self.basis, node.leaves.clone()));
                }
            }
        }
        let chunks: Vec<_> = tasks
            .into_iter()
            .enumerate()
            .flat_map(|(task, (bases, range))| {
                range.clone().step_by(WEIGHTED_MSM_CHUNK).map(move |start| {
                    (
                        task,
                        bases,
                        start..range.end.min(start + WEIGHTED_MSM_CHUNK),
                    )
                })
            })
            .collect();
        let sums = cfg_into_iter!(chunks).map(|(task, bases, range)| {
            // `Σ bases[leaf] * delta` over the chunk.
            let scalars = &deltas[range.clone()];
            let sum = match &leaves[range] {
                [idx] => bases[*idx] * scalars[0],
                idxs => {
                    let points: Vec<G1Aff<PE>> = idxs.iter().map(|idx| bases[*idx]).collect();
                    G1::<PE>::msm_unchecked(&points, scalars)
                }
            };
            (task, sum)
        });
        let mut results = merge_sorted(sums.collect::<Vec<_>>())
            .into_iter()
            .map(|(_, sum)| sum);

        // Collect the results in task order, summing commitments bottom-up.
        let mut proofs: Vec<Vec<G1<PE>>> = vec![Vec::new(); levels];
        let mut commitments: Vec<Vec<Option<G1<PE>>>> = vec![Vec::new(); levels];
        for (d, nodes) in plan.iter().enumerate() {
            for node in nodes {
                if node.keep {
                    proofs[d].push(results.next().unwrap());
                }
                commitments[d].push((node.keep && !node.summed).then(|| results.next().unwrap()));
            }
        }
        debug_assert!(results.next().is_none());
        for d in (0..levels - 1).rev() {
            let (above, below) = commitments.split_at_mut(d + 1);
            for (node, commitment) in plan[d].iter().zip(&mut above[d]) {
                if node.summed {
                    let children = below[0][node.children.clone()].iter();
                    *commitment = Some(children.map(|c| c.unwrap()).sum());
                }
            }
        }

        let mut diffs = Vec::new();
        for (d, nodes) in plan.iter().enumerate() {
            let kept = nodes
                .iter()
                .zip(&commitments[d])
                .filter(|(node, _)| node.keep);
            for ((node, commitment), proof) in kept.zip(&proofs[d]) {
                diffs.push(NodeDiff {
                    level: d,
                    index: node.index,
                    commitment: commitment.expect("kept nodes are computed"),
                    proof: *proof,
                });
            }
        }
        diffs
    }

    /// A `keep` filter for the nodes making up the proof of leaf `idx`: at
    /// every level, the sibling of the node containing `idx`.
    pub fn path_siblings(&self, idx: usize) -> impl Fn(usize, usize) -> bool + Copy + Send + Sync {
        assert!(idx < self.len(), "leaf index {idx} out of range");
        let depth = ark_std::log2(self.len()) as usize;
        move |level, index| index == (idx >> (depth - 1 - level)) ^ 1
    }

    /// The proof of leaf `idx` in the AMT whose non-zero leaves are `values`
    /// (`(index, value)`, summed if an index repeats), computed directly
    /// from the values without a proof tree. Check it with [`Self::verify`]
    /// against the commitment of `values`.
    pub fn prove_sparse(
        &self,
        values: impl IntoIterator<Item = (usize, Fr<PE>)>,
        idx: usize,
    ) -> Proof<PE> {
        let mut nodes = vec![[G1::<PE>::zero(); 2]; self.quotients.len()];
        for diff in self.proof_tree_diffs_weighted(values, self.path_siblings(idx)) {
            nodes[diff.level] = [diff.commitment, diff.proof];
        }
        let affine = G1::<PE>::normalize_batch(nodes.as_flattened());
        Proof::new(affine.chunks_exact(2).map(|p| (p[0], p[1])).collect())
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
    use crate::{
        proofs::SLOT_BITS,
        prove_params::tests::{AMT, PE, TEST_LENGTH, TEST_LEVEL},
    };
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

    fn random_value(rng: &mut impl Rng) -> F {
        pack_slots(&std::array::from_fn(|_| rng.gen_range(0..1 << SLOT_BITS))).unwrap()
    }

    fn sparse_values(rng: &mut impl Rng, count: usize) -> Vec<(usize, F)> {
        (0..count)
            .map(|_| (rng.gen_range(0..TEST_LENGTH), random_value(rng)))
            .collect()
    }

    fn dense(values: &[(usize, F)]) -> Vec<F> {
        let mut out = vec![F::zero(); TEST_LENGTH];
        for (idx, v) in values {
            out[*idx] += v;
        }
        out
    }

    fn assert_trees_eq(a: &ProofTree<PE>, b: &ProofTree<PE>) {
        assert_eq!(a.levels(), b.levels());
        assert_eq!(a.stored_nodes(), b.stored_nodes());
        for d in 0..a.levels() {
            for k in 0..2 << d {
                assert_eq!(a.node(d, k), b.node(d, k), "d = {d}, k = {k}");
            }
        }
    }

    #[test]
    fn test_weighted_matches_repeated_slot_changes() {
        let mut rng = rand::thread_rng();
        let mut weighted = Vec::new();
        let mut repeated = Vec::new();
        for _ in 0..20 {
            let idx = rng.gen_range(0..TEST_LENGTH);
            let counts: [u64; SLOT_SIZE] = std::array::from_fn(|_| rng.gen_range(0..4));
            weighted.push((idx, pack_slots::<F>(&counts).unwrap()));
            for round in 0..3 {
                repeated.push((idx, counts.map(|c| c > round)));
            }
        }
        let keep = |level: usize, index: usize| (level + index) % 3 != 1;
        for keep in [
            &(|_, _| true) as &(dyn Fn(usize, usize) -> bool + Sync),
            &keep,
        ] {
            assert_eq!(
                AMT.proof_tree_diffs_weighted(weighted.iter().copied(), keep),
                AMT.proof_tree_diffs_with(as_refs(&repeated), Some(&POWERS), keep)
            );
        }
    }

    #[test]
    fn test_weighted_from_zero_matches_build() {
        let mut rng = rand::thread_rng();
        for depth in 1..=TEST_LEVEL {
            let params = AMT.reduce_prove_depth(depth);
            for count in [1, 5, TEST_LENGTH, 3 * TEST_LENGTH] {
                let values = sparse_values(&mut rng, count);
                let mut tree = ProofTree::new(&params);
                tree.apply(&params.proof_tree_diffs_weighted(values.iter().copied(), |_, _| true));
                let (root, built) = params.build_proof_tree(&dense(&values));
                assert_eq!(tree.commitment(), root);
                assert_trees_eq(&tree, &ProofTree::from_all_proofs(&built));
            }
        }
    }

    #[test]
    fn test_prove_sparse() {
        let mut rng = rand::thread_rng();
        for count in [0, 1, 7, TEST_LENGTH] {
            let values = sparse_values(&mut rng, count);
            let dense = dense(&values);
            let (root, built) = AMT.build_proof_tree(&dense);
            for (idx, value) in dense.iter().enumerate() {
                let proof = AMT.prove_sparse(values.iter().copied(), idx);
                assert_eq!(proof, built.get_proof(idx));
                assert_eq!(AMT.verify(&root, idx, *value, &proof), Ok(()));

                let diffs =
                    AMT.proof_tree_diffs_weighted(values.iter().copied(), AMT.path_siblings(idx));
                assert!(diffs.len() <= TEST_LEVEL);
                assert!(diffs
                    .iter()
                    .all(|d| AMT.path_siblings(idx)(d.level, d.index)));
            }
        }
    }

    #[test]
    fn test_weighted_keep_filter() {
        let mut rng = rand::thread_rng();
        let values = sparse_values(&mut rng, 30);
        let all = AMT.proof_tree_diffs_weighted(values.iter().copied(), |_, _| true);
        for keep in [
            &(|level: usize, index: usize| (level + index) % 3 != 0) as &(dyn Fn(_, _) -> _ + Sync),
            &AMT.path_siblings(5),
            &|level, _| level == TEST_LEVEL - 1,
            &|level, _| level == 0,
            &|_, _| false,
        ] {
            let expected: Vec<_> = all
                .iter()
                .filter(|d| keep(d.level, d.index))
                .cloned()
                .collect();
            assert_eq!(
                AMT.proof_tree_diffs_weighted(values.iter().copied(), keep),
                expected
            );
        }
    }

    #[test]
    fn test_weighted_negation_rolls_back() {
        let mut rng = rand::thread_rng();
        let base = sparse_values(&mut rng, 10);
        let mut tree = ProofTree::new(&AMT);
        tree.apply(&AMT.proof_tree_diffs_weighted(base.iter().copied(), |_, _| true));
        let original = tree.clone();

        let deltas = sparse_values(&mut rng, 10);
        tree.apply(&AMT.proof_tree_diffs_weighted(deltas.iter().copied(), |_, _| true));
        let negated = deltas.iter().map(|(idx, v)| (*idx, -*v));
        tree.apply(&AMT.proof_tree_diffs_weighted(negated, |_, _| true));
        assert_trees_eq(&tree, &original);

        // Removing every value leaves no stored node.
        let negated = base.iter().map(|(idx, v)| (*idx, -*v));
        tree.apply(&AMT.proof_tree_diffs_weighted(negated, |_, _| true));
        assert_eq!(tree.stored_nodes(), 0);
    }

    #[test]
    fn test_weighted_cancelling_deltas() {
        let v = F::from(12345u64);
        assert!(AMT
            .proof_tree_diffs_weighted([(3, v), (3, -v)], |_, _| true)
            .is_empty());
        assert!(AMT
            .proof_tree_diffs_weighted([(3, F::zero())], |_, _| true)
            .is_empty());
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
