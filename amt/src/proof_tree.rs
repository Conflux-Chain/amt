//! Building the AMT proof tree.
//!
//! The proof tree has one level per prove depth `d` in `0..L` (`L` is
//! `quotients.len()`); level `d` has `2^{d+1}` nodes and node `k` covers the
//! contiguous leaf range `k * n / 2^{d+1} ..`. Each node holds
//!   - the commitment `Σ values[s] * basis[s]` over its range, and
//!   - the quotient proof `Σ values[s] * quotients[d][s]` over its range.
//!
//! See [`crate::proofs`] for how a proof is taken from the tree and checked.

use ark_ec::pairing::Pairing;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    ec_algebra::{CurveGroup, Fr, G1Aff, VariableBaseMSM, Zero, G1},
    proofs::AllProofs,
    AmtParams,
};

impl<PE: Pairing> AmtParams<PE> {
    /// Builds the whole proof tree of `values` from scratch, for initializing
    /// or rebuilding an AMT. Returns the AMT commitment and the tree.
    ///
    /// Costs one size-`n` MSM worth of work per level for proofs, plus one
    /// for the bottom commitments. Use it for dense AMTs; for sparse ones,
    /// adding the non-zero leaves incrementally is cheaper.
    ///
    /// # Panics
    ///
    /// If `values.len()` differs from the AMT length.
    pub fn build_proof_tree(&self, values: &[Fr<PE>]) -> (G1Aff<PE>, AllProofs<PE>) {
        let n = self.basis.len();
        assert_eq!(values.len(), n, "expected one value per AMT leaf");
        let levels = self.quotients.len();
        assert!(levels >= 1 && n >> levels >= 1);

        // Proofs: one chunked MSM against `quotients[d]` per level.
        let mut proofs: Vec<Vec<G1<PE>>> = (0..levels)
            .map(|d| chunked_msm(&self.quotients[d], values, n >> (d + 1)))
            .collect();

        // Commitments: chunked MSM at the bottom level, then pairwise sums
        // up to the top level (2 nodes) and the root.
        let mut commitments: Vec<Vec<G1<PE>>> = Vec::with_capacity(levels);
        commitments.push(chunked_msm(&self.basis, values, n >> levels));
        for _ in 1..levels {
            let below = commitments.last().unwrap();
            let above = below.chunks_exact(2).map(|p| p[0] + p[1]).collect();
            commitments.push(above);
        }
        commitments.reverse();
        let root = commitments[0][0] + commitments[0][1];

        // Normalize every node with a single batch inversion.
        let mut flat: Vec<G1<PE>> = Vec::with_capacity(4 * n);
        flat.push(root);
        for level in commitments.iter_mut().chain(proofs.iter_mut()) {
            flat.append(level);
        }
        let mut flat = G1::<PE>::normalize_batch(&flat).into_iter();

        let root = flat.next().unwrap();
        let mut take_levels = || -> Vec<Vec<G1Aff<PE>>> {
            (0..levels)
                .map(|d| flat.by_ref().take(2 << d).collect())
                .collect()
        };
        let commitments = take_levels();
        let proofs = take_levels();

        let tree = AllProofs {
            commitments,
            proofs,
            input_len: n,
            batch_size: n >> levels,
        };
        (root, tree)
    }
}

/// `bases.len() / chunk` MSMs over consecutive chunks. All-zero chunks are
/// skipped, which makes sparse inputs cheap.
fn chunked_msm<G: VariableBaseMSM>(
    bases: &[G::MulBase],
    scalars: &[G::ScalarField],
    chunk: usize,
) -> Vec<G> {
    #[cfg(feature = "parallel")]
    let pairs = bases
        .par_chunks(chunk)
        .zip(scalars.par_chunks(chunk))
        .with_min_len(min_len(chunk));
    #[cfg(not(feature = "parallel"))]
    let pairs = bases.chunks(chunk).zip(scalars.chunks(chunk));

    pairs
        .map(|(b, s)| {
            if s.iter().all(Zero::is_zero) {
                G::zero()
            } else {
                G::msm_unchecked(b, s)
            }
        })
        .collect()
}

/// Groups small chunks into rayon tasks of at least ~64 points.
#[cfg(feature = "parallel")]
fn min_len(chunk: usize) -> usize {
    (64 / chunk).max(1)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{
        ec_algebra::UniformRand,
        prove_params::tests::{AMT, PE, TEST_LENGTH, TEST_LEVEL},
    };
    use rand::Rng;

    type F = Fr<PE>;

    /// Every node straight from the definition, one MSM per node.
    pub(crate) fn naive_tree(
        params: &AmtParams<PE>,
        values: &[F],
    ) -> Vec<Vec<(G1Aff<PE>, G1Aff<PE>)>> {
        (0..params.quotients.len())
            .map(|d| {
                let chunk = TEST_LENGTH >> (d + 1);
                (0..1 << (d + 1))
                    .map(|k| {
                        let range = k * chunk..(k + 1) * chunk;
                        let c = G1::<PE>::msm(&params.basis[range.clone()], &values[range.clone()])
                            .unwrap();
                        let p = G1::<PE>::msm(&params.quotients[d][range.clone()], &values[range])
                            .unwrap();
                        (c.into_affine(), p.into_affine())
                    })
                    .collect()
            })
            .collect()
    }

    fn check_against_naive(params: &AmtParams<PE>, values: &[F]) {
        let (root, tree) = params.build_proof_tree(values);
        assert_eq!(
            root,
            G1::<PE>::msm(&params.basis, values).unwrap().into_affine()
        );
        let naive = naive_tree(params, values);
        assert_eq!(tree.commitments.len(), naive.len());
        for (d, level) in naive.iter().enumerate() {
            let commitments: Vec<_> = level.iter().map(|x| x.0).collect();
            let proofs: Vec<_> = level.iter().map(|x| x.1).collect();
            assert_eq!(tree.commitments[d], commitments, "commitments, d = {d}");
            assert_eq!(tree.proofs[d], proofs, "proofs, d = {d}");
        }
    }

    #[test]
    fn test_build_matches_naive_dense() {
        let mut rng = rand::thread_rng();
        let values: Vec<F> = (0..TEST_LENGTH).map(|_| F::rand(&mut rng)).collect();
        check_against_naive(&AMT, &values);
    }

    #[test]
    fn test_build_matches_naive_sparse() {
        let mut rng = rand::thread_rng();
        for nonzero in [0, 1, 3, TEST_LENGTH / 4] {
            let mut values = vec![F::zero(); TEST_LENGTH];
            for _ in 0..nonzero {
                values[rng.gen_range(0..TEST_LENGTH)] = F::rand(&mut rng);
            }
            check_against_naive(&AMT, &values);
        }
    }

    #[test]
    fn test_build_reduced_prove_depth() {
        let mut rng = rand::thread_rng();
        let values: Vec<F> = (0..TEST_LENGTH).map(|_| F::rand(&mut rng)).collect();
        for depth in 1..TEST_LEVEL {
            let reduced = AMT.reduce_prove_depth(depth);
            check_against_naive(&reduced, &values);
            let (_, tree) = reduced.build_proof_tree(&values);
            assert_eq!(tree.batch_size, TEST_LENGTH >> depth);
        }
    }

    #[test]
    fn test_build_then_verify_every_leaf() {
        let mut rng = rand::thread_rng();
        let values: Vec<F> = (0..TEST_LENGTH).map(|_| F::rand(&mut rng)).collect();
        let (root, tree) = AMT.build_proof_tree(&values);
        for (idx, value) in values.iter().enumerate() {
            let proof = tree.get_proof(idx);
            assert_eq!(AMT.verify(&root, idx, *value, &proof), Ok(()));
        }
    }
}
