//! Incremental commitment updates for LVMT.
//!
//! Each AMT node value is split into `SLOT_SIZE` slots of 40 bits. Bumping
//! slot `s` of node `idx` by one changes the commitment by
//! `basis(idx) * 2^{40 * s}`, which `AmtParams` stores precomputed, so an
//! update is a handful of mixed additions followed by one batched
//! normalization.

use ark_ec::short_weierstrass::{Affine, Projective, SWCurveConfig};
use ark_ff::{Field, Zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    ec_algebra::{G1Aff, Pairing, G1},
    prove_params::SLOT_SIZE,
    AmtParams,
};

/// Slot changes of one AMT node: `changes[s] == true` means slot `s` was
/// bumped by one.
pub type SlotChanges = [bool; SLOT_SIZE];

/// With the `parallel` feature, `batch_normalize` only goes parallel from
/// this many points on. Waking the rayon pool costs ~60µs, so a smaller batch
/// is faster on the current thread (serial normalize is ~0.2µs per point).
pub const NORMALIZE_PARALLEL_THRESHOLD: usize = 1024;

/// Minimum number of commitments per rayon task in `apply_diffs`
/// (~2µs of mixed additions each for a typical update).
pub const APPLY_DIFFS_MIN_LEN: usize = 64;

impl<PE: Pairing> AmtParams<PE> {
    /// Commitment difference caused by `changes`, a list of
    /// `(node index, slot changes)`.
    pub fn commitment_diff<'a>(
        &self,
        changes: impl IntoIterator<Item = (usize, &'a SlotChanges)>,
    ) -> G1<PE> {
        let mut acc = G1::<PE>::zero();
        self.accumulate_changes(&mut acc, changes);
        acc
    }

    fn accumulate_changes<'a>(
        &self,
        acc: &mut G1<PE>,
        changes: impl IntoIterator<Item = (usize, &'a SlotChanges)>,
    ) {
        for (idx, slots) in changes {
            let (basis, powers) = self.basis_power_at(idx);
            if slots[0] {
                *acc += basis;
            }
            for (changed, power) in slots[1..].iter().zip(powers) {
                if *changed {
                    *acc += power;
                }
            }
        }
    }
}

impl<PE, C> AmtParams<PE>
where
    PE: Pairing<G1 = Projective<C>, G1Affine = Affine<C>>,
    C: SWCurveConfig,
{
    /// Applies a batch of commitment updates and returns the new commitments
    /// in affine form, in input order.
    ///
    /// Each entry is `(old commitment, changes)` where `changes` lists
    /// `(node index, slot changes)` for that commitment. All results are
    /// normalized with a single [`batch_normalize`].
    pub fn apply_diffs<'a, I>(&self, updates: Vec<(G1<PE>, I)>) -> Vec<G1Aff<PE>>
    where
        I: IntoIterator<Item = (usize, &'a SlotChanges)> + Send,
    {
        let apply = |(mut commitment, changes): (G1<PE>, I)| {
            self.accumulate_changes(&mut commitment, changes);
            commitment
        };

        #[cfg(feature = "parallel")]
        let updated: Vec<G1<PE>> = updates
            .into_par_iter()
            .with_min_len(APPLY_DIFFS_MIN_LEN)
            .map(apply)
            .collect();
        #[cfg(not(feature = "parallel"))]
        let updated: Vec<G1<PE>> = updates.into_iter().map(apply).collect();

        batch_normalize(&updated)
    }
}

/// Converts projective points to affine with Montgomery's batch inversion.
///
/// Same result as `CurveGroup::normalize_batch`. With the `parallel` feature,
/// `ark-ff` splits the inversion into one chunk per thread with no minimum
/// size, so a small batch pays rayon wake-up plus one inversion per thread
/// (e.g. 75 points: ~63µs instead of ~19µs serially on a 32-thread CPU).
/// Batches below `NORMALIZE_PARALLEL_THRESHOLD` are therefore normalized on
/// the current thread; larger ones still use the parallel ark implementation.
pub fn batch_normalize<C: SWCurveConfig>(points: &[Projective<C>]) -> Vec<Affine<C>> {
    #[cfg(feature = "parallel")]
    if points.len() >= NORMALIZE_PARALLEL_THRESHOLD {
        return ark_ec::CurveGroup::normalize_batch(points);
    }

    normalize_serial(points)
}

fn normalize_serial<C: SWCurveConfig>(points: &[Projective<C>]) -> Vec<Affine<C>> {
    let mut z_inv: Vec<C::BaseField> = points.iter().map(|p| p.z).collect();
    serial_batch_inversion(&mut z_inv);

    points
        .iter()
        .zip(z_inv)
        .map(|(p, z_inv)| {
            if p.is_zero() {
                Affine::identity()
            } else {
                // Jacobian coordinates: (X / Z^2, Y / Z^3).
                let z_inv2 = z_inv.square();
                Affine::new_unchecked(p.x * z_inv2, p.y * z_inv2 * z_inv)
            }
        })
        .collect()
}

/// Inverts every non-zero element in place with a single field inversion
/// (Montgomery's trick). Zero elements are left unchanged.
fn serial_batch_inversion<F: Field>(values: &mut [F]) {
    let mut prefix = Vec::with_capacity(values.len());
    let mut acc = F::one();
    for v in values.iter().filter(|v| !v.is_zero()) {
        acc *= v;
        prefix.push(acc);
    }

    // `acc` is a product of non-zero elements (or one), so it is invertible.
    let mut inv = acc.inverse().expect("product of non-zero elements");

    // Walk backwards: before the update `inv` = 1 / (v_0 * ... * v_i).
    let prev_prefix = prefix.into_iter().rev().skip(1).chain(Some(F::one()));
    for (v, prev) in values
        .iter_mut()
        .rev()
        .filter(|v| !v.is_zero())
        .zip(prev_prefix)
    {
        let next_inv = inv * *v;
        *v = inv * prev;
        inv = next_inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ec_algebra::{CurveGroup, UniformRand},
        prove_params::tests::{AMT, PE, TEST_LENGTH},
    };

    type G1P = G1<PE>;

    fn naive_diff(changes: &[(usize, SlotChanges)]) -> G1P {
        let mut acc = G1P::zero();
        for (idx, slots) in changes {
            let powers = AMT.get_basis_power_at(*idx);
            for (changed, power) in slots.iter().zip(powers) {
                if *changed {
                    acc += power;
                }
            }
        }
        acc
    }

    fn random_changes(rng: &mut impl rand::Rng, nodes: usize) -> Vec<(usize, SlotChanges)> {
        (0..nodes)
            .map(|_| {
                let idx = rng.gen_range(0..TEST_LENGTH);
                let mut slots = [false; SLOT_SIZE];
                slots.iter_mut().for_each(|s| *s = rng.gen_bool(0.4));
                (idx, slots)
            })
            .collect()
    }

    #[test]
    fn test_basis_power_at_matches_get_basis_power_at() {
        for idx in 0..TEST_LENGTH {
            let (basis, powers) = AMT.basis_power_at(idx);
            let owned = AMT.get_basis_power_at(idx);
            assert_eq!(owned[0], *basis);
            assert_eq!(owned[1..], powers[..]);
        }
    }

    #[test]
    fn test_commitment_diff_matches_naive() {
        let mut rng = rand::thread_rng();
        for nodes in [0, 1, 3, 20] {
            let changes = random_changes(&mut rng, nodes);
            let diff = AMT.commitment_diff(changes.iter().map(|(i, s)| (*i, s)));
            assert_eq!(diff, naive_diff(&changes));
        }
    }

    #[test]
    fn test_apply_diffs_matches_naive() {
        let mut rng = rand::thread_rng();
        // Cover both the serial and the parallel paths.
        for count in [0, 1, 7, APPLY_DIFFS_MIN_LEN * 3 + 5] {
            let bases: Vec<G1P> = (0..count)
                .map(|i| match i % 3 {
                    0 => G1P::zero(),
                    1 => G1P::rand(&mut rng).into_affine().into(),
                    _ => G1P::rand(&mut rng),
                })
                .collect();
            let changes: Vec<Vec<(usize, SlotChanges)>> = (0..count)
                .map(|i| random_changes(&mut rng, i % 5))
                .collect();

            let updates = bases
                .iter()
                .zip(&changes)
                .map(|(base, ch)| (*base, ch.iter().map(|(i, s)| (*i, s))))
                .collect();
            let result = AMT.apply_diffs(updates);

            let expected: Vec<_> = bases
                .iter()
                .zip(&changes)
                .map(|(base, ch)| (*base + naive_diff(ch)).into_affine())
                .collect();
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_batch_normalize_matches_ark() {
        let mut rng = rand::thread_rng();
        for len in [
            0,
            1,
            2,
            100,
            NORMALIZE_PARALLEL_THRESHOLD - 1,
            NORMALIZE_PARALLEL_THRESHOLD,
            3 * NORMALIZE_PARALLEL_THRESHOLD + 3,
        ] {
            let points: Vec<G1P> = (0..len)
                .map(|i| {
                    // Mix in points at infinity and points with z == 1.
                    match i % 7 {
                        0 => G1P::zero(),
                        1 => G1P::rand(&mut rng).into_affine().into(),
                        _ => G1P::rand(&mut rng),
                    }
                })
                .collect();
            assert_eq!(
                batch_normalize(&points),
                G1P::normalize_batch(&points),
                "len = {len}"
            );
        }
    }

    #[test]
    fn test_serial_batch_inversion() {
        use crate::ec_algebra::Fr;
        let mut rng = rand::thread_rng();
        let mut values: Vec<Fr<PE>> = (0..20)
            .map(|i| {
                if i % 4 == 0 {
                    Fr::<PE>::zero()
                } else {
                    Fr::<PE>::rand(&mut rng)
                }
            })
            .collect();
        let expected: Vec<_> = values
            .iter()
            .map(|v| v.inverse().unwrap_or(Fr::<PE>::zero()))
            .collect();
        serial_batch_inversion(&mut values);
        assert_eq!(values, expected);

        let mut zeros = vec![Fr::<PE>::zero(); 3];
        serial_batch_inversion(&mut zeros);
        assert!(zeros.iter().all(|v| v.is_zero()));
    }
}
