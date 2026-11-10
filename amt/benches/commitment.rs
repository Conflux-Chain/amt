//! Benchmarks for the commit hot path. Run from the `amt` directory, e.g.
//! `cargo bench --features bls12-381,parallel --bench commitment`
//! (test params are loaded from `./pp`).

use amt::{
    batch_normalize,
    ec_algebra::{CurveGroup, UniformRand, G1},
    AmtParams, CreateMode, PowerTau, SlotChanges, SLOT_SIZE,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[cfg(not(feature = "bls12-381"))]
type PE = ark_bn254::Bn254;
#[cfg(feature = "bls12-381")]
type PE = ark_bls12_381::Bls12_381;

const DEPTH: usize = 5;
const SIZES: [usize; 7] = [16, 64, 128, 256, 512, 1024, 4096];

fn load_params() -> AmtParams<PE> {
    let pp = PowerTau::<PE>::from_dir_mont("./pp", DEPTH, false);
    AmtParams::from_dir_mont("./pp", DEPTH, DEPTH, CreateMode::Neither, Some(&pp))
}

fn random_points(rng: &mut StdRng, n: usize) -> Vec<G1<PE>> {
    (0..n).map(|_| G1::<PE>::rand(rng)).collect()
}

/// One commitment per entry, each with `nodes` changed nodes and 1-2 slots per node.
fn random_changes(rng: &mut StdRng, n: usize, nodes: usize) -> Vec<Vec<(usize, SlotChanges)>> {
    (0..n)
        .map(|_| {
            (0..nodes)
                .map(|_| {
                    let mut slots = [false; SLOT_SIZE];
                    slots[rng.gen_range(0..SLOT_SIZE)] = true;
                    slots[SLOT_SIZE - 1] = true;
                    (rng.gen_range(0..1 << DEPTH), slots)
                })
                .collect()
        })
        .collect()
}

fn bench_normalize(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut group = c.benchmark_group("normalize");
    for n in SIZES {
        let points = random_points(&mut rng, n);
        group.bench_with_input(
            BenchmarkId::new("ark_normalize_batch", n),
            &points,
            |b, p| b.iter(|| G1::<PE>::normalize_batch(p)),
        );
        group.bench_with_input(
            BenchmarkId::new("amt_batch_normalize", n),
            &points,
            |b, p| b.iter(|| batch_normalize(p)),
        );
    }
    group.finish();
}

fn bench_apply_diffs(c: &mut Criterion) {
    let params = load_params();
    let mut rng = StdRng::seed_from_u64(1);
    let mut group = c.benchmark_group("apply_diffs");
    for n in SIZES {
        let bases = random_points(&mut rng, n);
        let changes = random_changes(&mut rng, n, 3);

        // Baseline: what cfx-storage2 does today (serial diffs + ark normalize_batch).
        group.bench_with_input(BenchmarkId::new("baseline", n), &n, |b, _| {
            b.iter(|| {
                let sums: Vec<G1<PE>> = bases
                    .iter()
                    .zip(&changes)
                    .map(|(base, ch)| {
                        *base + params.commitment_diff(ch.iter().map(|(i, s)| (*i, s)))
                    })
                    .collect();
                G1::<PE>::normalize_batch(&sums)
            })
        });
        group.bench_with_input(BenchmarkId::new("apply_diffs", n), &n, |b, _| {
            b.iter(|| {
                let updates = bases
                    .iter()
                    .zip(&changes)
                    .map(|(base, ch)| (*base, ch.iter().map(|(i, s)| (*i, s))))
                    .collect();
                params.apply_diffs(updates)
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_normalize, bench_apply_diffs);
criterion_main!(benches);
