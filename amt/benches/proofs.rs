//! Benchmarks for AMT proof trees. Run from the `amt` directory, e.g.
//! `cargo bench --features bls12-381,parallel --bench proofs`.
//!
//! Params are loaded from `./pp` at depth 5 by default. Set `AMT_BENCH_DIR`
//! and `AMT_BENCH_DEPTH` to benchmark other (already generated) params, e.g.
//! depth 16.

use amt::{
    ec_algebra::Fr, pack_slots, AmtParams, CreateMode, PowerTau, ProofTree, QuotientPower,
    SlotChanges, SLOT_SIZE,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[cfg(not(feature = "bls12-381"))]
type PE = ark_bn254::Bn254;
#[cfg(feature = "bls12-381")]
type PE = ark_bls12_381::Bls12_381;

/// Changed leaves per batch.
const SIZES: [usize; 5] = [1, 16, 256, 1024, 4096];

fn load_params() -> (AmtParams<PE>, QuotientPower<PE>) {
    let dir = std::env::var("AMT_BENCH_DIR").unwrap_or_else(|_| "./pp".into());
    let depth = std::env::var("AMT_BENCH_DEPTH").map_or(5, |d| d.parse().unwrap());
    let pp = PowerTau::<PE>::from_dir_mont(&dir, depth, false);
    let params = AmtParams::from_dir_mont(&dir, depth, depth, CreateMode::Neither, Some(&pp));
    let powers = QuotientPower::from_dir(&dir, &params, true);
    (params, powers)
}

/// `m` distinct leaves, each bumping one random slot (a typical commit).
fn random_changes(rng: &mut StdRng, n: usize, m: usize) -> Vec<(usize, SlotChanges)> {
    rand::seq::index::sample(rng, n, m)
        .into_iter()
        .map(|idx| {
            let mut slots = [false; SLOT_SIZE];
            slots[rng.gen_range(0..SLOT_SIZE)] = true;
            (idx, slots)
        })
        .collect()
}

fn random_values(rng: &mut StdRng, n: usize) -> Vec<Fr<PE>> {
    (0..n)
        .map(|_| {
            let vers: [u64; SLOT_SIZE] = std::array::from_fn(|_| rng.gen_range(0..1 << 20));
            pack_slots(&vers).unwrap()
        })
        .collect()
}

fn bench_proof_tree_update(c: &mut Criterion) {
    let (params, powers) = load_params();
    let n = params.len();
    let mut rng = StdRng::seed_from_u64(0);
    let (_, built) = params.build_proof_tree(&random_values(&mut rng, n));
    let tree = ProofTree::from_all_proofs(&built);

    let mut group = c.benchmark_group("proof_tree_update");
    for m in SIZES.into_iter().filter(|m| *m <= n) {
        let changes = random_changes(&mut rng, n, m);
        let refs = || changes.iter().map(|(i, s)| (*i, s));
        group.bench_with_input(BenchmarkId::new("diffs_with_powers", m), &m, |b, _| {
            b.iter(|| params.proof_tree_diffs(refs(), Some(&powers)))
        });
        // Scalar multiplications are slow; skip the largest batches.
        if m <= 256 {
            group.bench_with_input(BenchmarkId::new("diffs_scalar_mul", m), &m, |b, _| {
                b.iter(|| params.proof_tree_diffs(refs(), None))
            });
        }
        let diffs = params.proof_tree_diffs(refs(), Some(&powers));
        group.bench_with_input(BenchmarkId::new("apply", m), &m, |b, _| {
            b.iter_batched(
                || tree.clone(),
                |mut t| {
                    t.apply(&diffs);
                    t
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

/// Non-zero leaves of a level-1 AMT (40M keys) and of the whole root AMT.
const WEIGHTED_SIZES: [usize; 2] = [610, 65536];

fn bench_weighted(c: &mut Criterion) {
    let (params, _) = load_params();
    let n = params.len();
    let mut rng = StdRng::seed_from_u64(3);
    let mut group = c.benchmark_group("proof_tree_weighted");
    group.sample_size(10);
    for m in WEIGHTED_SIZES.into_iter().filter(|m| *m <= n) {
        let leaves = rand::seq::index::sample(&mut rng, n, m).into_vec();
        let changes: Vec<(usize, Fr<PE>)> =
            leaves.into_iter().zip(random_values(&mut rng, m)).collect();
        let idx = rng.gen_range(0..n);
        group.bench_with_input(BenchmarkId::new("path_siblings", m), &m, |b, _| {
            b.iter(|| {
                params.proof_tree_diffs_weighted(changes.iter().copied(), params.path_siblings(idx))
            })
        });
        group.bench_with_input(BenchmarkId::new("all", m), &m, |b, _| {
            b.iter(|| params.proof_tree_diffs_weighted(changes.iter().copied(), |_, _| true))
        });
    }
    group.finish();
}

fn bench_verify(c: &mut Criterion) {
    let (params, _) = load_params();
    let n = params.len();
    let mut rng = StdRng::seed_from_u64(1);
    let values = random_values(&mut rng, n);
    let (root, built) = params.build_proof_tree(&values);
    let idx = rng.gen_range(0..n);
    let proof = built.get_proof(idx);
    c.bench_function("verify", |b| {
        b.iter(|| params.verify(&root, idx, values[idx], &proof).unwrap())
    });
}

fn bench_build(c: &mut Criterion) {
    let (params, _) = load_params();
    let n = params.len();
    let mut rng = StdRng::seed_from_u64(2);
    let values = random_values(&mut rng, n);
    let mut group = c.benchmark_group("build_proof_tree");
    group.sample_size(10);
    group.bench_function("dense", |b| b.iter(|| params.build_proof_tree(&values)));
    group.finish();
}

criterion_group!(
    benches,
    bench_proof_tree_update,
    bench_weighted,
    bench_verify,
    bench_build
);
criterion_main!(benches);
