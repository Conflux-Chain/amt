#[macro_use]
extern crate tracing;

pub mod commitment;
pub mod ec_algebra;
mod error;
mod power_tau;
mod proof_tree;
pub mod proofs;
mod prove_params;
mod quotient_power;
mod utils;

pub use commitment::{batch_normalize, SlotChanges};
pub use power_tau::PowerTau;
pub use proofs::{pack_slots, unpack_slots, AmtProofError, Proof};
pub use prove_params::{AmtParams, CreateMode, SLOT_SIZE};
pub use quotient_power::QuotientPower;
pub use utils::ptau_file_name;

#[cfg(not(feature = "bls12-381"))]
pub use prove_params::fast_serde_bn254;

#[cfg(feature = "bls12-381")]
pub use prove_params::fast_serde_bls12_381;
