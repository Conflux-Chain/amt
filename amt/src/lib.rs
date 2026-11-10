#[macro_use]
extern crate tracing;

pub mod commitment;
pub mod ec_algebra;
mod error;
mod power_tau;
mod proofs;
mod prove_params;
mod utils;

pub use commitment::{batch_normalize, SlotChanges};
pub use power_tau::PowerTau;
pub use proofs::AmtProofError;
pub use prove_params::{AmtParams, CreateMode, SLOT_SIZE};
pub use utils::ptau_file_name;

#[cfg(not(feature = "bls12-381"))]
pub use prove_params::fast_serde_bn254;

#[cfg(feature = "bls12-381")]
pub use prove_params::fast_serde_bls12_381;
