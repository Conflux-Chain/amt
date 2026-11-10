#!/bin/bash
set -e
# rustfmt comes from the toolchain pinned in rust-toolchain.toml.
cargo fmt --all $@
