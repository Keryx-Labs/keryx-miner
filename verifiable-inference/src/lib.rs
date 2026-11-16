//! Verification of inference trace openings.

pub mod answer;
pub mod attn;
pub mod attn_blocks;
pub mod block_task;
pub mod chain;
#[cfg(any(test, feature = "testkit"))]
pub mod chainkit;
pub mod dot;
pub mod draw;
pub mod f16;
pub mod manifest;
pub mod merkle;
pub mod opening;
pub mod prefix;
pub mod q8;
pub mod quant;
pub mod recur;
pub mod seq;
pub mod shape;
pub mod support;
#[cfg(any(test, feature = "testkit"))]
pub mod testkit;
pub mod verify;
