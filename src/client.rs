use async_trait::async_trait;
use tokio::sync::mpsc::Sender;

pub mod grpc;
pub mod stratum;

use crate::pow::BlockSeed;
use crate::{Error, MinerManager};

#[async_trait(?Send)]
pub trait Client {
    async fn register(&mut self) -> Result<(), Error>;
    async fn listen(&mut self, miner: &mut MinerManager) -> Result<(), Error>;
    fn get_block_channel(&self) -> Sender<BlockSeed>;
    fn flush_escrow_state(&mut self) -> Result<(), Error> {
        Ok(())
    }
}
