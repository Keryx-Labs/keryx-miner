use super::statum_codec::{StratumCommand, StratumLine, StratumLinePayload};
use crate::Error;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

const MAX_PAYLOAD_BYTES: usize = 1_048_576;

pub(super) struct AiRequest {
    pub task_id: String,
    pub request_hash: String,
    pub model_hex: String,
    pub model_id: [u8; 32],
    pub prompt: String,
    pub max_tokens: usize,
}

impl AiRequest {
    pub fn parse(fields: (String, String, String, String, String, u32, String)) -> Result<Self, Error> {
        let (task_id, txid, request_hash, model_hex, prompt_b64, max_tokens, reward) = fields;
        if !task_id.eq_ignore_ascii_case(&request_hash) || max_tokens == 0 || max_tokens > i32::MAX as u32 {
            return Err("Invalid AI task metadata".into());
        }
        for value in [&task_id, &txid, &request_hash, &model_hex] {
            if value.len() != 64 || !value.bytes().all(|c| c.is_ascii_hexdigit()) {
                return Err("Invalid AI task identifier".into());
            }
        }
        reward.parse::<u64>()?;
        if prompt_b64.len() > MAX_PAYLOAD_BYTES.div_ceil(3) * 4 {
            return Err("AI prompt is too large".into());
        }
        let prompt = String::from_utf8(BASE64.decode(prompt_b64)?)?;
        if prompt.len() > MAX_PAYLOAD_BYTES || prompt.contains('\0') {
            return Err("Invalid AI prompt".into());
        }
        let mut model_id = [0; 32];
        hex::decode_to_slice(&model_hex, &mut model_id)?;
        Ok(Self { task_id, request_hash, model_hex, model_id, prompt, max_tokens: max_tokens as usize })
    }

    pub fn response(&self, id: u32, worker: String, result: &str) -> Result<StratumLine, Error> {
        if result.is_empty() || result.len() > MAX_PAYLOAD_BYTES {
            return Err("Invalid AI result size".into());
        }
        Ok(StratumLine {
            id: Some(id),
            payload: StratumLinePayload::StratumCommand(StratumCommand::MiningAiResponse((
                worker,
                self.task_id.clone(),
                self.request_hash.clone(),
                self.model_hex.clone(),
                BASE64.encode(result.as_bytes()),
            ))),
            jsonrpc: Some("2.0".into()),
            error: None,
        })
    }
}

pub(super) struct InferenceGuard {
    miner: Arc<AtomicBool>,
    busy: Arc<AtomicBool>,
}

impl InferenceGuard {
    pub fn new(miner: Arc<AtomicBool>, busy: Arc<AtomicBool>) -> Self {
        miner.store(true, Ordering::SeqCst);
        Self { miner, busy }
    }
}

impl Drop for InferenceGuard {
    fn drop(&mut self) {
        self.miner.store(false, Ordering::SeqCst);
        self.busy.store(false, Ordering::SeqCst);
    }
}
