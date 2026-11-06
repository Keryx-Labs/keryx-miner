use bytes::BytesMut;
use log::error;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use serde_repr::*;
use std::fmt::{Display, Formatter};
use std::{fmt, io};
use tokio_util::codec::{Decoder, Encoder, LinesCodec};

#[derive(Serialize_repr, Deserialize_repr, Debug, Clone)]
#[repr(u8)]
pub enum ErrorCode {
    Unknown = 20,
    JobNotFound = 21,
    DuplicateShare = 22,
    LowDifficultyShare = 23,
    Unauthorized = 24,
    NotSubscribed = 25,
    UnknownResponse = 41,
    Busy = 46,
}

impl Display for ErrorCode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match &self {
            ErrorCode::Unknown => write!(f, "Unknown"),
            ErrorCode::JobNotFound => write!(f, "JobNotFound"),
            ErrorCode::DuplicateShare => write!(f, "DuplicateShare"),
            ErrorCode::LowDifficultyShare => write!(f, "LowDifficultyShare"),
            ErrorCode::Unauthorized => write!(f, "Unauthorized"),
            ErrorCode::NotSubscribed => write!(f, "NotSubscribed"),
            ErrorCode::UnknownResponse => write!(f, "UnknownResponse"),
            ErrorCode::Busy => write!(f, "Busy"),
        }
    }
}

#[derive(Serialize, Debug, Clone)]
pub(crate) struct StratumError(pub(crate) ErrorCode, pub(crate) String, #[serde(default)] pub(crate) Option<Value>);

impl<'de> Deserialize<'de> for StratumError {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum WireError {
            Tuple(i64, String, #[serde(default)] Option<Value>),
            Object {
                code: i64,
                message: String,
                #[serde(default)]
                data: Option<Value>,
            },
        }
        let (code, message, data) = match WireError::deserialize(deserializer)? {
            WireError::Tuple(code, message, data) => (code, message, data),
            WireError::Object { code, message, data } => (code, message, data),
        };
        let code = match code {
            21 => ErrorCode::JobNotFound,
            22 => ErrorCode::DuplicateShare,
            23 => ErrorCode::LowDifficultyShare,
            24 => ErrorCode::Unauthorized,
            25 => ErrorCode::NotSubscribed,
            _ => ErrorCode::Unknown,
        };
        Ok(Self(code, message, data))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
#[allow(clippy::enum_variant_names)]
pub(crate) enum MiningNotify {
    // job_id, protobuf RpcBlockHeader (hex), timestamp, daa_score, bits, window_seed (hex)
    MiningNotifyV4((String, String, u64, u64, u32, String)),
    MiningNotifyWithTaskV3((String, [u64; 4], u64, u64, u32, String)),
    MiningNotifyShortV3((String, [u64; 4], u64, u64, u32)),
    // 5-element: job_id, header_hash, timestamp, daa_score, task_json (AiRequest payload)
    MiningNotifyWithTask((String, [u64; 4], u64, u64, String)),
    MiningNotifyShortV2((String, [u64; 4], u64, u64)),
    MiningNotifyShort((String, [u64; 4], u64)),
    MiningNotifyLong((String, String, String, String, Vec<String>, String, String, String, bool)),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
#[allow(clippy::enum_variant_names)]
pub enum MiningSubmit {
    // address, job_id, nonce, opoi_tag, cid, pom_proof (hex), task_root (hex)
    MiningSubmitWithPomTask((String, String, String, String, String, String, String)),
    MiningSubmitWithPom((String, String, String, String, String, String)),
    // 5-element: address, job_id, nonce, opoi_tag, ipfs_cid (Phase 2 full inference submit)
    MiningSubmitWithCID((String, String, String, String, String)),
    MiningSubmitWithTag((String, String, String, String)), // address, job_id, nonce, opoi_tag
    MiningSubmitShort((String, String, String)),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum MiningSubscribe {
    MiningSubscribeDefault((String,)),
    MiningSubscribeOptions((String, String)),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum SetExtranonce {
    SetExtranoncePlain((String, u32)),
    SetExtranoncePlainEth((String,)),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "method", content = "params")]
pub(crate) enum StratumCommand {
    #[serde(rename = "mining.set_extranonce", alias = "set_extranonce")]
    SetExtranonce(SetExtranonce),
    #[serde(rename = "mining.set_difficulty")]
    MiningSetDifficulty((f32,)),
    #[serde(rename = "mining.notify")]
    MiningNotify(MiningNotify),
    #[serde(rename = "mining.subscribe")]
    Subscribe(MiningSubscribe),
    #[serde(rename = "mining.authorize")]
    Authorize((String, String)),
    #[serde(rename = "mining.keepalive")]
    MiningKeepalive([(); 0]),
    #[serde(rename = "mining.submit")]
    MiningSubmit(MiningSubmit),
    // Phase 2 Inference: miner → bridge — declare loaded LLM model IDs (sent after authorize)
    #[serde(rename = "mining.declare_capabilities")]
    MiningDeclareCapabilities(Vec<String>),
    #[serde(rename = "mining.ai_request")]
    MiningAiRequest((String, String, String, String, String, u32, String)),
    // worker, task_id, request_hash, model, result b64, trace commitment hex ("" if untraced)
    #[serde(rename = "mining.ai_response")]
    MiningAiResponse((String, String, String, String, String, String)),
    /// An inference command whose params did not parse; answered with an error.
    #[serde(skip)]
    MalformedAi,
    // request_hash, response_txid, seed, deadline_daa, accepted_daa, pieces owed, attention path
    #[serde(rename = "mining.ai_draw")]
    MiningAiDraw((String, String, String, u64, u64, u32, Vec<u64>)),
    // worker, request_hash, response_txid, seed, unsigned opening body b64
    #[serde(rename = "mining.ai_opening")]
    MiningAiOpening((String, String, String, String, String)),
    // Phase 2 Inference: bridge → miner — "model_id_hex:nonce_hex" capability challenge
    #[serde(rename = "mining.challenge")]
    MiningChallenge((String, String)),
    // Capability response: [model_id_hex, nonce_hex, result_b64].
    // response. The nonce is echoed back so the bridge can reject replayed/stale responses.
    #[serde(rename = "mining.challenge_response")]
    MiningChallengeResponse((String, String, String)),
    /*#[serde(rename = "mining.submit_hashrate")]
    MiningSubmitHashrate {
        params: (String, String),
        worker: String,
    },*/ //{"id":9,"method":"mining.submit_hashrate","jsonrpc":"2.0","worker":"rig","params":["0x00000000000000000000000000000000","0x85198cd10b915d560722cdfdf490d4d93892d2cc3fa5f2ff2195d499d04ee54c"]}
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub(crate) enum StratumResult {
    Plain(Option<bool>),
    Eth((bool, String)),
    Subscribe((Vec<(String, String)>, String, u32)),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub(crate) enum StratumLinePayload {
    StratumCommand(StratumCommand),
    StratumResult { result: StratumResult },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct StratumLine {
    pub(crate) id: Option<u32>,
    #[serde(flatten)]
    pub(crate) payload: StratumLinePayload,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) jsonrpc: Option<String>,
    pub(crate) error: Option<StratumError>,
}

impl StratumLine {
    /// Success reply to the pool's command `id`.
    pub(crate) fn ok_reply(id: u32) -> Self {
        Self {
            id: Some(id),
            payload: StratumLinePayload::StratumResult { result: StratumResult::Plain(Some(true)) },
            jsonrpc: Some("2.0".into()),
            error: None,
        }
    }

    /// Error reply to the pool's command `id`.
    pub(crate) fn error_reply(id: u32, code: ErrorCode, message: &str) -> Self {
        Self {
            id: Some(id),
            payload: StratumLinePayload::StratumResult { result: StratumResult::Plain(None) },
            jsonrpc: Some("2.0".into()),
            error: Some(StratumError(code, message.into(), None)),
        }
    }
}

/// An error occurred while encoding or decoding a line.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum NewLineJsonCodecError {
    JsonParseError(String),
    JsonEncodeError,
    LineSplitError,
    LineEncodeError,
    Io(io::Error),
}

impl fmt::Display for NewLineJsonCodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Some error occured")
    }
}
impl From<io::Error> for NewLineJsonCodecError {
    fn from(e: io::Error) -> NewLineJsonCodecError {
        NewLineJsonCodecError::Io(e)
    }
}
impl std::error::Error for NewLineJsonCodecError {}

impl From<(String, String)> for NewLineJsonCodecError {
    fn from(e: (String, String)) -> Self {
        NewLineJsonCodecError::JsonParseError(format!("{}: {}", e.0, e.1))
    }
}

pub(crate) struct NewLineJsonCodec {
    lines_codec: LinesCodec,
}

impl NewLineJsonCodec {
    pub fn new() -> Self {
        Self { lines_codec: LinesCodec::new_with_max_length(2 * 1024 * 1024) }
    }
}

impl Decoder for NewLineJsonCodec {
    type Item = StratumLine;
    type Error = NewLineJsonCodecError;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        loop {
            match self.lines_codec.decode(src) {
                Ok(Some(s)) => match parse_line(s)? {
                    Some(line) => return Ok(Some(line)),
                    None => continue,
                },
                Err(_) => return Err(NewLineJsonCodecError::LineSplitError),
                _ => return Ok(None),
            }
        }
    }

    fn decode_eof(&mut self, buf: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        loop {
            match self.lines_codec.decode_eof(buf) {
                Ok(Some(s)) => match parse_line(s)? {
                    Some(line) => return Ok(Some(line)),
                    None => continue,
                },
                Err(_) => return Err(NewLineJsonCodecError::LineSplitError),
                _ => return Ok(None),
            }
        }
    }
}

/// Parses one line; a malformed `mining.ai_*` command becomes `MalformedAi`, or is skipped when it
/// has no id.
fn parse_line(s: String) -> Result<Option<StratumLine>, NewLineJsonCodecError> {
    match serde_json::from_str::<StratumLine>(&s) {
        Ok(line) => Ok(Some(line)),
        Err(e) => {
            let value = serde_json::from_str::<Value>(&s).ok();
            let method = value.as_ref().and_then(|v| v.get("method")).and_then(Value::as_str);
            match method.filter(|m| m.starts_with("mining.ai_")) {
                Some(m) => {
                    log::warn!("Stratum: malformed {} ignored: {}", m, e);
                    let id = value.as_ref().and_then(|v| v.get("id")).and_then(Value::as_u64).and_then(|i| u32::try_from(i).ok());
                    Ok(id.map(|id| StratumLine {
                        id: Some(id),
                        payload: StratumLinePayload::StratumCommand(StratumCommand::MalformedAi),
                        jsonrpc: None,
                        error: None,
                    }))
                }
                None => Err((e.to_string(), s).into()),
            }
        }
    }
}

impl Encoder<StratumLine> for NewLineJsonCodec {
    type Error = NewLineJsonCodecError;

    fn encode(&mut self, item: StratumLine, dst: &mut BytesMut) -> Result<(), Self::Error> {
        match serde_json::to_string(&item) {
            Ok(json) => self.lines_codec.encode(json, dst).map_err(|_| NewLineJsonCodecError::LineEncodeError),
            Err(e) => {
                error!("Error! {:?}", e);
                Err(NewLineJsonCodecError::JsonEncodeError)
            }
        }
    }
}

impl Default for NewLineJsonCodec {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_ai_command_is_skipped() {
        let mut codec = NewLineJsonCodec::new();
        let mut buf = BytesMut::from(
            &b"{\"method\":\"mining.ai_draw\",\"params\":[\"x\"]}\n{\"id\":8,\"result\":true,\"error\":null}\n"[..],
        );
        assert_eq!(codec.decode(&mut buf).unwrap().unwrap().id, Some(8));
        let mut buf = BytesMut::from(&b"{\"id\":7,\"method\":\"mining.ai_draw\",\"params\":[\"x\"]}\n"[..]);
        let line = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(line.id, Some(7));
        assert!(matches!(line.payload, StratumLinePayload::StratumCommand(StratumCommand::MalformedAi)));
        let mut buf = BytesMut::from(&b"{\"id\":9,\"method\":\"mining.notify\",\"params\":[1]}\n"[..]);
        assert!(codec.decode(&mut buf).is_err());
    }

    #[test]
    fn error_reply_is_a_result() {
        let line = StratumLine::error_reply(5, ErrorCode::Busy, "busy");
        let json = serde_json::to_value(&line).unwrap();
        assert_eq!(json["id"], 5);
        assert!(json["result"].is_null());
        assert_eq!(json["error"][0], 46);
    }
}
