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
pub(crate) enum MiningNotify {
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
pub enum MiningSubmit {
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
    // Phase 2 OPoI: miner → bridge — declare loaded SLM model IDs (sent after authorize)
    #[serde(rename = "mining.declare_capabilities")]
    MiningDeclareCapabilities(Vec<String>),
    #[serde(rename = "mining.ai_request")]
    MiningAiRequest((String, String, String, String, String, u32, String)),
    #[serde(rename = "mining.ai_response")]
    MiningAiResponse((String, String, String, String, String)),
    // Phase 2 OPoI: bridge → miner — "model_id_hex:nonce_hex" capability challenge
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
        match self.lines_codec.decode(src) {
            Ok(Some(s)) => {
                serde_json::from_str::<StratumLine>(s.as_str()).map_err(|e| (e.to_string(), s).into()).map(Some)
            }
            Err(_) => Err(NewLineJsonCodecError::LineSplitError),
            _ => Ok(None),
        }
    }

    fn decode_eof(&mut self, buf: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        match self.lines_codec.decode_eof(buf) {
            Ok(Some(s)) => serde_json::from_str(s.as_str()).map_err(|e| (e.to_string(), s).into()),
            Err(_) => Err(NewLineJsonCodecError::LineSplitError),
            _ => Ok(None),
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
