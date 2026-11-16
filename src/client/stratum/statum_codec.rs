use bytes::BytesMut;
use log::error;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use serde_repr::*;
use std::fmt::{Display, Formatter};
use std::{fmt, io};
use tokio_util::codec::{Decoder, Encoder, LinesCodec, LinesCodecError};

#[derive(Serialize_repr, Deserialize_repr, Debug, Clone)]
#[repr(u8)]
pub enum ErrorCode {
    Unknown = 20,
    JobNotFound = 21,
    DuplicateShare = 22,
    LowDifficultyShare = 23,
    Unauthorized = 24,
    NotSubscribed = 25,
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

/// `mining.ai_request` params: task_id, txid, request_hash, model, prompt b64, max_tokens, reward,
/// and an optional extension object carrying the trace parameters.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub(crate) enum AiRequestParams {
    WithExt((String, String, String, String, String, u32, String, Value)),
    Plain((String, String, String, String, String, u32, String)),
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
    MiningAiRequest(AiRequestParams),
    // worker, task_id, request_hash, model, answer text b64, optional {"trace": hex, "body": hex}
    #[serde(rename = "mining.ai_response")]
    MiningAiResponse(Vec<Value>),
    /// An inference command whose params did not parse; answered with an error.
    #[serde(skip)]
    MalformedAi,
    // request_hash, response_txid, draw seed, attention path, previous partials hex, record url
    #[serde(rename = "mining.ai_opening")]
    MiningAiOpening(Vec<Value>),
    // worker, request_hash, response_txid, path, unsigned body hex, error ("" on success)
    #[serde(rename = "mining.ai_opening_result")]
    MiningAiOpeningResult(Vec<Value>),
    // Worker description sent once after authorize: [{"v":1,"miner":{..},"trace":1,"model":{..}}]
    #[serde(rename = "mining.hello")]
    MiningHello(Vec<Value>),
    // Periodic worker state: [{"v":1,"trace":0|1,"models":[hex]}]
    #[serde(rename = "mining.telemetry")]
    MiningTelemetry(Vec<Value>),
    // Off-chain chat request from the pool: {"reqId", "model_id", "prompt", "max_tokens", ...}
    #[serde(rename = "mining.inference_request")]
    MiningInferenceRequest(Value),
    // Answer to an off-chain chat request: {"reqId", "ok", "error" | "text", ...}
    #[serde(rename = "mining.inference_result")]
    MiningInferenceResult(Value),
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
                // The codec skips the rest of an oversized line; the connection stays up.
                Err(LinesCodecError::MaxLineLengthExceeded) => log::warn!("Stratum: oversized line from the pool skipped"),
                Err(_) => return Err(NewLineJsonCodecError::LineSplitError),
                Ok(None) => return Ok(None),
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
                Err(LinesCodecError::MaxLineLengthExceeded) => log::warn!("Stratum: oversized line from the pool skipped"),
                Err(_) => return Err(NewLineJsonCodecError::LineSplitError),
                Ok(None) => return Ok(None),
            }
        }
    }
}

/// Methods this worker parses; any other method is skipped.
const KNOWN_METHODS: &[&str] = &[
    "mining.set_extranonce",
    "set_extranonce",
    "mining.set_difficulty",
    "mining.notify",
    "mining.subscribe",
    "mining.authorize",
    "mining.keepalive",
    "mining.submit",
    "mining.declare_capabilities",
    "mining.challenge",
    "mining.challenge_response",
    "mining.inference_request",
];

/// Parses one line; a malformed `mining.ai_*` command becomes `MalformedAi`, or is skipped when it
/// has no id; a line with an unknown method is skipped.
fn parse_line(s: String) -> Result<Option<StratumLine>, NewLineJsonCodecError> {
    match serde_json::from_str::<StratumLine>(&s) {
        Ok(line) => Ok(Some(line)),
        Err(e) => {
            let value = serde_json::from_str::<Value>(&s).ok();
            let method = value.as_ref().and_then(|v| v.get("method")).and_then(Value::as_str);
            if let Some(m) = method.filter(|m| !m.starts_with("mining.ai_") && !KNOWN_METHODS.contains(m)) {
                log::warn!("Stratum: unknown method {} ignored", m);
                return Ok(None);
            }
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
    fn an_oversized_line_is_skipped() {
        let mut codec = NewLineJsonCodec { lines_codec: LinesCodec::new_with_max_length(64) };
        let mut buf = BytesMut::from(format!("{}\n{{\"id\":4,\"result\":true,\"error\":null}}\n", "x".repeat(200)).as_bytes());
        assert_eq!(codec.decode(&mut buf).unwrap().unwrap().id, Some(4));
    }

    #[test]
    fn unknown_methods_are_skipped() {
        let mut codec = NewLineJsonCodec::new();
        let mut buf = BytesMut::from(
            &b"{\"id\":null,\"method\":\"mining.unheard_of\",\"params\":{\"reqId\":\"a\"}}\n{\"id\":3,\"result\":true,\"error\":null}\n"[..],
        );
        assert_eq!(codec.decode(&mut buf).unwrap().unwrap().id, Some(3));
    }

    #[test]
    fn ai_request_takes_seven_or_eight_params() {
        let h = "ab".repeat(32);
        let plain = format!("{{\"id\":1,\"method\":\"mining.ai_request\",\"params\":[\"{h}\",\"{h}\",\"{h}\",\"{h}\",\"aGk=\",8,\"5\"]}}");
        let ext = format!(
            "{{\"id\":2,\"method\":\"mining.ai_request\",\"params\":[\"{h}\",\"{h}\",\"{h}\",\"{h}\",\"aGk=\",8,\"5\",{{\"trace\":{{\"tier\":2,\"prompt_len\":17}}}}]}}"
        );
        for (s, with_ext) in [(plain, false), (ext, true)] {
            let line = parse_line(s).unwrap().unwrap();
            match line.payload {
                StratumLinePayload::StratumCommand(StratumCommand::MiningAiRequest(p)) => {
                    assert_eq!(matches!(p, AiRequestParams::WithExt(_)), with_ext)
                }
                other => panic!("{:?}", other),
            }
        }
    }

    #[test]
    fn ai_opening_parses_any_param_list() {
        let s = "{\"id\":null,\"method\":\"mining.ai_opening\",\"params\":[\"a\",\"b\",\"c\",[1,2],\"00\"]}".to_string();
        let line = parse_line(s).unwrap().unwrap();
        assert!(matches!(line.payload, StratumLinePayload::StratumCommand(StratumCommand::MiningAiOpening(p)) if p.len() == 5));
    }

    #[test]
    fn chat_requests_parse_and_hello_serializes() {
        let s = "{\"id\":null,\"method\":\"mining.inference_request\",\"params\":{\"reqId\":\"r1\",\"prompt\":\"hi\"}}".to_string();
        let line = parse_line(s).unwrap().unwrap();
        assert!(matches!(line.payload, StratumLinePayload::StratumCommand(StratumCommand::MiningInferenceRequest(v)) if v["reqId"] == "r1"));
        let hello = StratumLine {
            id: Some(2),
            payload: StratumLinePayload::StratumCommand(StratumCommand::MiningHello(vec![serde_json::json!({"v": 1, "trace": 1})])),
            jsonrpc: None,
            error: None,
        };
        let v = serde_json::to_value(&hello).unwrap();
        assert_eq!(v["method"], "mining.hello");
        assert_eq!(v["params"][0]["trace"], 1);
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
