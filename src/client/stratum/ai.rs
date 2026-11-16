use super::statum_codec::{AiRequestParams, ErrorCode, StratumCommand, StratumLine, StratumLinePayload};
use crate::Error;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

const MAX_PAYLOAD_BYTES: usize = 1_048_576;
const MAX_UPLOAD_URL: usize = 2048;
const UPLOAD_ATTEMPTS: u32 = 4;

/// Trace parameters the pool sets for a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TraceSpec {
    pub tier: u8,
    pub chain: bool,
    pub attention: bool,
}

pub(super) struct AiRequest {
    pub task_id: String,
    pub request_hash: String,
    pub model_hex: String,
    pub model_id: [u8; 32],
    pub prompt: String,
    /// Plaintext prompt length the node sees: the pool's value, else the byte length sent.
    pub prompt_len: u32,
    pub max_tokens: usize,
    /// Trace parameters set by the pool, when it sent them.
    pub trace: Option<TraceSpec>,
    /// Where the pool wants the trace record posted.
    pub upload: Option<String>,
    /// Id of the pool's `mining.ai_request`, for an error reply.
    pub reply_id: Option<u32>,
}

/// The `trace` object of a request extension; `None` when the extension carries none.
fn parse_trace(ext: &Value) -> Result<Option<(TraceSpec, Option<u32>, Option<String>)>, Error> {
    let Some(t) = ext.get("trace").filter(|t| !t.is_null()) else { return Ok(None) };
    let tier = t.get("tier").and_then(Value::as_u64).filter(|v| *v < 16).ok_or("Invalid AI trace tier")? as u8;
    let flag = |k: &str| t.get(k).and_then(Value::as_bool).unwrap_or(true);
    let (chain, attention) = (flag("chain"), flag("attention"));
    let prompt_len = match t.get("prompt_len") {
        None => None,
        Some(v) => Some(v.as_u64().and_then(|v| u32::try_from(v).ok()).ok_or("Invalid AI trace prompt length")?),
    };
    let upload = match t.get("upload").and_then(Value::as_str) {
        None => None,
        Some(u) if u.len() <= MAX_UPLOAD_URL && (u.starts_with("http://") || u.starts_with("https://")) => Some(u.to_string()),
        Some(_) => return Err("Invalid AI trace upload URL".into()),
    };
    Ok(Some((TraceSpec { tier, chain, attention: chain && attention }, prompt_len, upload)))
}

impl AiRequest {
    pub fn parse(params: AiRequestParams) -> Result<Self, Error> {
        let ((task_id, txid, request_hash, model_hex, prompt_b64, max_tokens, reward), ext) = match params {
            AiRequestParams::WithExt((a, b, c, d, e, f, g, ext)) => ((a, b, c, d, e, f, g), ext),
            AiRequestParams::Plain(fields) => (fields, Value::Null),
        };
        let trace = parse_trace(&ext)?;
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
        let raw = BASE64.decode(prompt_b64)?;
        if raw.len() > MAX_PAYLOAD_BYTES {
            return Err("Invalid AI prompt".into());
        }
        let prompt = crate::client::grpc::decode_private_prompt(&raw);
        let sent_len = raw.len().min(keryx_inference::MAX_AI_REQUEST_PRIVATE_PAYLOAD_LEN) as u32;
        let prompt_len = trace.as_ref().and_then(|(_, l, _)| *l).map_or(sent_len, |l| l.min(keryx_inference::MAX_AI_REQUEST_PRIVATE_PAYLOAD_LEN as u32));
        let mut model_id = [0; 32];
        hex::decode_to_slice(&model_hex, &mut model_id)?;
        Ok(Self {
            task_id,
            request_hash,
            model_hex,
            model_id,
            prompt,
            prompt_len,
            max_tokens: max_tokens as usize,
            upload: trace.as_ref().and_then(|(_, _, u)| u.clone()),
            trace: trace.map(|(t, _, _)| t),
            reply_id: None,
        })
    }

    /// Busy reply refusing this request, when the pool gave it an id.
    pub fn refusal(&self, message: &str) -> Option<StratumLine> {
        self.reply_id.map(|id| StratumLine::error_reply(id, ErrorCode::Busy, message))
    }

    /// The answer: its text, and for a traced answer the commitment and the body the pool seals
    /// (from the trace chain on, the text followed by its generated token ids).
    pub fn response(&self, id: u32, worker: String, text: &str, traced: Option<(String, Vec<u8>)>) -> Result<StratumLine, Error> {
        if text.len() > MAX_PAYLOAD_BYTES || (text.is_empty() && traced.is_none()) {
            return Err("Invalid AI result size".into());
        }
        let mut params = vec![
            json!(worker),
            json!(self.task_id),
            json!(self.request_hash),
            json!(self.model_hex),
            json!(BASE64.encode(text.as_bytes())),
        ];
        if let Some((commitment, body)) = traced {
            if body.is_empty() || body.len() > MAX_PAYLOAD_BYTES {
                return Err("Invalid AI result size".into());
            }
            params.push(json!({ "trace": commitment, "body": hex::encode(body) }));
        }
        Ok(StratumLine {
            id: Some(id),
            payload: StratumLinePayload::StratumCommand(StratumCommand::MiningAiResponse(params)),
            jsonrpc: Some("2.0".into()),
            error: None,
        })
    }
}

/// An audit opening the pool asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AiOpening {
    pub request_hash: String,
    pub response_txid: [u8; 32],
    pub seed: [u8; 32],
    pub path: Vec<u64>,
}

impl AiOpening {
    /// Parses `[request_hash, response_txid, seed, path?, prev_hex?, record_url?]`.
    pub fn parse(params: &[Value]) -> Result<Self, Error> {
        let hex32 = |i: usize| -> Result<[u8; 32], Error> {
            let s = params.get(i).and_then(Value::as_str).ok_or("Invalid AI opening params")?;
            let mut out = [0u8; 32];
            hex::decode_to_slice(s, &mut out).map_err(|_| "Invalid AI opening identifier")?;
            Ok(out)
        };
        let request_hash = hex::encode(hex32(0)?);
        let (response_txid, seed) = (hex32(1)?, hex32(2)?);
        let path = match params.get(3) {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(a)) => a.iter().map(|v| v.as_u64().ok_or("Invalid AI opening path")).collect::<Result<_, _>>()?,
            Some(_) => return Err("Invalid AI opening path".into()),
        };
        Ok(Self { request_hash, response_txid, seed, path })
    }

    /// The `mining.ai_opening_result` line: the unsigned body, or the reason it is missing.
    pub fn result(&self, id: u32, worker: &str, body: &[u8], error: &str) -> StratumLine {
        StratumLine {
            id: Some(id),
            payload: StratumLinePayload::StratumCommand(StratumCommand::MiningAiOpeningResult(vec![
                json!(worker),
                json!(self.request_hash),
                json!(hex::encode(self.response_txid)),
                json!(self.path),
                json!(hex::encode(body)),
                json!(error),
            ])),
            jsonrpc: Some("2.0".into()),
            error: None,
        }
    }
}

pub(super) struct InferenceGuard {
    miner: Arc<AtomicBool>,
    busy: Arc<AtomicBool>,
    /// The one device paused with the model it serves, or `None` for the whole rig.
    device: Option<(u32, [u8; 32])>,
}

impl InferenceGuard {
    /// Pauses `device` alone for `model_id` when given, else the whole rig.
    pub fn new(miner: Arc<AtomicBool>, busy: Arc<AtomicBool>, device: Option<u32>, model_id: [u8; 32]) -> Self {
        let device = device.map(|d| (d, model_id));
        match device {
            Some((dev, model)) => keryx_miner::pom_gpu::set_inference_paused_on(dev, &model, true),
            None => {
                keryx_miner::pom_gpu::set_inference_paused(true);
                miner.store(true, Ordering::SeqCst);
            }
        }
        Self { miner, busy, device }
    }

    /// Whether the whole rig is paused, so templates must be held.
    pub fn whole_rig(&self) -> bool {
        self.device.is_none()
    }
}

impl Drop for InferenceGuard {
    fn drop(&mut self) {
        match self.device {
            Some((dev, model)) => keryx_miner::pom_gpu::set_inference_paused_on(dev, &model, false),
            None => keryx_miner::pom_gpu::set_inference_paused(false),
        }
        self.miner.store(false, Ordering::SeqCst);
        self.busy.store(false, Ordering::SeqCst);
        super::audit::shared().wake.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tuple(prompt: &[u8]) -> (String, String, String, String, String, u32, String) {
        let h = "ab".repeat(32);
        (h.clone(), h.clone(), h.clone(), h, BASE64.encode(prompt), 64, "0".into())
    }

    fn fields(prompt: &[u8]) -> AiRequestParams {
        AiRequestParams::Plain(tuple(prompt))
    }

    fn with_ext(prompt: &[u8], ext: Value) -> AiRequestParams {
        let (a, b, c, d, e, f, g) = tuple(prompt);
        AiRequestParams::WithExt((a, b, c, d, e, f, g, ext))
    }

    #[test]
    fn parse_keeps_odd_prompts_and_the_sent_length() {
        let r = AiRequest::parse(fields(b"Say hel\0lo")).unwrap();
        assert_eq!((r.prompt.as_str(), r.prompt_len), ("Say hel\0lo", 10));
        let r = AiRequest::parse(fields(&[b'h', b'i', 0xFF])).unwrap();
        assert_eq!((r.prompt.as_str(), r.prompt_len), ("hi\u{FFFD}", 3));
    }

    #[test]
    fn parse_inflates_a_compressed_prompt_but_keeps_its_sent_length() {
        use std::io::Write;
        let text = "abc ".repeat(2_000);
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::best());
        enc.write_all(text.as_bytes()).unwrap();
        let mut packed = vec![0x00, b'K', b'Z', 0x01];
        packed.extend(enc.finish().unwrap());
        let r = AiRequest::parse(fields(&packed)).unwrap();
        assert_eq!(r.prompt, text);
        assert_eq!(r.prompt_len as usize, packed.len());
    }

    #[test]
    fn parse_caps_the_sent_length_and_rejects_bad_input() {
        let big = vec![b'a'; keryx_inference::MAX_AI_REQUEST_PRIVATE_PAYLOAD_LEN * 2];
        let r = AiRequest::parse(fields(&big)).unwrap();
        assert_eq!(r.prompt_len as usize, keryx_inference::MAX_AI_REQUEST_PRIVATE_PAYLOAD_LEN);
        assert_eq!(r.prompt.len(), big.len());
        assert!(AiRequest::parse(fields(&vec![b'a'; MAX_PAYLOAD_BYTES + 1])).is_err());
        let mut f = tuple(b"hi");
        f.4 = "not base64!".into();
        assert!(AiRequest::parse(AiRequestParams::Plain(f)).is_err());
    }

    #[test]
    fn parse_takes_the_pool_trace_parameters() {
        let r = AiRequest::parse(fields(b"hi")).unwrap();
        assert_eq!(r.trace, None);
        let r = AiRequest::parse(with_ext(b"hi", json!({"trace": {"tier": 3, "prompt_len": 1162}}))).unwrap();
        assert_eq!(r.trace, Some(TraceSpec { tier: 3, chain: true, attention: true }));
        assert_eq!(r.prompt_len, 1162);
        let r = AiRequest::parse(with_ext(b"hi", json!({"trace": {"tier": 1, "chain": false, "attention": true}}))).unwrap();
        assert_eq!(r.trace, Some(TraceSpec { tier: 1, chain: false, attention: false }));
        assert_eq!(r.prompt_len, 2);
        let r = AiRequest::parse(with_ext(b"hi", json!({"trace": {"tier": 0, "prompt_len": 1_000_000}}))).unwrap();
        assert_eq!(r.prompt_len as usize, keryx_inference::MAX_AI_REQUEST_PRIVATE_PAYLOAD_LEN);
        assert_eq!(AiRequest::parse(with_ext(b"hi", Value::Null)).unwrap().trace, None);
        assert_eq!(AiRequest::parse(with_ext(b"hi", json!({"other": 1}))).unwrap().trace, None);
        assert!(AiRequest::parse(with_ext(b"hi", json!({"trace": {"tier": 16}}))).is_err());
        assert!(AiRequest::parse(with_ext(b"hi", json!({"trace": {"prompt_len": 3}}))).is_err());
        assert!(AiRequest::parse(with_ext(b"hi", json!({"trace": {"tier": 1, "prompt_len": "3"}}))).is_err());
    }

    #[test]
    fn parse_takes_the_upload_url() {
        let t = |u: Value| AiRequest::parse(with_ext(b"hi", json!({"trace": {"tier": 1, "upload": u}})));
        assert_eq!(t(json!("https://pool.example/traces/")).unwrap().upload.as_deref(), Some("https://pool.example/traces/"));
        assert_eq!(t(json!("http://10.0.0.1:8080/r")).unwrap().upload.as_deref(), Some("http://10.0.0.1:8080/r"));
        assert_eq!(t(json!(7)).unwrap().upload, None);
        assert_eq!(t(Value::Null).unwrap().upload, None);
        assert_eq!(AiRequest::parse(fields(b"hi")).unwrap().upload, None);
        assert!(t(json!("ftp://pool.example/")).is_err());
        assert!(t(json!("file:///etc/passwd")).is_err());
        assert!(t(json!(format!("https://{}", "a".repeat(MAX_UPLOAD_URL)))).is_err());
    }

    #[test]
    fn upload_retries_server_errors_and_stops_on_a_refusal() {
        use std::io::{Read, Write};
        let serve = |codes: Vec<u16>| {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let port = l.local_addr().unwrap().port();
            let h = std::thread::spawn(move || {
                let mut paths = Vec::new();
                for code in codes {
                    let (mut s, _) = l.accept().unwrap();
                    let mut buf = vec![0; 65536];
                    let n = s.read(&mut buf).unwrap();
                    let head = String::from_utf8_lossy(&buf[..n]).to_string();
                    paths.push(head.lines().next().unwrap_or_default().to_string());
                    write!(s, "HTTP/1.1 {code} X\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
                }
                paths
            });
            (format!("http://127.0.0.1:{port}/rec/"), h)
        };
        let (url, h) = serve(vec![201]);
        assert_eq!(upload_record(&url, "ABCD", b"x"), Ok(201));
        assert_eq!(h.join().unwrap(), vec!["POST /rec/abcd HTTP/1.1".to_string()]);
        let (url, h) = serve(vec![403]);
        assert_eq!(upload_record(&url, "ab", b"x"), Err("HTTP 403".into()));
        assert_eq!(h.join().unwrap().len(), 1);
        let (url, h) = serve(vec![503, 200]);
        assert_eq!(upload_record(&url, "ab", b"x"), Ok(200));
        assert_eq!(h.join().unwrap().len(), 2);
    }

    #[test]
    fn response_carries_the_text_and_the_trace_object() {
        let r = AiRequest::parse(fields(b"hi")).unwrap();
        let line = r.response(9, "w".into(), "Paris", Some(("aa".into(), vec![1, 2]))).unwrap();
        let v = serde_json::to_value(&line).unwrap();
        assert_eq!(v["method"], "mining.ai_response");
        assert_eq!(v["params"][4], BASE64.encode("Paris"));
        assert_eq!(v["params"][5], json!({"trace": "aa", "body": "0102"}));
        let v = serde_json::to_value(r.response(9, "w".into(), "Paris", None).unwrap()).unwrap();
        assert_eq!(v["params"].as_array().unwrap().len(), 5);
        assert!(r.response(9, "w".into(), "", Some(("aa".into(), vec![1]))).is_ok());
        assert!(r.response(9, "w".into(), "", None).is_err());
        assert!(r.response(9, "w".into(), "x", Some(("aa".into(), Vec::new()))).is_err());
    }

    #[test]
    fn opening_parses_and_answers() {
        let (rh, tx, seed) = ("AB".repeat(32), "cd".repeat(32), "ef".repeat(32));
        let o = AiOpening::parse(&[json!(rh), json!(tx), json!(seed)]).unwrap();
        assert_eq!((o.request_hash.as_str(), o.path.len()), ("ab".repeat(32).as_str(), 0));
        let o = AiOpening::parse(&[json!(rh), json!(tx), json!(seed), json!([2, 0]), json!("00"), json!("https://x")]).unwrap();
        assert_eq!(o.path, vec![2, 0]);
        assert!(AiOpening::parse(&[json!(rh), json!(tx), json!(seed), Value::Null]).unwrap().path.is_empty());
        assert!(AiOpening::parse(&[json!(rh), json!(tx)]).is_err());
        assert!(AiOpening::parse(&[json!(rh), json!(tx), json!("zz")]).is_err());
        assert!(AiOpening::parse(&[json!(rh), json!(tx), json!(seed), json!("x")]).is_err());
        let v = serde_json::to_value(o.result(4, "w", &[0xB0, 1], "")).unwrap();
        assert_eq!(v["method"], "mining.ai_opening_result");
        assert_eq!(v["params"], json!(["w", "ab".repeat(32), tx, [2, 0], "b001", ""]));
    }
}

/// POSTs a trace record to `{url}/{request_hash}`. Blocking.
pub(super) fn upload_record(url: &str, request_hash_hex: &str, blob: &[u8]) -> Result<u16, String> {
    let target = format!("{}/{}", url.trim_end_matches('/'), request_hash_hex.to_ascii_lowercase());
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(300))
        .build();
    let mut last = String::new();
    for attempt in 1..=UPLOAD_ATTEMPTS {
        match agent.post(&target).set("Content-Type", "application/octet-stream").send_bytes(blob) {
            Ok(r) => return Ok(r.status()),
            Err(ureq::Error::Status(code, _)) if code < 500 && code != 408 && code != 429 => {
                return Err(format!("HTTP {code}"));
            }
            Err(ureq::Error::Status(code, _)) => last = format!("HTTP {code}"),
            Err(e) => last = e.to_string(),
        }
        if attempt < UPLOAD_ATTEMPTS {
            std::thread::sleep(std::time::Duration::from_secs(5 * attempt as u64));
        }
    }
    Err(last)
}
