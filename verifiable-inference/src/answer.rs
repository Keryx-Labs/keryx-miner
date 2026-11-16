//! Plaintext of a stage-2 sealed answer: the text, then the token ids of the generated positions.

/// Closes a plaintext that carries token ids.
pub const TOKENS_MAGIC: &[u8; 4] = b"KXTK";
/// Bytes after the ids: their count, then the magic.
pub const TRAILER_LEN: usize = 8;

/// `text ‖ ids (u32 LE) ‖ count (u32 LE) ‖ TOKENS_MAGIC`.
pub fn encode(text: &[u8], ids: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len() + 4 * ids.len() + TRAILER_LEN);
    out.extend_from_slice(text);
    for id in ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    out.extend_from_slice(&(ids.len() as u32).to_le_bytes());
    out.extend_from_slice(TOKENS_MAGIC);
    out
}

/// The text and token ids of a plaintext, `None` when it carries no ids.
pub fn decode(plain: &[u8]) -> Option<(&[u8], Vec<u32>)> {
    let body = plain.len().checked_sub(TRAILER_LEN)?;
    if &plain[body + 4..] != TOKENS_MAGIC {
        return None;
    }
    let n = u32::from_le_bytes(plain[body..body + 4].try_into().ok()?) as usize;
    let text_len = body.checked_sub(n.checked_mul(4)?)?;
    let ids = plain[text_len..body].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    Some((&plain[..text_len], ids))
}

/// Text length of a plaintext of `answer_len` bytes carrying `n_ids` ids, `None` when it is too
/// short to carry them.
pub fn text_len(answer_len: u32, n_ids: u32) -> Option<u32> {
    answer_len.checked_sub(n_ids.checked_mul(4)?.checked_add(TRAILER_LEN as u32)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_sizes_the_text() {
        let plain = encode(b"hello", &[7, 1 << 20, 3]);
        assert_eq!(plain.len(), 5 + 12 + TRAILER_LEN);
        let (text, ids) = decode(&plain).unwrap();
        assert_eq!((text, ids.as_slice()), (&b"hello"[..], &[7, 1 << 20, 3][..]));
        assert_eq!(text_len(plain.len() as u32, 3), Some(5));
        assert_eq!(text_len(19, 3), None);
        assert_eq!(decode(b"plain text without ids"), None);
        let mut lying = encode(b"x", &[1]);
        let n = lying.len();
        lying[n - 8..n - 4].copy_from_slice(&9u32.to_le_bytes());
        assert_eq!(decode(&lying), None);
        assert_eq!(decode(&encode(b"", &[])), Some((&b""[..], vec![])));
    }
}
