//! Shape rule: the positions a trace commits against the sealed sizes of its request and response.

/// Prompt bytes per committed prompt position, at most.
pub const PROMPT_BYTES_PER_POSITION: u32 = 16;
/// Answer bytes per committed generated position, at most.
pub const ANSWER_BYTES_PER_POSITION: u32 = 10;
/// Answer bytes per token of the request's budget, at most.
pub const MAX_ANSWER_BYTES_PER_TOKEN: u32 = 32;
/// Positions a trace commits, at most: the largest context of the lineup.
pub const MAX_TRACE_POSITIONS: u32 = 131_072;

/// Prompt positions a trace must commit for a prompt of `prompt_len` bytes.
pub fn min_prompt_positions(prompt_len: u32) -> u32 {
    prompt_len / PROMPT_BYTES_PER_POSITION
}

/// Generated positions a trace must commit for an answer of `answer_len` bytes under `max_tokens`.
pub fn min_generated_positions(answer_len: u32, max_tokens: u32) -> u32 {
    (answer_len / ANSWER_BYTES_PER_POSITION).min(max_tokens)
}

/// Longest answer the request's token budget allows.
pub fn max_answer_len(max_tokens: u32) -> u64 {
    max_tokens as u64 * MAX_ANSWER_BYTES_PER_TOKEN as u64
}

/// Whether a commitment of `n_prompt` prompt positions and `n_pos` positions in total fits a
/// prompt of `prompt_len` bytes and an answer of `answer_len` bytes under `max_tokens`.
pub fn shape_ok(prompt_len: u32, answer_len: u32, max_tokens: u32, n_prompt: u32, n_pos: u32) -> bool {
    n_prompt > 0
        && n_pos >= n_prompt
        && n_pos <= MAX_TRACE_POSITIONS
        && n_pos - n_prompt <= max_tokens
        && n_prompt >= min_prompt_positions(prompt_len)
        && n_pos - n_prompt >= min_generated_positions(answer_len, max_tokens)
        && answer_len as u64 <= max_answer_len(max_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_follow_the_sealed_sizes() {
        assert_eq!(min_prompt_positions(0), 0);
        assert_eq!(min_prompt_positions(15), 0);
        assert_eq!(min_prompt_positions(16), 1);
        assert_eq!(min_prompt_positions(29_000), 1_812);
        assert_eq!(min_generated_positions(400, 2_048), 40);
        assert_eq!(min_generated_positions(4_000, 64), 64);
        assert_eq!(max_answer_len(2_048), 65_536);
    }

    #[test]
    fn a_real_median_response_passes() {
        // ~3 KiB prompt, ~100 B answer, 2 048 tokens; a templated prompt and a short answer.
        assert!(shape_ok(3_100, 100, 2_048, 220 + 700, 220 + 700 + 25));
    }

    #[test]
    fn a_trivial_trace_fails() {
        assert!(!shape_ok(3_100, 100, 2_048, 1, 2));
        assert!(!shape_ok(3_100, 100, 2_048, 220 + 700, 220 + 700 + 1));
        assert!(!shape_ok(3_100, 2_100, 64, 1_000, 1_064));
        assert!(!shape_ok(3_100, 100, 2_048, 0, 0));
        assert!(!shape_ok(3_100, 100, 2_048, 1_000, 999));
    }

    #[test]
    fn exact_bounds_pass() {
        assert!(shape_ok(1_600, 1_000, 2_048, 100, 200));
        assert!(!shape_ok(1_600, 1_000, 2_048, 99, 200));
        assert!(!shape_ok(1_600, 1_000, 2_048, 100, 199));
        assert!(shape_ok(0, 2_048, 64, 1, 65));
        assert!(!shape_ok(0, 2_049, 64, 1, 65));
        assert!(!shape_ok(0, 100, 64, 1, 66), "more generated positions than the budget");
        assert!(shape_ok(0, 100, 64, MAX_TRACE_POSITIONS - 64, MAX_TRACE_POSITIONS));
        assert!(!shape_ok(0, 100, 64, MAX_TRACE_POSITIONS - 63, MAX_TRACE_POSITIONS + 1), "past the largest context");
    }
}
