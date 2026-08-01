use std::collections::HashMap;

/// BLEU score calculation for evaluating translation quality.
///
/// Paper: 6.1, Table 2 — BLEU evaluation on WMT 2014 datasets.
///
/// Implements BLEU-4 with:
/// - Modified n-gram precision (up to 4-grams)
/// - Brevity penalty
/// - Smoothing via Add-1 (for zero n-gram matches)
///
/// Reference: Papineni et al., 2002. "BLEU: a method for automatic evaluation of machine translation"
pub fn bleu_score(reference: &[String], candidate: &[String]) -> f64 {
    if candidate.is_empty() {
        return 0.0;
    }

    let max_n = 4;
    let mut precisions = Vec::with_capacity(max_n);
    let total_ref_len = reference.len();
    let cand_len = candidate.len();

    for n in 1..=max_n {
        let (matches, total) = ngram_precision(reference, candidate, n);
        let precision = if total == 0 {
            0.0
        } else if matches == 0 {
            // Smoothing: add-1
            1.0 / (total as f64 + 1.0)
        } else {
            matches as f64 / total as f64
        };
        precisions.push(precision);
    }

    // Brevity penalty
    let bp = if cand_len < total_ref_len {
        (1.0 - total_ref_len as f64 / cand_len as f64).exp()
    } else {
        1.0
    };

    // Geometric mean of n-gram precisions
    let log_sum: f64 = precisions.iter().map(|p| p.ln()).sum();
    let avg_log_precision = log_sum / max_n as f64;

    bp * avg_log_precision.exp()
}

/// Compute BLEU score for corpus-level evaluation.
///
/// Processes multiple sentence pairs and returns the corpus BLEU.
pub fn corpus_bleu(references: &[Vec<String>], candidates: &[Vec<String>]) -> f64 {
    if references.is_empty() || candidates.is_empty() || references.len() != candidates.len() {
        return 0.0;
    }

    let max_n = 4;
    let mut total_matches = vec![0u64; max_n];
    let mut total_ngrams = vec![0u64; max_n];
    let mut total_ref_len = 0;
    let mut total_cand_len = 0;

    for (ref_sent, cand_sent) in references.iter().zip(candidates.iter()) {
        total_ref_len += ref_sent.len();
        total_cand_len += cand_sent.len();

        for n in 1..=max_n {
            let (matches, total) = ngram_precision(ref_sent, cand_sent, n);
            total_matches[n - 1] += matches as u64;
            total_ngrams[n - 1] += total as u64;
        }
    }

    // Compute precision for each n-gram order, skipping orders with zero total n-grams.
    // If an order has zero total n-grams (e.g. candidate shorter than n), we skip it
    // rather than including 0.0 (which would give ln(0) = -inf and collapse the score to 0).
    let mut log_sum = 0.0_f64;
    let mut n_used = 0_usize;
    for n in 0..max_n {
        if total_ngrams[n] == 0 {
            continue;
        }
        let precision = if total_matches[n] == 0 {
            // Smoothing: add-1
            1.0 / (total_ngrams[n] as f64 + 1.0)
        } else {
            total_matches[n] as f64 / total_ngrams[n] as f64
        };
        log_sum += precision.ln();
        n_used += 1;
    }

    if n_used == 0 {
        return 0.0;
    }

    let bp = if total_cand_len < total_ref_len {
        (1.0 - total_ref_len as f64 / total_cand_len as f64).exp()
    } else {
        1.0
    };

    let avg_log_precision = log_sum / n_used as f64;

    bp * avg_log_precision.exp()
}

/// Compute modified n-gram precision for a single sentence pair.
fn ngram_precision(reference: &[String], candidate: &[String], n: usize) -> (usize, usize) {
    if candidate.len() < n {
        return (0, 0);
    }

    // Count n-grams in reference
    let mut ref_ngrams: HashMap<Vec<String>, usize> = HashMap::new();
    for i in 0..=reference.len().saturating_sub(n) {
        let gram: Vec<String> = reference[i..i + n].to_vec();
        *ref_ngrams.entry(gram).or_insert(0) += 1;
    }

    // Count clipped n-gram matches in candidate
    let mut cand_ngrams: HashMap<Vec<String>, usize> = HashMap::new();
    for i in 0..=candidate.len().saturating_sub(n) {
        let gram: Vec<String> = candidate[i..i + n].to_vec();
        *cand_ngrams.entry(gram.clone()).or_insert(0) += 1;
    }

    let mut matches = 0;
    let mut total = 0;

    for (gram, cand_count) in &cand_ngrams {
        let max_ref = ref_ngrams.get(gram).copied().unwrap_or(0);
        matches += (*cand_count).min(max_ref);
        total += cand_count;
    }

    if total == 0 {
        (0, candidate.len().saturating_sub(n - 1))
    } else {
        (matches, total)
    }
}

/// Convert token IDs (usize) to string tokens for BLEU evaluation.
///
/// Uses a simple default mapping: token index to its string representation.
#[allow(dead_code)]
pub fn ids_to_tokens(ids: &[usize]) -> Vec<String> {
    ids.iter()
        .map(|&id| {
            if id == 0 {
                "<pad>".to_string()
            } else if id == 1 {
                "<bos>".to_string()
            } else if id == 2 {
                "<eos>".to_string()
            } else {
                format!("tok_{}", id)
            }
        })
        .collect()
}

/// Evaluate BLEU for a batch of predictions against references.
pub fn evaluate_bleu(references: &[Vec<usize>], predictions: &[Vec<usize>]) -> f64 {
    let ref_str: Vec<Vec<String>> = references
        .iter()
        .map(|r| {
            r.iter()
                .filter(|&&id| id != 0 && id != 1) // Remove pad and bos
                .map(|&id| {
                    if id == 2 {
                        "<eos>".to_string()
                    } else {
                        format!("tok_{}", id)
                    }
                })
                .collect()
        })
        .collect();

    let pred_str: Vec<Vec<String>> = predictions
        .iter()
        .map(|p| {
            p.iter()
                .filter(|&&id| id != 0 && id != 1) // Remove pad and bos
                .map(|&id| {
                    if id == 2 {
                        "<eos>".to_string()
                    } else {
                        format!("tok_{}", id)
                    }
                })
                .collect()
        })
        .collect();

    corpus_bleu(&ref_str, &pred_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bleu_perfect_match() {
        let ref_sent: Vec<String> = vec![
            "the".into(),
            "cat".into(),
            "sat".into(),
            "on".into(),
            "the".into(),
            "mat".into(),
        ];
        let cand_sent = ref_sent.clone();
        let score = bleu_score(&ref_sent, &cand_sent);
        assert!(
            (score - 1.0).abs() < 1e-6,
            "Perfect match should give BLEU=1.0, got {}",
            score
        );
    }

    #[test]
    fn test_bleu_no_match() {
        let ref_sent: Vec<String> = vec!["the".into(), "cat".into(), "sat".into()];
        let cand_sent: Vec<String> = vec!["xyz".into(), "abc".into(), "def".into()];
        let score = bleu_score(&ref_sent, &cand_sent);
        assert!(score < 0.5, "No match should give low BLEU");
    }

    #[test]
    fn test_bleu_partial_match() {
        let ref_sent: Vec<String> = vec![
            "the".into(),
            "cat".into(),
            "sat".into(),
            "on".into(),
            "the".into(),
            "mat".into(),
        ];
        let cand_sent: Vec<String> = vec![
            "the".into(),
            "cat".into(),
            "lay".into(),
            "on".into(),
            "the".into(),
            "rug".into(),
        ];
        let score = bleu_score(&ref_sent, &cand_sent);
        assert!(
            score > 0.0 && score < 1.0,
            "Partial match should give BLEU in (0,1), got {}",
            score
        );
    }

    #[test]
    fn test_bleu_brevity_penalty() {
        let ref_sent: Vec<String> = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        let cand_sent: Vec<String> = vec!["a".into(), "b".into()];
        let score = bleu_score(&ref_sent, &cand_sent);
        assert!(score < 1.0, "Shorter candidate should be penalized");
    }

    #[test]
    fn test_bleu_empty_candidate() {
        let ref_sent: Vec<String> = vec!["hello".into(), "world".into()];
        let cand_sent: Vec<String> = vec![];
        let score = bleu_score(&ref_sent, &cand_sent);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_corpus_bleu() {
        let refs = vec![
            vec!["the".into(), "cat".into()],
            vec!["hello".into(), "world".into()],
        ];
        let cands = vec![
            vec!["the".into(), "cat".into()],
            vec!["hello".into(), "world".into()],
        ];
        let score = corpus_bleu(&refs, &cands);
        // Short sequences may not get exact 1.0 due to brevity penalty mechanics
        assert!(
            score > 0.9,
            "Corpus BLEU for perfect match should be high, got {}",
            score
        );
    }

    #[test]
    fn test_ngram_precision_unigram() {
        let ref_sent: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let cand_sent: Vec<String> = vec!["a".into(), "b".into()];
        let (matches, total) = ngram_precision(&ref_sent, &cand_sent, 1);
        assert_eq!(matches, 2);
        assert_eq!(total, 2);
    }

    #[test]
    fn test_ngram_precision_bigram() {
        let ref_sent: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let cand_sent: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let (matches, total) = ngram_precision(&ref_sent, &cand_sent, 2);
        assert_eq!(matches, 2);
        assert_eq!(total, 2);
    }

    #[test]
    fn test_evaluate_bleu_ids() {
        let refs = vec![vec![3, 4, 5, 6]];
        let preds = vec![vec![3, 4, 5, 6]];
        let score = evaluate_bleu(&refs, &preds);
        assert!((score - 1.0).abs() < 1e-6);
    }
}
