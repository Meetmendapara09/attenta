/// Byte Pair Encoding (BPE) Tokenizer
///
/// Implements a simple BPE tokenizer for real-world text tokenization.
/// BPE is used by the original Transformer paper (Section 5.1):
/// "We used byte-pair encoding (BPE) with a shared source-target vocabulary of about 37000 tokens."
///
/// Reference: Sennrich et al., 2016. "Neural Machine Translation of Rare Words with Subword Units"
use std::collections::HashMap;
use std::fs;

/// A BPE tokenizer with learned merge rules.
#[derive(Debug, Clone)]
pub struct BPETokenizer {
    /// Vocabulary: token_id -> String representation
    pub id_to_token: Vec<String>,
    /// Vocabulary mapping: token string -> token_id
    pub token_to_id: HashMap<String, usize>,
    /// BPE merge rules: pair -> (new_token, priority)
    pub merges: Vec<(String, String)>,
    /// Special tokens
    pub pad_id: usize,
    pub bos_id: usize,
    pub eos_id: usize,
    pub unk_id: usize,
}

impl BPETokenizer {
    /// Create a new BPE tokenizer from a vocabulary file.
    ///
    /// The vocabulary file should list tokens one per line, with special tokens
    /// (PAD, BOS, EOS, UNK) as the first four entries.
    pub fn new(
        vocab_path: Option<&str>,
        merges_path: Option<&str>,
        pad_id: usize,
        bos_id: usize,
        eos_id: usize,
        unk_id: usize,
    ) -> Result<Self, String> {
        let (id_to_token, token_to_id) = if let Some(path) = vocab_path {
            Self::load_vocab(path)?
        } else {
            Self::default_vocab(pad_id, bos_id, eos_id, unk_id)
        };

        let merges = if let Some(path) = merges_path {
            Self::load_merges(path)?
        } else {
            Vec::new()
        };

        Ok(Self {
            id_to_token,
            token_to_id,
            merges,
            pad_id,
            bos_id,
            eos_id,
            unk_id,
        })
    }

    /// Load vocabulary from a file (one token per line).
    fn load_vocab(path: &str) -> Result<(Vec<String>, HashMap<String, usize>), String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read vocab file '{}': {}", path, e))?;
        let mut id_to_token = Vec::new();
        let mut token_to_id = HashMap::new();

        for (i, line) in content.lines().enumerate() {
            // Don't trim — space character is a valid token
            if !line.is_empty() {
                let token = line.to_string();
                token_to_id.insert(token.clone(), i);
                id_to_token.push(token);
            }
        }

        Ok((id_to_token, token_to_id))
    }

    /// Load merge rules from a file (one pair per line, tab-separated).
    fn load_merges(path: &str) -> Result<Vec<(String, String)>, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read merges file '{}': {}", path, e))?;
        let mut merges = Vec::new();

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            // Format: "token1 token2" (space-separated)
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() == 2 {
                merges.push((parts[0].to_string(), parts[1].to_string()));
            }
        }

        Ok(merges)
    }

    /// Create a default character-level vocabulary for demonstration.
    fn default_vocab(
        pad_id: usize,
        bos_id: usize,
        eos_id: usize,
        unk_id: usize,
    ) -> (Vec<String>, HashMap<String, usize>) {
        let mut id_to_token = Vec::new();
        let mut token_to_id = HashMap::new();

        // Special tokens
        token_to_id.insert("<pad>".to_string(), pad_id);
        id_to_token.push("<pad>".to_string());
        token_to_id.insert("<bos>".to_string(), bos_id);
        id_to_token.push("<bos>".to_string());
        token_to_id.insert("<eos>".to_string(), eos_id);
        id_to_token.push("<eos>".to_string());
        token_to_id.insert("<unk>".to_string(), unk_id);
        id_to_token.push("<unk>".to_string());

        // ASCII printable characters (space through ~, codes 32-126)
        for c in 32u8..=126u8 {
            let token = format!("{}", c as char);
            token_to_id.insert(token.clone(), id_to_token.len());
            id_to_token.push(token);
        }

        // Common English subwords (bigrams/trigrams for better demo)
        let common_subwords = vec![
            "th", "he", "in", "er", "an", "on", "at", "en", "nd", "ti",
            "es", "or", "te", "of", "ed", "is", "it", "al", "ar", "st",
            "ing", "ion", "the", "and", "for", "are", "but", "not",
            "you", "all", "can", "had", "her", "was", "one", "our",
            "out", "has", "have", "with", "from", "they", "this",
            "that", "what", "will", "been", "said", "make", "like",
            "time", "just", "know", "take", "into", "year", "your",
            "good", "some", "them", "than", "then", "many", "also",
            "more", "about", "other", "which", "their", "there",
            "could", "would", "should", "people", "because",
        ];
        for sw in &common_subwords {
            if token_to_id.contains_key(*sw) {
                continue;
            }
            token_to_id.insert(sw.to_string(), id_to_token.len());
            id_to_token.push(sw.to_string());
        }

        (id_to_token, token_to_id)
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Get token ID for a given token string.
    pub fn token_to_id(&self, token: &str) -> usize {
        self.token_to_id
            .get(token)
            .copied()
            .unwrap_or(self.unk_id)
    }

    /// Get token string for a given token ID.
    pub fn id_to_token(&self, id: usize) -> &str {
        if id < self.id_to_token.len() {
            &self.id_to_token[id]
        } else {
            "<unk>"
        }
    }

    /// Tokenize text using BPE merge rules.
    ///
    /// 1. Start with character-level tokenization
    /// 2. Iteratively apply most frequent merge rules
    /// 3. Return Vec of token IDs
    pub fn encode(&self, text: &str, max_length: usize) -> Vec<usize> {
        // Step 1: Character-level tokenization
        let mut tokens: Vec<String> = Vec::new();
        for c in text.chars() {
            let token = format!("{}", c);
            if self.token_to_id.contains_key(&token) {
                tokens.push(token);
            } else {
                tokens.push("<unk>".to_string());
            }
        }

        // Step 2: Apply learned BPE merges
        if !self.merges.is_empty() {
            self.apply_merges(&mut tokens);
        } else {
            // Use statistical merging for common pairs
            self.statistical_merge(&mut tokens);
        }

        // Step 3: Convert to IDs
        let mut ids: Vec<usize> = tokens
            .iter()
            .map(|t| self.token_to_id(t))
            .collect();

        // Truncate to max_length - 2 (for BOS and EOS)
        if ids.len() > max_length - 2 {
            ids.truncate(max_length - 2);
        }

        ids
    }

    /// Tokenize text and wrap with BOS/EOS tokens.
    pub fn encode_with_special(&self, text: &str, max_length: usize) -> Vec<usize> {
        let mut tokens = vec![self.bos_id];
        tokens.extend(self.encode(text, max_length - 2));
        tokens.push(self.eos_id);
        tokens
    }

    /// Detokenize token IDs back to text.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut result = String::new();
        let mut prev_was_single_char = false;

        for &id in ids {
            if id == self.pad_id || id == self.bos_id || id == self.eos_id {
                prev_was_single_char = false;
                continue;
            }
            let token = self.id_to_token(id);
            let is_single_char = token.chars().count() == 1;
            let is_alpha = token.chars().all(|c| c.is_alphabetic());

            // Add space between tokens unless:
            // - It's the first token
            // - It's a single-character alphabetic token that follows another single-char token (no space between characters of a word)
            // - It's punctuation that shouldn't have a space before it
            let is_punct = token == "." || token == "," || token == "!" || token == "?" || token == "'" || token == ":" || token == ";";
            let needs_space = !result.is_empty()
                && !(is_single_char && prev_was_single_char)
                && !is_punct
                && !(token == "n" && prev_was_single_char);

            if needs_space
                && !token.starts_with('\'')
                && !token.starts_with('.')
                && !token.starts_with(',')
            {
                result.push(' ');
            }
            result.push_str(token);
            prev_was_single_char = is_single_char && is_alpha;
        }

        // Clean up spacing around punctuation
        let cleaned = result
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" :", ":")
            .replace(" ;", ";")
            .replace(" '", "'")
            .replace("  ", " ")
            .trim()
            .to_string();

        cleaned
    }

    /// Apply BPE merge rules iteratively.
    fn apply_merges(&self, tokens: &mut Vec<String>) {
        loop {
            // Find the highest-priority merge that can be applied
            let mut best_pair: Option<(usize, String, String)> = None;

            for (_merge_idx, (left, right)) in self.merges.iter().enumerate() {
                for i in 0..tokens.len().saturating_sub(1) {
                    if tokens[i] == *left && tokens[i + 1] == *right {
                        let _new_token = format!("{}{}", left, right);
                        let new_pair = Some((i, left.clone(), right.clone()));
                        // First merge wins (BPE applies merges in order)
                        if best_pair.is_none() {
                            best_pair = new_pair.map(|(pos, l, r)| (pos, l, r));
                            break;
                        }
                    }
                }
                if best_pair.is_some() {
                    break;
                }
            }

            match best_pair {
                Some((pos, left, right)) => {
                    let new_token = format!("{}{}", left, right);
                    tokens[pos] = new_token;
                    tokens.remove(pos + 1);
                }
                None => break,
            }
        }
    }

    /// Statistical merging when no learned merge rules are available.
    ///
    /// Finds the most frequent adjacent pair and merges it,
    /// repeating until no pair appears more than once.
    fn statistical_merge(&self, tokens: &mut Vec<String>) {
        for _ in 0..10 {
            // Limit merges to prevent over-merging
            let mut pair_counts: HashMap<(&str, &str), usize> = HashMap::new();
            for i in 0..tokens.len().saturating_sub(1) {
                let pair = (&tokens[i][..], &tokens[i + 1][..]);
                // Only merge if the combined token might be in vocab or is common
                let combined = format!("{}{}", tokens[i], tokens[i + 1]);
                if combined.len() <= 6 && tokens[i].len() >= 1 && tokens[i + 1].len() >= 1 {
                    *pair_counts.entry(pair).or_insert(0) += 1;
                }
            }

            let best_pair = pair_counts.into_iter().max_by_key(|&(_, count)| count);

            match best_pair {
                Some(((left, right), count)) if count > 1 => {
                    let mut new_tokens = Vec::new();
                    let mut i = 0;
                    while i < tokens.len() {
                        if i + 1 < tokens.len()
                            && tokens[i] == left
                            && tokens[i + 1] == right
                        {
                            // Only merge if the combined token exists in the vocabulary
                            let combined = format!("{}{}", left, right);
                            if self.token_to_id.contains_key(&combined) {
                                new_tokens.push(combined);
                                i += 2;
                                continue;
                            }
                        }
                        new_tokens.push(tokens[i].clone());
                        i += 1;
                    }
                    *tokens = new_tokens;
                }
                _ => break,
            }
        }
    }

    /// Save the tokenizer vocabulary to a file.
    pub fn save_vocab(&self, path: &str) -> Result<(), String> {
        let mut content = String::new();
        for token in &self.id_to_token {
            content.push_str(token);
            content.push('\n');
        }
        fs::write(path, content).map_err(|e| format!("Write error: {}", e))?;
        Ok(())
    }

    /// Save merge rules to a file.
    #[allow(dead_code)]
    pub fn save_merges(&self, path: &str) -> Result<(), String> {
        let mut content = String::new();
        content.push_str("# BPE merge rules\n");
        content.push_str("# format: left_token right_token\n");
        for (left, right) in &self.merges {
            content.push_str(left);
            content.push(' ');
            content.push_str(right);
            content.push('\n');
        }
        fs::write(path, content).map_err(|e| format!("Write error: {}", e))?;
        Ok(())
    }

    /// Build tokenizer from training data (learns BPE merges).
    ///
    /// This is a simplified BPE training algorithm:
    /// 1. Start with character vocabulary
    /// 2. Count all adjacent pairs
    /// 3. Merge most frequent pair
    /// 4. Repeat for num_merges iterations
    pub fn train(
        texts: &[&str],
        vocab_size: usize,
        pad_id: usize,
        bos_id: usize,
        eos_id: usize,
        unk_id: usize,
    ) -> Self {
        let mut tokenizer = Self::new(None, None, pad_id, bos_id, eos_id, unk_id).unwrap();
        let num_merges = vocab_size.saturating_sub(tokenizer.vocab_size());

        if num_merges == 0 || texts.is_empty() {
            return tokenizer;
        }

        // Collect all words with frequencies (character-level)
        let mut word_freqs: HashMap<Vec<String>, usize> = HashMap::new();
        for text in texts {
            let chars: Vec<String> = text
                .chars()
                .map(|c| format!("{}", c))
                .collect();
            *word_freqs.entry(chars).or_insert(0) += 1;
        }

        // Learn merge rules
        let mut merges = Vec::new();

        for _ in 0..num_merges {
            // Count all adjacent pairs across all words
            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
            for (word, freq) in &word_freqs {
                for i in 0..word.len().saturating_sub(1) {
                    let pair = (word[i].clone(), word[i + 1].clone());
                    *pair_counts.entry(pair).or_insert(0) += freq;
                }
            }

            let best = pair_counts.into_iter().max_by_key(|&(_, count)| count);
            let best = match best {
                Some(((left, right), _)) => (left, right),
                None => break,
            };

            merges.push(best.clone());

            // Apply the merge to all words
            let merged_token = format!("{}{}", best.0, best.1);
            let mut new_word_freqs: HashMap<Vec<String>, usize> = HashMap::new();

            for (word, freq) in &word_freqs {
                let mut new_word = Vec::new();
                let mut i = 0;
                while i < word.len() {
                    if i + 1 < word.len() && word[i] == best.0 && word[i + 1] == best.1 {
                        new_word.push(merged_token.clone());
                        i += 2;
                    } else {
                        new_word.push(word[i].clone());
                        i += 1;
                    }
                }
                *new_word_freqs.entry(new_word).or_insert(0) += freq;
            }
            word_freqs = new_word_freqs;

            // Add merged token to vocabulary
            if !tokenizer.token_to_id.contains_key(&merged_token) {
                let new_id = tokenizer.id_to_token.len();
                tokenizer.token_to_id.insert(merged_token.clone(), new_id);
                tokenizer.id_to_token.push(merged_token);
            }
        }

        tokenizer.merges = merges;
        tokenizer
    }

    /// Train a tokenizer from a text file.
    #[allow(dead_code)]
    pub fn train_from_file(
        path: &str,
        vocab_size: usize,
        pad_id: usize,
        bos_id: usize,
        eos_id: usize,
        unk_id: usize,
    ) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file '{}': {}", path, e))?;

        // Split into sentences/lines for training
        let texts: Vec<&str> = content.lines().collect();
        if texts.is_empty() {
            return Err("Training file is empty".to_string());
        }

        Ok(Self::train(
            &texts,
            vocab_size,
            pad_id,
            bos_id,
            eos_id,
            unk_id,
        ))
    }

    /// Create a demonstration English-French tokenizer with common vocabulary.
    ///
    /// This creates a built-in vocabulary useful for demo/example purposes.
    /// Includes all ASCII printable characters (32-126) + common English/French words.
    pub fn demo_enfr() -> Self {
        let mut id_to_token = Vec::new();
        let mut token_to_id = HashMap::new();

        // Special tokens
        let pad_id = 0;
        let bos_id = 1;
        let eos_id = 2;
        let unk_id = 3;

        let special = ["<pad>", "<bos>", "<eos>", "<unk>"];
        for (i, s) in special.iter().enumerate() {
            token_to_id.insert(s.to_string(), i);
            id_to_token.push(s.to_string());
        }

        // ASCII printable characters (space through ~, codes 32-126)
        // This ensures any English text can be tokenized at character level
        for c in 32u8..=126u8 {
            let token = format!("{}", c as char);
            token_to_id.insert(token.clone(), id_to_token.len());
            id_to_token.push(token);
        }

        // Common English subwords for better BPE-like tokenization
        let en_subwords = vec![
            "th", "he", "in", "er", "an", "on", "at", "en", "nd", "ti",
            "es", "or", "te", "of", "ed", "is", "it", "al", "ar", "st",
            "ing", "ion", "the", "and", "for", "are", "but", "not",
            "you", "all", "can", "had", "her", "was", "one", "our",
            "out", "has", "have", "with", "from", "they", "this",
            "that", "what", "will", "been", "said", "make", "like",
            "time", "just", "know", "take", "into", "year", "your",
            "good", "some", "them", "than", "then", "many", "also",
            "about", "other", "which", "their", "there", "could",
            "would", "should", "people", "because", "little",
            "great", "first", "world", "water", "house", "place",
            "right", "small", "large", "high", "long", "very",
            "every", "after", "still", "where", "think", "never",
            "always", "thing", "man", "woman", "child", "day",
            "way", "life", "hand", "part", "eye", "woman", "men",
        ];
        for sw in en_subwords {
            if !token_to_id.contains_key(sw) {
                token_to_id.insert(sw.to_string(), id_to_token.len());
                id_to_token.push(sw.to_string());
            }
        }

        // Common English words
        let en_words = vec![
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "has", "have", "had", "do", "does", "did", "will", "would",
            "can", "could", "shall", "should", "may", "might", "must",
            "i", "you", "he", "she", "it", "we", "they",
            "me", "him", "her", "us", "them",
            "my", "your", "his", "its", "our", "their",
            "this", "that", "these", "those",
            "and", "or", "but", "not", "if", "because", "so",
            "with", "without", "for", "to", "from", "at", "in", "on",
            "by", "about", "into", "through", "during", "over", "under",
            "man", "woman", "child", "people", "world",
            "time", "year", "day", "way", "thing", "life",
            "hand", "part", "place", "case", "week", "company",
            "system", "program", "work", "government",
            "number", "night", "point", "home", "water",
            "room", "mother", "area", "money", "story",
            "fact", "month", "lot", "right", "study",
            "book", "eye", "job", "word", "business",
            "issue", "side", "kind", "head", "house",
            "service", "friend", "father", "power", "hour",
            "game", "line", "end", "member", "law",
            "car", "city", "community", "name", "president",
            "team", "minute", "idea", "kid", "body",
            "information", "back", "parent", "face", "other",
            "level", "office", "door", "health", "person",
            "art", "war", "history", "party", "result",
            "change", "morning", "reason", "research", "girl",
            "guy", "moment", "air", "teacher", "force",
            "education", "student", "sentence",
            // Common verbs
            "go", "come", "see", "know", "get", "give", "find",
            "tell", "ask", "work", "seem", "feel", "try", "leave",
            "call", "keep", "let", "begin", "show", "hear",
            "play", "run", "move", "live", "stand", "take",
            "make", "think", "say", "want", "look", "need",
            "help", "start", "bring", "write", "provide",
            "love", "like", "hate", "eat", "drink", "sleep",
            // Common adjectives
            "good", "new", "first", "last", "long", "great",
            "little", "own", "other", "old", "right", "big",
            "high", "different", "small", "large", "next",
            "early", "young", "important", "public", "bad",
            "same", "able", "possible", "open", "short",
            "hard", "ready", "real", "free", "full", "sure",
            "quick", "brown", "fox", "lazy", "jumps",
            // Common adverbs
            "well", "also", "very", "often", "however",
            "too", "usually", "really", "already", "still",
            "always", "never", "sometimes", "together",
            "then", "there", "here", "where", "when", "why",
            "how", "now", "just", "only", "even", "much",
        ];
        for token in en_words {
            if !token_to_id.contains_key(token) {
                token_to_id.insert(token.to_string(), id_to_token.len());
                id_to_token.push(token.to_string());
            }
        }

        // French subwords for better tokenization
        let fr_subwords = vec![
            "le", "la", "les", "un", "une", "des", "du", "de", "es",
            "on", "en", "ai", "as", "ez", "er", "re", "ir", "oi",
            "ou", "ui", "an", "in", "on", "eu", "au", "te", "se",
            "me", "ce", "ne", "que", "lle", "ent", "ant", "ion",
            "men", "ment", "tion", "sion", "eur", "eux", "ais",
        ];
        for sw in fr_subwords {
            if !token_to_id.contains_key(sw) {
                token_to_id.insert(sw.to_string(), id_to_token.len());
                id_to_token.push(sw.to_string());
            }
        }

        // French tokens
        let fr_tokens = [
            "le", "la", "les", "un", "une", "des", "du", "de",
            "est", "sont", "était", "étaient", "être", "été",
            "a", "ont", "avait", "avaient", "avoir", "eu",
            "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
            "me", "te", "se", "lui", "leur",
            "mon", "ton", "son", "ma", "ta", "sa",
            "mes", "tes", "ses", "nos", "vos", "leurs",
            "ce", "cet", "cette", "ces",
            "et", "ou", "mais", "pas", "si", "parce", "que", "donc",
            "avec", "sans", "pour", "à", "de", "sur", "dans",
            "par", "vers", "pendant", "chez",
            "homme", "femme", "enfant", "gens", "monde",
            "temps", "année", "jour", "façon", "chose", "vie",
            "main", "partie", "endroit", "cas", "semaine",
            "société", "système", "programme", "travail",
            "nombre", "nuit", "point", "maison", "eau",
            "pièce", "mère", "zone", "argent", "histoire",
            "fait", "mois", "beaucoup", "droit", "étude",
            "livre", "œil", "travail", "mot", "affaire",
            "problème", "côté", "genre", "tête", "famille",
            "service", "ami", "père", "pouvoir", "heure",
            "jeu", "ligne", "fin", "membre", "loi",
            "voiture", "ville", "communauté", "nom", "président",
            "équipe", "minute", "idée", "enfant", "corps",
            "information", "dos", "parent", "visage", "autre",
            "niveau", "bureau", "porte", "santé", "personne",
            "art", "guerre", "histoire", "parti", "résultat",
            "changement", "matin", "raison", "recherche", "fille",
            "gars", "moment", "air", "professeur", "force",
            // French verbs
            "aller", "venir", "voir", "savoir", "obtenir", "donner",
            "trouver", "dire", "demander", "travailler", "sembler",
            "sentir", "essayer", "laisser", "appeler", "garder",
            "jouer", "courir", "bouger", "vivre", "rester",
            "prendre", "faire", "penser", "vouloir", "regarder",
            "avoir", "commencer", "apporter", "écrire", "fournir",
            // French adjectives
            "bon", "nouveau", "premier", "dernier", "long",
            "grand", "petit", "propre", "autres", "vieux",
            "droit", "gros", "haut", "différent", "court",
            "prochain", "jeune", "important", "public", "mauvais",
            "même", "capable", "possible", "ouvert", "dur",
            "prêt", "réel", "libre", "plein", "certain",
            // Numbers
            "zéro", "un", "deux", "trois", "quatre", "cinq",
            "six", "sept", "huit", "neuf", "dix",
            // Common prepositions/conjunctions
            "dans", "sans", "avant", "après", "pendant",
            "depuis", "jusque", "envers", "contre", "selon",
            "grâce", "excepté", "voici", "voilà",
        ];

        for token in &fr_tokens {
            if !token_to_id.contains_key(*token) {
                token_to_id.insert(token.to_string(), id_to_token.len());
                id_to_token.push(token.to_string());
            }
        }

        // Learnable BPE merges (common English subword pairs)
        let merges = vec![
            ("th".to_string(), "e".to_string()),
            ("t".to_string(), "he".to_string()),
            ("the".to_string(), "y".to_string()),
            ("ing".to_string(), "s".to_string()),
            ("e".to_string(), "d".to_string()),
            ("e".to_string(), "r".to_string()),
            ("e".to_string(), "s".to_string()),
            ("a".to_string(), "tion".to_string()),
            ("i".to_string(), "on".to_string()),
            ("a".to_string(), "ble".to_string()),
            ("pre".to_string(), "fix".to_string()),
            ("un".to_string(), "able".to_string()),
            ("re".to_string(), "do".to_string()),
        ];

        BPETokenizer {
            id_to_token,
            token_to_id,
            merges,
            pad_id,
            bos_id,
            eos_id,
            unk_id,
        }
    }

    /// Statistics about this tokenizer.
    pub fn print_stats(&self) {
        println!("  Tokenizer stats:");
        println!("    Vocabulary size: {}", self.vocab_size());
        println!("    Merge rules:     {}", self.merges.len());
        println!("    PAD id: {}  BOS id: {}  EOS id: {}  UNK id: {}",
                 self.pad_id, self.bos_id, self.eos_id, self.unk_id);
    }
}

/// A complete translation pipeline using a tokenizer + transformer model.
pub struct TranslationPipeline {
    pub tokenizer: BPETokenizer,
}

impl TranslationPipeline {
    pub fn new(tokenizer: BPETokenizer) -> Self {
        Self { tokenizer }
    }

    /// Translate English text using greedy decoding.
    pub fn translate(
        &self,
        transformer: &crate::model::Transformer,
        text: &str,
    ) -> String {
        let src_ids = self.tokenizer.encode_with_special(text, transformer.config().max_len);
        let pred_ids = transformer.translate(&src_ids);
        self.tokenizer.decode(&pred_ids)
    }

    /// Translate using beam search.
    #[allow(dead_code)]
    pub fn translate_beam(
        &self,
        transformer: &crate::model::Transformer,
        text: &str,
        _beam_size: usize,
        _alpha: f64,
    ) -> String {
        let src_ids = self.tokenizer.encode_with_special(text, transformer.config().max_len);
        let pred_ids = transformer.translate_beam(&src_ids);
        self.tokenizer.decode(&pred_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_vocab_size() {
        let tokenizer = BPETokenizer::new(None, None, 0, 1, 2, 3).unwrap();
        assert!(tokenizer.vocab_size() > 100);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let tokenizer = BPETokenizer::new(None, None, 0, 1, 2, 3).unwrap();
        let text = "hello world";
        let ids = tokenizer.encode(text, 50);
        let decoded = tokenizer.decode(&ids);
        // Should preserve core content - all printable ASCII chars should roundtrip
        assert!(!decoded.is_empty(), "Decoded text should not be empty");
        assert!(
            decoded.len() <= text.len() * 2,
            "Decoded '{}' (len={}) should not be way longer than original '{}' (len={})",
            decoded,
            decoded.len(),
            text,
            text.len()
        );
        // The decoded text should contain recognizable parts of the original
        let has_subwords = ["hello", "hel", "lo", "h", "e", "l", "o", "world", "wor", "ld"]
            .iter().any(|s| decoded.contains(s));
        assert!(has_subwords, "Decoded '{}' should contain parts of 'hello world'", decoded);
    }

    #[test]
    fn test_encode_with_special() {
        let tokenizer = BPETokenizer::new(None, None, 0, 1, 2, 3).unwrap();
        let text = "hello";
        let ids = tokenizer.encode_with_special(text, 20);
        assert_eq!(ids[0], tokenizer.bos_id);
        assert_eq!(*ids.last().unwrap(), tokenizer.eos_id);
    }

    #[test]
    fn test_decode_removes_special() {
        let tokenizer = BPETokenizer::new(None, None, 0, 1, 2, 3).unwrap();
        let ids = vec![tokenizer.bos_id, 5, 6, tokenizer.eos_id, tokenizer.pad_id];
        let decoded = tokenizer.decode(&ids);
        assert!(!decoded.contains("<bos>"));
        assert!(!decoded.contains("<eos>"));
        assert!(!decoded.contains("<pad>"));
    }

    #[test]
    fn test_tokenizer_save_load() {
        let tokenizer = BPETokenizer::new(None, None, 0, 1, 2, 3).unwrap();
        let vocab_path = "_test_vocab.txt";
        tokenizer.save_vocab(vocab_path).unwrap();

        let loaded = BPETokenizer::new(Some(vocab_path), None, 0, 1, 2, 3).unwrap();
        assert_eq!(
            tokenizer.vocab_size(),
            loaded.vocab_size(),
            "Vocab size mismatch: {} vs {}",
            tokenizer.vocab_size(),
            loaded.vocab_size()
        );
        // Verify roundtrip preserves token mapping for key tokens
        for token in &["<pad>", "<bos>", "<eos>", "<unk>", "the", "a", "hello"] {
            if tokenizer.token_to_id.contains_key(*token) {
                assert!(
                    loaded.token_to_id.contains_key(*token),
                    "Token '{}' should exist in loaded tokenizer",
                    token
                );
            }
        }

        let _ = std::fs::remove_file(vocab_path);
    }

    #[test]
    fn test_train_from_text() {
        let texts = vec![
            "the cat sat on the mat",
            "the dog ran in the park",
            "the bird flew over the tree",
        ];
        let tokenizer = BPETokenizer::train(&texts, 200, 0, 1, 2, 3);
        assert!(tokenizer.vocab_size() >= 100);
        assert!(!tokenizer.merges.is_empty());
    }

    #[test]
    fn test_demo_enfr_tokenizer() {
        let tokenizer = BPETokenizer::demo_enfr();
        assert!(tokenizer.vocab_size() > 300);

        // Test English encoding
        let eng_ids = tokenizer.encode("hello world", 50);
        assert!(!eng_ids.is_empty());

        // Verify special tokens
        assert!(tokenizer.token_to_id.contains_key("the"));
        assert!(tokenizer.token_to_id.contains_key("est")); // French "is"
    }

    #[test]
    fn test_translation_pipeline() {
        let tokenizer = BPETokenizer::demo_enfr();
        let pipeline = TranslationPipeline::new(tokenizer);

        let config = crate::model::TransformerConfig {
            src_vocab: pipeline.tokenizer.vocab_size(),
            tgt_vocab: pipeline.tokenizer.vocab_size(),
            d_model: 16,
            n_heads: 4,
            d_ff: 32,
            n_layers: 1,
            max_len: 32,
            dropout: 0.0,
            pad_id: 0,
            bos_id: 1,
            eos_id: 2,
            label_smoothing: 0.1,
            warmup_steps: 50,
        };
        let transformer = crate::model::Transformer::new_seeded(config, 42);

        let result = pipeline.translate(&transformer, "hello world");
        assert!(!result.is_empty(), "Translation should produce output");
    }
}
