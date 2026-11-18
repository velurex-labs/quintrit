//! HuggingFace `tokenizers` crate integration for sub-millisecond BPE encoding and decoding.

use ahash::AHashMap;
use std::path::Path;
use tokenizers::models::bpe::BPE;
use tokenizers::{AddedToken, Tokenizer};

/// GGUF token type of a control token (`tokenizer.ggml.token_type`, llama.cpp `LLAMA_TOKEN_TYPE_CONTROL`), such as
/// `<|endoftext|>`.
pub const GGUF_TOKEN_TYPE_CONTROL: i32 = 3;

/// GGUF token type of a user-defined token (`LLAMA_TOKEN_TYPE_USER_DEFINED`), such as `<think>` in the Qwen
/// vocabularies or the line feed in Falcon3's.
pub const GGUF_TOKEN_TYPE_USER_DEFINED: i32 = 4;

/// High-performance tokenizer wrapper providing BPE text encoding and decoding.
pub struct QuintritTokenizer {
    inner: Tokenizer,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

impl QuintritTokenizer {
    /// Loads a tokenizer from a HuggingFace `tokenizer.json` specification file.
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        bos_id: Option<u32>,
        eos_id: Option<u32>,
    ) -> Result<Self, String> {
        let tokenizer = Tokenizer::from_file(path).map_err(|e| e.to_string())?;
        Ok(Self {
            inner: tokenizer,
            bos_token_id: bos_id,
            eos_token_id: eos_id,
        })
    }

    /// Constructs a HuggingFace BPE tokenizer directly from GGUF metadata tokens and merges.
    ///
    /// No token is treated as a special token: text such as `<|endoftext|>` is split into ordinary BPE pieces. Use
    /// [`QuintritTokenizer::from_gguf_metadata_with_types`] to give control tokens their own ids.
    pub fn from_gguf_metadata(
        tokens: &[String],
        merges: Option<&[String]>,
        bos_id: Option<u32>,
        eos_id: Option<u32>,
    ) -> Result<Self, String> {
        Self::from_gguf_metadata_with_types(tokens, merges, None, bos_id, eos_id)
    }

    /// Like [`QuintritTokenizer::from_gguf_metadata`], but also registers every control token
    /// ([`GGUF_TOKEN_TYPE_CONTROL`] in `token_types`, the `tokenizer.ggml.token_type` array) as a special token.
    ///
    /// The text of a control token is then encoded as its single id, as the reference runtime does when it parses
    /// special tokens (the Falcon3 chat template ends every assistant turn with `<|endoftext|>`, for example), and
    /// decoding skips it.
    pub fn from_gguf_metadata_with_types(
        tokens: &[String],
        merges: Option<&[String]>,
        token_types: Option<&[i32]>,
        bos_id: Option<u32>,
        eos_id: Option<u32>,
    ) -> Result<Self, String> {
        Self::build(tokens, merges, token_types, PreTokenizerKind::Gpt2, bos_id, eos_id)
    }

    /// Like [`QuintritTokenizer::from_gguf_metadata_with_types`], with the pre-tokenizer selected by
    /// `tokenizer.ggml.pre` (`pre`), as llama.cpp selects it: the text is split by the regular expressions of that type
    /// before byte-level BPE, and for the types that llama.cpp marks `ignore_merges` (the LLaMA 3 and Tekken families)
    /// a piece that is a vocabulary entry as a whole is taken as one token. `None` keeps the GPT-2 split.
    pub fn from_gguf_metadata_with_pre(
        tokens: &[String],
        merges: Option<&[String]>,
        token_types: Option<&[i32]>,
        pre: Option<&str>,
        bos_id: Option<u32>,
        eos_id: Option<u32>,
    ) -> Result<Self, String> {
        Self::build(tokens, merges, token_types, PreTokenizerKind::from_gguf(pre)?, bos_id, eos_id)
    }

    fn build(
        tokens: &[String],
        merges: Option<&[String]>,
        token_types: Option<&[i32]>,
        pre_kind: PreTokenizerKind,
        bos_id: Option<u32>,
        eos_id: Option<u32>,
    ) -> Result<Self, String> {
        let mut vocab: AHashMap<String, u32> = AHashMap::with_capacity(tokens.len());
        for (i, token) in tokens.iter().enumerate() {
            let key = token.replace('\n', "Ċ").replace('\r', "č").replace('\t', "ċ");
            if key != *token {
                // A token stored with raw control characters (Falcon3's user-defined "\n") also keeps its raw text,
                // so that registering it as an added token below finds this id instead of creating a new one.
                vocab.entry(token.clone()).or_insert(i as u32);
            }
            vocab.insert(key, i as u32);
        }

        let mut parsed_merges: Vec<(String, String)> = Vec::new();
        if let Some(merges_list) = merges {
            parsed_merges.reserve(merges_list.len());
            for m in merges_list {
                let parts: Vec<&str> = m.split_whitespace().collect();
                if parts.len() == 2 {
                    parsed_merges.push((parts[0].to_string(), parts[1].to_string()));
                }
            }
        }

        let bpe_builder = BPE::builder()
            .vocab_and_merges(vocab, parsed_merges)
            .ignore_merges(pre_kind.ignore_merges());
        let bpe_model = bpe_builder.build().map_err(|e| e.to_string())?;
        let mut tokenizer = Tokenizer::new(bpe_model);

        // Configure ByteLevel pre-tokenizer and decoder for GPT-2/Falcon/LLaMA BPE compatibility
        tokenizer.with_pre_tokenizer(Some(pre_kind.build()?));
        tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));

        if let Some(types) = token_types {
            let specials: Vec<AddedToken> = tokens
                .iter()
                .zip(types)
                .filter(|(text, ty)| **ty == GGUF_TOKEN_TYPE_CONTROL && !text.is_empty())
                .map(|(text, _)| AddedToken::from(text.clone(), true))
                .collect();
            // An added token whose text is already in the vocabulary keeps that id.
            tokenizer.add_special_tokens(&specials);
            // User-defined tokens are matched as a whole as well (llama.cpp splits the text at them whether or not it
            // parses special tokens), but they decode to their text.
            let user: Vec<AddedToken> = tokens
                .iter()
                .zip(types)
                .filter(|(text, ty)| **ty == GGUF_TOKEN_TYPE_USER_DEFINED && !text.is_empty())
                .map(|(text, _)| AddedToken::from(text.clone(), false))
                .collect();
            tokenizer.add_tokens(&user);
        }

        Ok(Self {
            inner: tokenizer,
            bos_token_id: bos_id,
            eos_token_id: eos_id,
        })
    }

    /// Encodes text into a vector of token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self.inner.encode(text, false) {
            Ok(encoding) => encoding.get_ids().to_vec(),
            Err(_) => Vec::new(),
        }
    }

    /// Decodes a sequence of token IDs back into human-readable text.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let s = self.inner.decode(token_ids, true).unwrap_or_default();
        s.replace('Ġ', " ").replace('Ċ', "\n")
    }

    /// Decodes a single token ID into a string fragment.
    ///
    /// A byte-level token can hold only part of a multi-byte UTF-8 character, which comes back as U+FFFD. Use
    /// [`StreamDecoder`] to decode generated tokens one at a time.
    pub fn decode_token(&self, token_id: u32) -> String {
        let s = self.inner.decode(&[token_id], true).unwrap_or_else(|_| {
            self.inner.id_to_token(token_id).unwrap_or_default()
        });
        s.replace('Ġ', " ").replace('Ċ', "\n")
    }

    /// Returns the Beginning-of-Sequence (BOS) token ID.
    #[inline]
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// Returns the End-of-Sequence (EOS) token ID.
    #[inline]
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Total vocabulary size.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Access inner tokenizer
    pub fn inner(&self) -> &Tokenizer {
        &self.inner
    }
}

/// Pre-tokenizer families, keyed by the GGUF metadata `tokenizer.ggml.pre` like llama.cpp's `llama_vocab::load`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreTokenizerKind {
    /// GPT-2 byte-level split (also used when the key is absent and the caller does not ask for llama.cpp's default).
    Gpt2,
    /// llama.cpp's `default` type (the key is `default` or absent): four regular expressions applied in sequence.
    Default,
    /// LLaMA 3 split (`llama3`, `llama-bpe`, `falcon3`, ...), with `ignore_merges`.
    Llama3,
    /// Qwen2 split (`qwen2`): like LLaMA 3, but digits one by one.
    Qwen2,
    /// Qwen3.5 split (`qwen35`): like Qwen2, letters include combining marks.
    Qwen35,
    /// Tekken split (`tekken`, `pixtral`), with `ignore_merges`.
    Tekken,
    /// Falcon-E (`falcon_e`, not known to llama.cpp): the Hugging Face tokenizer of the model, a contiguous
    /// punctuation split, the Tekken expression and a split of every digit.
    FalconE,
}

const GPT2_REGEX: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const LLAMA3_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN2_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN35_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const TEKKEN_REGEX: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
/// llama.cpp's `default` type: these expressions split the text one after the other.
const DEFAULT_REGEXES: [&str; 4] = [
    r"[\p{P}\$\+<=>\^~\|]+",
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)",
    r"\p{N}+",
    r"[0-9][0-9][0-9]",
];

impl PreTokenizerKind {
    /// Maps `tokenizer.ggml.pre` to a kind. `None` (key absent) is llama.cpp's `default`; an unknown name is an error,
    /// as in llama.cpp.
    pub fn from_gguf(pre: Option<&str>) -> Result<Self, String> {
        Ok(match pre {
            None | Some("default") => Self::Default,
            Some("llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "midm-2.0" | "jina-v5-nano" | "lfm2") => {
                Self::Llama3
            }
            Some("qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2") => Self::Qwen2,
            Some("qwen35") => Self::Qwen35,
            Some("tekken" | "pixtral") => Self::Tekken,
            Some("falcon_e") => Self::FalconE,
            Some("gpt-2" | "phi-2" | "gpt2") => Self::Gpt2,
            Some(other) => return Err(format!("unknown pre-tokenizer type '{other}' (tokenizer.ggml.pre)")),
        })
    }

    /// Whether a piece that is in the vocabulary as a whole skips the merges (llama.cpp `ignore_merges`).
    pub fn ignore_merges(self) -> bool {
        matches!(self, Self::Llama3 | Self::Tekken)
    }

    fn build(self) -> Result<tokenizers::PreTokenizerWrapper, String> {
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::pre_tokenizers::punctuation::Punctuation;
        use tokenizers::pre_tokenizers::sequence::Sequence;
        use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
        use tokenizers::{PreTokenizerWrapper, SplitDelimiterBehavior};
        let split = |re: &str| -> Result<PreTokenizerWrapper, String> {
            Ok(Split::new(SplitPattern::Regex(re.to_string()), SplitDelimiterBehavior::Isolated, false)
                .map_err(|e| e.to_string())?
                .into())
        };
        let bytes: PreTokenizerWrapper = ByteLevel::default().add_prefix_space(false).use_regex(false).into();
        let seq = |mut parts: Vec<PreTokenizerWrapper>| -> PreTokenizerWrapper {
            parts.push(ByteLevel::default().add_prefix_space(false).use_regex(false).into());
            Sequence::new(parts).into()
        };
        Ok(match self {
            Self::Gpt2 => seq(vec![split(GPT2_REGEX)?]),
            Self::Default => seq(DEFAULT_REGEXES.iter().map(|re| split(re)).collect::<Result<_, _>>()?),
            Self::Llama3 => seq(vec![split(LLAMA3_REGEX)?]),
            Self::Qwen2 => seq(vec![split(QWEN2_REGEX)?]),
            Self::Qwen35 => seq(vec![split(QWEN35_REGEX)?]),
            Self::Tekken => seq(vec![split(TEKKEN_REGEX)?]),
            Self::FalconE => Sequence::new(vec![
                Punctuation::new(SplitDelimiterBehavior::Contiguous).into(),
                split(TEKKEN_REGEX)?,
                bytes,
                split("[0-9]")?,
            ])
            .into(),
        })
    }
}

/// Turns a stream of token ids into text without breaking multi-byte UTF-8 characters.
///
/// A byte-level BPE token can end in the middle of a character (`ü` is the two tokens `0xC3` and `0xBC` in the Falcon3
/// vocabulary); decoding such a token alone gives a replacement character. This follows the usual streaming scheme (as
/// in Hugging Face `TextStreamer`): the pending tokens are decoded together with a few already emitted ones, and text
/// that ends in an incomplete character is held back until a later token completes it.
#[derive(Debug, Clone, Default)]
pub struct StreamDecoder {
    ids: Vec<u32>,
    /// Start of the window that is re-decoded for context.
    prefix: usize,
    /// End of the already emitted part of the window.
    read: usize,
}

impl StreamDecoder {
    /// Creates an empty decoder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a token and returns the newly completed text (empty while a character is still incomplete).
    pub fn push(&mut self, tokenizer: &QuintritTokenizer, id: u32) -> String {
        self.ids.push(id);
        let prefix_text = tokenizer.decode(&self.ids[self.prefix..self.read]);
        let new_text = tokenizer.decode(&self.ids[self.prefix..]);
        if new_text.len() > prefix_text.len() && !new_text.ends_with('\u{FFFD}') {
            let piece = new_text.get(prefix_text.len()..).unwrap_or("").to_string();
            self.prefix = self.read;
            self.read = self.ids.len();
            piece
        } else {
            String::new()
        }
    }

    /// Returns the text held back at the end of the sequence (an incomplete character is decoded lossily).
    pub fn finish(&mut self, tokenizer: &QuintritTokenizer) -> String {
        let prefix_text = tokenizer.decode(&self.ids[self.prefix..self.read]);
        let new_text = tokenizer.decode(&self.ids[self.prefix..]);
        self.prefix = self.ids.len();
        self.read = self.ids.len();
        new_text.get(prefix_text.len()..).unwrap_or("").to_string()
    }

    /// All token ids pushed so far.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
}
