//! HuggingFace `tokenizers` crate integration for sub-millisecond BPE encoding and decoding.

use ahash::AHashMap;
use std::path::Path;
use tokenizers::models::bpe::BPE;
use tokenizers::{AddedToken, Tokenizer};

/// GGUF token type of a control token (`tokenizer.ggml.token_type`, llama.cpp `LLAMA_TOKEN_TYPE_CONTROL`), such as
/// `<|endoftext|>`.
pub const GGUF_TOKEN_TYPE_CONTROL: i32 = 3;

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
        let mut vocab: AHashMap<String, u32> = AHashMap::with_capacity(tokens.len());
        for (i, token) in tokens.iter().enumerate() {
            let key = token.replace('\n', "Ċ").replace('\r', "č").replace('\t', "ċ");
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

        let bpe_builder = BPE::builder().vocab_and_merges(vocab, parsed_merges);
        let bpe_model = bpe_builder.build().map_err(|e| e.to_string())?;
        let mut tokenizer = Tokenizer::new(bpe_model);

        // Configure ByteLevel pre-tokenizer and decoder for GPT-2/Falcon/LLaMA BPE compatibility
        let byte_level = tokenizers::pre_tokenizers::byte_level::ByteLevel::default().add_prefix_space(false);
        tokenizer.with_pre_tokenizer(Some(byte_level));
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
