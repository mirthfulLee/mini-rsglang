use rsglang_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}
pub struct TextProcessor {
    pub tokenizer: Tokenizer,
    template: String,
}
impl TextProcessor {
    pub fn load(path: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))
            .map_err(|e| Error::Invalid(e.to_string()))?;
        let cfg: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("tokenizer_config.json"))?)
                .map_err(|e| Error::Invalid(e.to_string()))?;
        let template = cfg
            .get("chat_template")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Invalid("missing chat template".into()))?
            .to_owned();
        Ok(Self {
            tokenizer,
            template,
        })
    }
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::Invalid(e.to_string()))?
            .get_ids()
            .to_vec())
    }
    pub fn chat(&self, messages: &[ChatMessage], enable_thinking: bool) -> Result<String> {
        if messages.is_empty()
            || messages
                .iter()
                .any(|m| !["system", "user", "assistant"].contains(&m.role.as_str()))
        {
            return Err(Error::Invalid(
                "chat requires nonempty text messages with system/user/assistant roles".into(),
            ));
        }
        let mut env = minijinja::Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template("chat", &self.template)
            .map_err(|e| Error::Invalid(e.to_string()))?;
        env.get_template("chat").map_err(|e|Error::Invalid(e.to_string()))?.render(minijinja::context!{messages=>messages,tools=>false,add_generation_prompt=>true,enable_thinking=>enable_thinking}).map_err(|e|Error::Invalid(e.to_string()))
    }
}
#[derive(Default)]
pub(crate) struct IncrementalDecoder {
    ids: Vec<u32>,
    prefix: String,
    index: usize,
    pub emitted: String,
}
impl IncrementalDecoder {
    pub fn step(&mut self, tok: &Tokenizer, id: u32) -> Result<String> {
        let text = tokenizers::tokenizer::step_decode_stream(
            tok,
            vec![id],
            true,
            &mut self.ids,
            &mut self.prefix,
            &mut self.index,
        )
        .map_err(|e| Error::Backend(e.to_string()))?
        .unwrap_or_default();
        self.emitted.push_str(&text);
        Ok(text)
    }
    pub fn flush(&mut self, tok: &Tokenizer, all_ids: &[u32]) -> Result<String> {
        let full = tok
            .decode(all_ids, true)
            .map_err(|e| Error::Backend(e.to_string()))?;
        let tail = full
            .strip_prefix(&self.emitted)
            .ok_or_else(|| Error::Backend("decoded prefix changed".into()))?
            .to_owned();
        self.emitted.push_str(&tail);
        Ok(tail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tokenizer() -> Tokenizer {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [
                    ("<0xC3>".to_owned(), 0),
                    ("<0xA9>".to_owned(), 1),
                    ("[UNK]".to_owned(), 2),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_decoder(Some(
            tokenizers::decoders::byte_fallback::ByteFallback::default(),
        ));
        tok
    }
    #[test]
    fn holds_partial_utf8_until_complete() {
        let tok = tokenizer();
        let mut d = IncrementalDecoder::default();
        assert_eq!(d.step(&tok, 0).unwrap(), "");
        assert_eq!(d.step(&tok, 1).unwrap(), "é");
        assert_eq!(d.flush(&tok, &[0, 1]).unwrap(), "");
    }
    #[test]
    fn flushes_incomplete_final_character() {
        let tok = tokenizer();
        let mut d = IncrementalDecoder::default();
        assert_eq!(d.step(&tok, 0).unwrap(), "");
        assert_eq!(d.flush(&tok, &[0]).unwrap(), "�");
    }
}
