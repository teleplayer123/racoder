use anyhow::Result;
use reqwest::Client;
use serde_json::json;
use std::clone::Clone;

#[derive(Clone)]
pub struct LlmClient {
    client: Client,
    base_url: String,
}

impl LlmClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    pub async fn complete(&self, prompt: &str) -> Result<String> {
        let res = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .json(&json!({
                "model": "qwen",
                "messages": [
                    {"role": "system", "content": "Return ONLY JSON."},
                    {"role": "user", "content": prompt}
                ]
            }))
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let content = res["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Response is not a string"))?
            .to_string();

        Ok(content)
    }
}
