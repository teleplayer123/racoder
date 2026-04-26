use anyhow::Result;
use reqwest::Client;
use serde_json::json;
use std::sync::mpsc;

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

    /// Streaming variant: sends each token to `stream_tx` as it arrives and
    /// returns the full accumulated response when done.
    pub async fn complete_streaming(
        &self,
        prompt: &str,
        stream_tx: mpsc::Sender<String>,
    ) -> Result<String> {
        use futures::StreamExt;

        let res = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .json(&json!({
                "model": "qwen",
                "stream": true,
                "messages": [
                    {"role": "system", "content": "Return ONLY JSON."},
                    {"role": "user", "content": prompt}
                ]
            }))
            .send()
            .await?;

        let mut byte_stream = res.bytes_stream();
        let mut full_text = String::new();
        // Accumulate raw bytes until we have a complete SSE line
        let mut raw_buf: Vec<u8> = Vec::new();

        while let Some(chunk) = byte_stream.next().await {
            raw_buf.extend_from_slice(&chunk?);

            // Process every complete newline-terminated line from the buffer
            while let Some(pos) = raw_buf.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = raw_buf.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line_bytes);
                let line = line.trim();

                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        continue;
                    }
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(token) = val["choices"][0]["delta"]["content"].as_str() {
                            full_text.push_str(token);
                            let _ = stream_tx.send(token.to_string());
                        }
                    }
                }
            }
        }

        Ok(full_text)
    }
}
