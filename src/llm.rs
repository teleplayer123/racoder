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
        let mut resp = self
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
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "LLM server returned HTTP {}: {}",
                status,
                body.trim()
            ));
        }

        let mut body: Vec<u8> = Vec::new();

        // First chunk: allow the LLM as much time as it needs to start responding.
        match tokio::time::timeout(
            std::time::Duration::from_secs(300),
            resp.chunk(),
        )
        .await
        {
            Ok(Ok(Some(chunk))) => body.extend_from_slice(&chunk),
            Ok(Ok(None))        => {} // empty body — will fail at parse below
            Ok(Err(e))          => return Err(e.into()),
            Err(_)              => return Err(anyhow::anyhow!("LLM did not respond within 5 minutes")),
        }

        // Remaining chunks: 10-second idle timeout. Many local servers keep the
        // connection alive after sending the full body; if no new bytes arrive
        // within 10 s of the last chunk we have the complete response.
        loop {
            match tokio::time::timeout(
                std::time::Duration::from_secs(10),
                resp.chunk(),
            )
            .await
            {
                Ok(Ok(Some(chunk))) => body.extend_from_slice(&chunk),
                Ok(Ok(None))        => break, // clean EOF
                Ok(Err(e))          => return Err(e.into()),
                Err(_)              => break, // idle timeout — parse what we have
            }
        }

        let value: serde_json::Value = serde_json::from_slice(&body)?;
        let content = value["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Response missing content field"))?
            .to_string();

        Ok(content)
    }

    /// Streaming variant: sends each token to `stream_tx` as it arrives and
    /// returns the full accumulated response when done.
    #[allow(dead_code)]
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

        'stream: while let Some(chunk) = byte_stream.next().await {
            raw_buf.extend_from_slice(&chunk?);

            // Process every complete newline-terminated line from the buffer
            while let Some(pos) = raw_buf.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = raw_buf.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line_bytes);
                let line = line.trim();

                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        // Break out of both loops — don't wait for the server
                        // to close the connection, which can hang indefinitely.
                        break 'stream;
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
