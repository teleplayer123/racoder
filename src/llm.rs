use anyhow::Result;
use reqwest::Client;
use serde_json::json;
use std::sync::mpsc;
use std::time::Instant;
use crate::{log_debug, log_error, log_warn};

#[derive(Clone)]
pub struct LlmClient {
    // `reqwest::Client` is connection-pooling and cheaply clonable; reusing one instance
    // across requests avoids re-establishing TCP connections to the local LLM server.
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

    /// Sends a blocking (non-streaming) chat request and returns the full response text.
    /// Used for compaction calls where we don't need live token updates in the UI.
    pub async fn complete(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url);
        log_debug!("complete — POST {} prompt_len={}", url, prompt.len());

        // `json!` (serde_json macro) builds a `serde_json::Value` literal inline and
        // `.json()` serializes it as the request body with Content-Type: application/json.
        let mut resp = self
            .client
            .post(&url)
            .json(&json!({
                "model": "Qwen3", // This name will not matter if there is only one LLM on the server.
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
            // Log the full body — HTTP errors from local LLMs often carry useful details
            // (e.g. "model not found", "context length exceeded") that are easy to miss.
            log_error!("complete — HTTP {} from {}: {}", status, url, body.trim());
            return Err(anyhow::anyhow!(
                "LLM server returned HTTP {}: {}",
                status,
                body.trim()
            ));
        }

        let mut body: Vec<u8> = Vec::new();
        let first_chunk_start = Instant::now();

        // `resp.chunk()` yields one network chunk at a time rather than buffering the full
        // body first, which avoids a memory spike on large responses and lets us apply
        // per-chunk timeouts. The first chunk gets a long timeout because local LLMs may
        // spend minutes generating before sending anything.
        match tokio::time::timeout(
            std::time::Duration::from_secs(300),
            resp.chunk(),
        )
        .await
        {
            Ok(Ok(Some(chunk))) => {
                let elapsed_ms = first_chunk_start.elapsed().as_millis();
                log_debug!("complete — first chunk arrived in {}ms, {} bytes", elapsed_ms, chunk.len());
                body.extend_from_slice(&chunk);
            }
            Ok(Ok(None)) => {
                log_warn!("complete — server closed connection before sending any data");
            }
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => {
                log_error!("complete — first-chunk timeout after 300s; is the LLM server running?");
                return Err(anyhow::anyhow!("LLM did not respond within 5 minutes"));
            }
        }

        // After the first chunk arrives we switch to a short idle timeout. Many local servers
        // keep the TCP connection open after the last byte; if no new data arrives within 10s
        // we assume the response is complete and parse what we have.
        loop {
            match tokio::time::timeout(
                std::time::Duration::from_secs(10),
                resp.chunk(),
            )
            .await
            {
                Ok(Ok(Some(chunk))) => body.extend_from_slice(&chunk),
                Ok(Ok(None)) => {
                    log_debug!("complete — clean EOF, total body={} bytes", body.len());
                    break;
                }
                Ok(Err(e)) => return Err(e.into()),
                Err(_) => {
                    // Idle timeout is normal for many local servers that hold the connection
                    // open. Treat it as end-of-response and parse what arrived.
                    log_debug!("complete — idle timeout; treating {} bytes as complete response", body.len());
                    break;
                }
            }
        }

        // Parse the accumulated bytes as JSON and extract the assistant message content
        // using the OpenAI-compatible response shape: choices[0].message.content
        let value: serde_json::Value = serde_json::from_slice(&body)?;
        let content = value["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| {
                // This fires when the LLM server returns a different API shape
                // (e.g. an error object, or a streaming-only response accidentally used here).
                log_error!(
                    "complete — response missing choices[0].message.content; body={}",
                    String::from_utf8_lossy(&body).chars().take(300).collect::<String>()
                );
                anyhow::anyhow!("Response missing content field")
            })?
            .to_string();

        log_debug!("complete — content_len={}", content.len());
        Ok(content)
    }

    /// Streaming variant: sends each token to `stream_tx` as it arrives and
    /// returns the full accumulated response when done.
    /// Used for plan_step calls so the TUI can show live token output in the Action panel.
    pub async fn complete_streaming(
        &self,
        prompt: &str,
        stream_tx: mpsc::Sender<String>,
    ) -> Result<String> {
        // `StreamExt` is a trait from the `futures` crate that adds `.next()` to any
        // `Stream`, giving us an async iterator over the byte chunks of the response body.
        use futures::StreamExt;

        let url = format!("{}/chat/completions", self.base_url);
        log_debug!("complete_streaming — POST {} prompt_len={}", url, prompt.len());

        let res = self
            .client
            .post(&url)
            .json(&json!({
                "model": "qwen",
                "stream": true,   // tells the server to use SSE (Server-Sent Events) format
                "messages": [
                    {"role": "system", "content": "Return ONLY JSON."},
                    {"role": "user", "content": prompt}
                ]
            }))
            .send()
            .await?;

        let status = res.status();
        if !status.is_success() {
            let body = res.text().await.unwrap_or_default();
            log_error!("complete_streaming — HTTP {} from {}: {}", status, url, body.trim());
            return Err(anyhow::anyhow!("LLM server returned HTTP {}: {}", status, body.trim()));
        }

        // `bytes_stream()` converts the response body into an async `Stream<Item=Bytes>`.
        // Each item is a raw network chunk; it may contain partial or multiple SSE lines.
        let mut byte_stream = res.bytes_stream();
        let mut full_text = String::new();
        // Raw byte buffer to accumulate chunks until we have a complete newline-terminated
        // SSE line — a single chunk may not align with line boundaries.
        let mut raw_buf: Vec<u8> = Vec::new();
        let mut token_count: usize = 0;
        let mut sse_parse_errors: usize = 0;

        'stream: while let Some(chunk) = byte_stream.next().await {
            raw_buf.extend_from_slice(&chunk?);

            // Consume every complete line from the front of the buffer before fetching
            // the next network chunk, so we don't fall behind under fast token output.
            while let Some(pos) = raw_buf.iter().position(|&b| b == b'\n') {
                // `drain(..=pos)` removes and returns bytes from the start up to and
                // including the newline, shifting the remaining bytes to the front.
                let line_bytes: Vec<u8> = raw_buf.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line_bytes);
                let line = line.trim();

                // SSE protocol: every event line is prefixed with "data: ".
                // Lines that don't match (empty lines, comment lines) are ignored.
                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        // The server sends "[DONE]" as the final SSE event. Break out of
                        // both loops immediately rather than waiting for the connection to
                        // close, which can hang indefinitely on some local servers.
                        log_debug!(
                            "complete_streaming — [DONE] received, tokens={}, total_len={}",
                            token_count, full_text.len()
                        );
                        break 'stream;
                    }
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                        // OpenAI streaming response shape: each chunk carries a `delta`
                        // instead of a `message`; `content` is the new token text.
                        if let Some(token) = val["choices"][0]["delta"]["content"].as_str() {
                            full_text.push_str(token);
                            token_count += 1;
                            // `send` fails only if the receiver was dropped (e.g. task
                            // was cancelled); silently ignore so the stream keeps running.
                            let _ = stream_tx.send(token.to_string());
                        }
                    } else {
                        // A non-JSON "data:" line that isn't "[DONE]" is unexpected.
                        // Track the count so we can warn once at the end rather than per-token.
                        sse_parse_errors += 1;
                    }
                }
            }
        }

        if sse_parse_errors > 0 {
            log_warn!(
                "complete_streaming — {} SSE lines failed JSON parse (malformed server output)",
                sse_parse_errors
            );
        }

        if full_text.is_empty() {
            log_warn!("complete_streaming — stream ended with no content extracted");
        } else {
            log_debug!("complete_streaming — done, tokens={}, content_len={}", token_count, full_text.len());
        }

        Ok(full_text)
    }
}
