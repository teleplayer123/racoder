use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::llm::LlmClient;
use crate::schema::ToolCall;
use crate::tools::ToolExecutor;

const COMPACT_THRESHOLD: usize = 20;
const COMPACT_TAKE: usize = 10;

#[derive(Debug, Clone)]
pub enum StepResult {
    Proposed(ToolCall),
    Final(String),
    Error(String),
}

#[derive(Serialize, Deserialize)]
struct SessionData {
    summary: Option<String>,
    history: Vec<String>,
}

#[derive(Clone)]
pub struct Agent {
    llm: LlmClient,
    pub history: Vec<String>,
    pub summary: Option<String>,
}

impl Agent {
    pub fn new(base_url: &str) -> Self {
        Self {
            llm: LlmClient::new(base_url),
            history: vec![],
            summary: None,
        }
    }

    /// Load a saved session from disk, or create a fresh agent if none exists.
    pub fn load_or_new(base_url: &str, path: &Path) -> (Self, bool) {
        let mut agent = Self::new(base_url);
        let loaded = std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str::<SessionData>(&s).ok())
            .map(|data| {
                agent.summary = data.summary;
                agent.history = data.history;
            })
            .is_some();
        (agent, loaded)
    }

    /// Persist history and summary to disk.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = SessionData {
            summary: self.summary.clone(),
            history: self.history.clone(),
        };
        std::fs::write(path, serde_json::to_string_pretty(&data)?)?;
        Ok(())
    }

    /// Returns the default session file path: ~/.racoder/session.json
    pub fn default_session_path() -> std::path::PathBuf {
        std::env::var("HOME")
            .map(|h| std::path::PathBuf::from(h).join(".racoder").join("session.json"))
            .unwrap_or_else(|_| std::path::PathBuf::from("racoder_session.json"))
    }

    fn build_prompt(&self, goal: &str) -> String {
        let mut prompt = String::new();

        prompt.push_str(r#"
You are a secure coding agent.

Rules:
- Respond ONLY with valid JSON — no prose, no markdown, no code fences
- No explanations
- Always choose exactly one action
- Use tools step by step

Available actions and their EXACT required JSON format:

{"action":"read_file","path":"<file path>"}
{"action":"write_file","path":"<destination file path>","content":"<full file content>"}
{"action":"run_command","command":"<shell command>"}
{"action":"list_dir","path":"<directory path>"}
{"action":"final","message":"<completion message>"}

IMPORTANT: write_file MUST include both "path" AND "content" fields. Never omit "path".
"#);

        prompt.push_str(&format!("\nGoal:\n{}\n\n", goal));

        if let Some(summary) = &self.summary {
            prompt.push_str(&format!("Context from prior work:\n{}\n\n", summary));
        }

        if !self.history.is_empty() {
            prompt.push_str("Recent history:\n");
            for h in &self.history {
                prompt.push_str(h);
                prompt.push('\n');
            }
        }

        prompt
    }

    fn extract_json(response: &str) -> Option<String> {
        let start = response.find('{')?;
        let end = response.rfind('}')?;
        Some(response[start..=end].to_string())
    }

    /// If history is long, drain the oldest entries and compress them into the summary.
    async fn maybe_compact(&mut self) {
        if self.history.len() <= COMPACT_THRESHOLD {
            return;
        }
        let entries: Vec<String> = self.history.drain(..COMPACT_TAKE).collect();
        let compact_prompt = match &self.summary {
            Some(existing) => format!(
                "You have this running summary of past agent work:\n{}\n\n\
                 Incorporate these additional history entries and produce a single updated \
                 summary paragraph. Preserve all key facts: file paths, code written, \
                 commands run, errors, decisions. Be concise.\n\nNew entries:\n{}",
                existing,
                entries.join("\n")
            ),
            None => format!(
                "Summarize these agent history entries into one dense paragraph capturing \
                 all key facts: file paths, code written, commands run, errors, decisions. \
                 Be concise.\n\nEntries:\n{}",
                entries.join("\n")
            ),
        };
        if let Ok(new_summary) = self.llm.complete(&compact_prompt).await {
            self.summary = Some(new_summary);
        }
        // On failure we simply keep the drained entries gone and move on —
        // the remaining history still provides context.
    }

    /// Plan the next step (DO NOT execute).
    pub async fn plan_step(
        &mut self,
        goal: &str,
        stream_tx: Option<std::sync::mpsc::Sender<String>>,
    ) -> Result<StepResult> {
        self.maybe_compact().await;

        let prompt = self.build_prompt(goal);
        let raw = match stream_tx {
            Some(tx) => self.llm.complete_streaming(&prompt, tx).await?,
            None => self.llm.complete(&prompt).await?,
        };

        self.history.push(format!("LLM: {}", raw));

        let json = match Self::extract_json(&raw) {
            Some(j) => j,
            None => return Ok(StepResult::Error("No JSON found".into())),
        };

        let parsed: ToolCall = match serde_json::from_str(&json) {
            Ok(v) => v,
            Err(e) => return Ok(StepResult::Error(format!("Parse error: {}", e))),
        };

        match parsed {
            ToolCall::Final { message } => {
                self.history.push(format!("Final: {}", message));
                Ok(StepResult::Final(message))
            }
            action => {
                self.history.push(format!("Proposed: {:?}", action));
                Ok(StepResult::Proposed(action))
            }
        }
    }

    /// Execute an approved action.
    pub fn execute_step(&mut self, action: ToolCall) -> Result<String> {
        let result = ToolExecutor::execute(action)?;
        self.history.push(format!("Tool result: {}", result));
        Ok(result)
    }
}
