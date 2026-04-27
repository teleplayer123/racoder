use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::llm::LlmClient;
use crate::schema::ToolCall;
use crate::tools::ToolExecutor;

// Compact history when it exceeds this many entries to keep prompt sizes manageable.
const COMPACT_THRESHOLD: usize = 20;
// Number of oldest entries to drain and compress each time compaction runs.
const COMPACT_TAKE: usize = 10;

#[derive(Debug, Clone)]
pub enum StepResult {
    Proposed(ToolCall),
    Final(String),
    Error(String),
}

// Serialization container for the session file. Keeping it separate from Agent
// lets us version the on-disk format independently of the runtime struct.
#[derive(Serialize, Deserialize)]
struct SessionData {
    summary: Option<String>,
    history: Vec<String>,
}

#[derive(Clone)]
pub struct Agent {
    llm: LlmClient,
    // Every action and result in the current session, in chronological order.
    // Injected verbatim into the LLM prompt so it has full context of what has happened.
    pub history: Vec<String>,
    // LLM-generated compression of history entries that have been evicted from `history`.
    // Prepended to the prompt so the LLM retains awareness of prior work without
    // the full token cost of the raw entries.
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
    /// Returns the agent and a bool indicating whether a session was found.
    pub fn load_or_new(base_url: &str, path: &Path) -> (Self, bool) {
        let mut agent = Self::new(base_url);
        // `.ok()` converts the Result to an Option, treating any read or parse error as
        // "no session found" rather than propagating an error on first run.
        let loaded = std::fs::read_to_string(path)
            .ok()
            // `serde_json::from_str::<SessionData>` — the turbofish tells serde which
            // concrete type to deserialize into, since it can't be inferred from context.
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
        // `create_dir_all` creates the target directory and any missing parents.
        // Equivalent to `mkdir -p`; safe to call if the directory already exists.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = SessionData {
            summary: self.summary.clone(),
            history: self.history.clone(),
        };
        // `to_string_pretty` adds indentation so the file is human-readable for debugging.
        std::fs::write(path, serde_json::to_string_pretty(&data)?)?;
        Ok(())
    }

    /// Returns the default session file path: ~/.racoder/session.json
    pub fn default_session_path() -> std::path::PathBuf {
        std::env::var("HOME")
            .map(|h| std::path::PathBuf::from(h).join(".racoder").join("session.json"))
            .unwrap_or_else(|_| std::path::PathBuf::from("racoder_session.json"))
    }

    /// Builds the full prompt string that is sent to the LLM.
    /// Order matters: system rules → goal → summary (compressed past) → recent raw history.
    /// Putting the goal before the summary keeps it prominent in the context window.
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

    /// Strips the outermost JSON object from the LLM response, discarding any prose the
    /// model may have added before or after the JSON despite instructions not to.
    fn extract_json(response: &str) -> Option<String> {
        let start = response.find('{')?;
        let end = response.rfind('}')?;
        Some(response[start..=end].to_string())
    }

    /// If history has grown past COMPACT_THRESHOLD, drain the oldest COMPACT_TAKE entries
    /// and ask the LLM to compress them into (or update) the running summary.
    /// Uses a non-streaming call so compaction tokens don't appear in the Action panel preview.
    async fn maybe_compact(&mut self) {
        if self.history.len() <= COMPACT_THRESHOLD {
            return;
        }
        // `drain(..COMPACT_TAKE)` removes and returns the first N elements in place,
        // shifting the remaining elements to the front of the Vec.
        let entries: Vec<String> = self.history.drain(..COMPACT_TAKE).collect();
        let compact_prompt = match &self.summary {
            // If a summary already exists, ask the LLM to incorporate the new entries
            // into it rather than starting fresh, so the summary stays bounded in size.
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
        // On LLM failure the drained entries are simply lost — the remaining history and
        // any existing summary still provide context, so the agent can continue.
    }

    /// Ask the LLM what to do next toward the goal. Does NOT execute the action.
    /// Compaction runs first if history is long, then the prompt is built and sent.
    /// If a stream_tx is provided the response tokens are forwarded live to the TUI.
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

        // Record the raw LLM response so future steps can see what was proposed.
        self.history.push(format!("LLM: {}", raw));

        let json = match Self::extract_json(&raw) {
            Some(j) => j,
            None => return Ok(StepResult::Error("No JSON found".into())),
        };

        // `serde_json::from_str` uses the `#[serde(tag = "action")]` annotation on ToolCall
        // to pick the correct enum variant based on the value of the "action" key.
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

    /// Execute a user-approved action and record the result in history.
    pub fn execute_step(&mut self, action: ToolCall) -> Result<String> {
        let result = ToolExecutor::execute(action)?;
        self.history.push(format!("Tool result: {}", result));
        Ok(result)
    }
}
