use anyhow::Result;

use crate::llm::LlmClient;
use crate::schema::ToolCall;
use crate::tools::ToolExecutor;

#[derive(Debug, Clone)]
pub enum StepResult {
    Proposed(ToolCall),
    Final(String),
    Error(String),
}

#[derive(Clone)]
pub struct Agent {
    llm: LlmClient,
    pub history: Vec<String>,
}

impl Agent {
    pub fn new(base_url: &str) -> Self {
        Self {
            llm: LlmClient::new(base_url),
            history: vec![],
        }
    }

    fn build_prompt(&self, goal: &str) -> String {
        let mut prompt = String::new();

        // System instructions
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

        // Goal
        prompt.push_str(&format!("\nGoal:\n{}\n\n", goal));

        // History
        if !self.history.is_empty() {
            prompt.push_str("History:\n");
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

    /// Plan the next step (DO NOT execute).
    pub async fn plan_step(&mut self, goal: &str) -> Result<StepResult> {
        let prompt = self.build_prompt(goal);
        let raw = self.llm.complete(&prompt).await?;

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

    /// Execute an approved action
    pub fn execute_step(&mut self, action: ToolCall) -> Result<String> {
        let result = ToolExecutor::execute(action)?;
        self.history.push(format!("Tool result: {}", result));
        Ok(result)
    }
}