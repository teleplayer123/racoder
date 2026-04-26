use anyhow::Result;
use std::sync::mpsc;

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
- Respond ONLY with valid JSON
- No explanations
- Always choose an action
- Use tools step by step

Available actions:
- read_file
- write_file
- run_command
- list_dir
- final
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
    /// If `stream_tx` is provided, tokens are forwarded to it as they arrive.
    pub async fn plan_step(
        &mut self,
        goal: &str,
        stream_tx: Option<mpsc::Sender<String>>,
    ) -> Result<StepResult> {
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

    /// Execute an approved action
    pub fn execute_step(&mut self, action: ToolCall) -> Result<String> {
        let result = ToolExecutor::execute(action)?;
        self.history.push(format!("Tool result: {}", result));
        Ok(result)
    }
}