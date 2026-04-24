use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ToolCall {
    ReadFile { path: String },
    WriteFile { path: String, content: String },
    RunCommand { command: String },
    ListDir { path: String },
    Final { message: String },
}