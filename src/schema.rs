use serde::{Deserialize, Serialize};

// `tag = "action"` tells serde to look for an "action" key to decide which variant to
// deserialize into (e.g. {"action":"read_file",...} → ToolCall::ReadFile).
// `rename_all = "snake_case"` maps Rust PascalCase variant names to snake_case JSON values
// automatically, so ReadFile → "read_file" without having to write a rename per variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ToolCall {
    ReadFile { path: String },
    // `#[serde(default)]` makes `path` optional in the JSON — if the LLM omits it, serde
    // fills in an empty String instead of returning a parse error. The TUI then prompts
    // the user to supply the path interactively.
    WriteFile { #[serde(default)] path: String, content: String },
    RunCommand { command: String },
    ListDir { path: String },
    Final { message: String },
}
