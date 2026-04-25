use anyhow::Result;
use std::fs;
use std::process::Command;
use similar::{ChangeTag, TextDiff};

pub struct ToolExecutor;

impl ToolExecutor {
    pub fn execute(call: crate::schema::ToolCall) -> Result<String> {
        match call {
            crate::schema::ToolCall::ReadFile { path } => {
                Ok(fs::read_to_string(path)?)
            }
            crate::schema::ToolCall::WriteFile { path, content } => {
                fs::write(path, content)?;
                Ok("File written".into())
            }
            crate::schema::ToolCall::RunCommand { command } => {
                let allowed = ["cargo check", "cargo run", "ls", "grep", "cat", "echo"];
                if !allowed.iter().any(|a| command.starts_with(a)) {
                    return Ok("Command not allowed".into());
                }

                let output = Command::new("sh")
                    .arg("-c")
                    .arg(command)
                    .output()?;

                Ok(String::from_utf8_lossy(&output.stdout).to_string())
            }
            crate::schema::ToolCall::ListDir { path } => {
                let entries = fs::read_dir(path)?;
                let mut names = vec![];
                for e in entries {
                    names.push(e?.file_name().to_string_lossy().to_string());
                }
                Ok(names.join("\n"))
            }
            crate::schema::ToolCall::Final { message } => Ok(message),
        }
    }
}

pub fn generate_diff(old: &str, new: &str) -> String {
    let diff = TextDiff::from_lines(old, new);

    let mut output = String::new();

    for change in diff.iter_all_changes() {
        let sign = match change.tag() {
            ChangeTag::Delete => "-",
            ChangeTag::Insert => "+",
            ChangeTag::Equal => " ",
        };

        output.push_str(&format!("{}{}", sign, change));
    }

    output
}