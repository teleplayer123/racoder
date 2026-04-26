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
                // Reject shell operators to prevent command injection
                let has_shell_op = command.contains("&&")
                    || command.contains("||")
                    || command.contains(';')
                    || command.contains('|')
                    || command.contains('>')
                    || command.contains('<')
                    || command.contains('`')
                    || command.contains('$');
                if has_shell_op {
                    return Ok("Command not allowed: shell operators are not permitted".into());
                }

                let tokens: Vec<&str> = command.split_whitespace().collect();
                let allowed = match tokens.as_slice() {
                    ["ls", ..] | ["grep", ..] | ["cat", ..] | ["echo", ..] => true,
                    ["cargo", sub, ..] if ["check", "run", "build", "test"].contains(sub) => true,
                    _ => false,
                };
                if !allowed {
                    return Ok("Command not allowed".into());
                }

                let output = Command::new("sh")
                    .arg("-c")
                    .arg(&command)
                    .output()?;

                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                if !output.status.success() {
                    Ok(format!(
                        "exit {}\nstdout: {}\nstderr: {}",
                        output.status, stdout, stderr
                    ))
                } else {
                    Ok(stdout.to_string())
                }
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
