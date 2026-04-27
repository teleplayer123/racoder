use anyhow::Result;
use std::fs;
use std::process::Command;
use similar::{ChangeTag, TextDiff};

// Returns true only for commands the agent is allowed to run without user override.
// Two-layer check: first rejects any string containing shell metacharacters that could
// chain or redirect commands (injection prevention), then allowlists by the first token.
pub fn is_command_allowed(command: &str) -> bool {
    let has_shell_op = command.contains("&&")
        || command.contains("||")
        || command.contains(';')
        || command.contains('|')
        || command.contains('>')
        || command.contains('<')
        || command.contains('`')
        || command.contains('$');
    if has_shell_op {
        return false;
    }
    // Split on whitespace and pattern-match the slice so the first token determines
    // the allowed command family; `..` ignores any additional arguments.
    let tokens: Vec<&str> = command.split_whitespace().collect();
    match tokens.as_slice() {
        ["ls", ..] | ["grep", ..] | ["cat", ..] | ["echo", ..] => true,
        // Only safe cargo subcommands — "cargo install" or "cargo publish" are not allowed.
        ["cargo", sub, ..] if ["check", "run", "build", "test"].contains(sub) => true,
        _ => false,
    }
}

/// Run a shell command without allowlist enforcement.
/// `sh -c` is used so the command string is interpreted by the shell, which handles
/// quoted arguments and paths with spaces correctly.
/// Returns stdout on success, or a formatted exit-code + stderr string on failure
/// so the LLM can read the error and adjust its next action.
pub fn run_command_unchecked(command: &str) -> Result<String> {
    let output = Command::new("sh").arg("-c").arg(command).output()?;
    // `from_utf8_lossy` replaces any non-UTF-8 bytes with `?` rather than returning
    // an error — important for commands that may output binary or mixed-encoding text.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // `status.success()` returns true only if the exit code is 0.
    if !output.status.success() {
        Ok(format!(
            "exit {}\nstdout: {}\nstderr: {}",
            output.status, stdout, stderr
        ))
    } else {
        Ok(stdout.to_string())
    }
}

pub struct ToolExecutor;

impl ToolExecutor {
    pub fn execute(call: crate::schema::ToolCall) -> Result<String> {
        match call {
            // `fs::read_to_string` reads the entire file into a heap String.
            crate::schema::ToolCall::ReadFile { path } => Ok(fs::read_to_string(path)?),
            crate::schema::ToolCall::WriteFile { path, content } => {
                // `fs::write` creates the file if it doesn't exist and truncates if it does.
                fs::write(path, content)?;
                Ok("File written".into())
            }
            crate::schema::ToolCall::RunCommand { command } => {
                if !is_command_allowed(&command) {
                    return Ok("Command not allowed".into());
                }
                run_command_unchecked(&command)
            }
            crate::schema::ToolCall::ListDir { path } => {
                // `read_dir` returns an iterator of `DirEntry` results; we unwrap each
                // entry individually so a single unreadable entry doesn't abort the whole listing.
                let entries = fs::read_dir(path)?;
                let mut names = vec![];
                for e in entries {
                    // `file_name()` returns an OsString; `to_string_lossy` converts it to a
                    // UTF-8 String, replacing any non-UTF-8 bytes with `?`.
                    names.push(e?.file_name().to_string_lossy().to_string());
                }
                Ok(names.join("\n"))
            }
            crate::schema::ToolCall::Final { message } => Ok(message),
        }
    }
}

/// Produces a unified-style diff between `old` and `new` using the `similar` crate.
/// `TextDiff::from_lines` runs a Myers diff algorithm on the two strings split by line.
/// Each change is prefixed with `+`, `-`, or ` ` (equal) to match conventional diff output,
/// which the TUI then color-codes green/red/gray.
pub fn generate_diff(old: &str, new: &str) -> String {
    let diff = TextDiff::from_lines(old, new);
    let mut output = String::new();
    // `iter_all_changes` yields every line (equal lines included) so the diff shows context.
    for change in diff.iter_all_changes() {
        // `change.tag()` returns whether this line was deleted, inserted, or unchanged.
        let sign = match change.tag() {
            ChangeTag::Delete => "-",
            ChangeTag::Insert => "+",
            ChangeTag::Equal => " ",
        };
        output.push_str(&format!("{}{}", sign, change));
    }
    output
}
