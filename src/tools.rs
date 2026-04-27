use anyhow::Result;
use std::fs;
use std::process::Command;
use similar::{ChangeTag, TextDiff};
use crate::{log_debug, log_info, log_warn};

/// Returns true if `path` is safe to write to — either a relative path with no `..`
/// components, or an absolute path that starts with the current working directory.
///
/// Relative paths without `..` are always within CWD once joined to it.
/// Absolute paths must be an exact prefix match against the CWD so we never
/// silently allow writes to system directories the LLM has no business touching.
pub fn is_path_within_cwd(path: &str) -> bool {
    let p = std::path::Path::new(path);
    if p.is_relative() {
        // Disallow any `..` component; a purely relative path like `src/foo.rs`
        // is safe, but `../../etc/passwd` is not.
        !p.components().any(|c| c == std::path::Component::ParentDir)
    } else {
        let cwd = match std::env::current_dir() {
            Ok(d) => d,
            Err(_) => return false,
        };
        p.starts_with(&cwd)
    }
}

/// If `path` is outside the CWD, strips it to just the filename so the LLM
/// can't silently write to system directories. Returns the path unchanged when
/// it is already within the CWD.
pub fn sanitize_write_path(path: &str) -> String {
    if is_path_within_cwd(path) {
        return path.to_string();
    }
    // Keep only the final filename component so the intent (what to name the file)
    // is preserved while the dangerous location is discarded.
    let filename = std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("output.txt")
        .to_string();
    log_warn!(
        "sanitize_write_path — {:?} is outside CWD, redirecting to {:?}",
        path, filename
    );
    filename
}

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
        log_warn!(
            "is_command_allowed — blocked (shell operator): {:?}",
            command
        );
        return false;
    }
    // Split on whitespace and pattern-match the slice so the first token determines
    // the allowed command family; `..` ignores any additional arguments.
    let tokens: Vec<&str> = command.split_whitespace().collect();
    let allowed = match tokens.as_slice() {
        ["ls", ..] | ["grep", ..] | ["cat", ..] | ["echo", ..] => true,
        // Only safe cargo subcommands — "cargo install" or "cargo publish" are not allowed.
        ["cargo", sub, ..] if ["check", "run", "build", "test"].contains(sub) => true,
        _ => false,
    };
    if !allowed {
        log_warn!("is_command_allowed — blocked (not on allowlist): {:?}", command);
    } else {
        log_debug!("is_command_allowed — allowed: {:?}", command);
    }
    allowed
}

/// Run a shell command without allowlist enforcement.
/// `sh -c` is used so the command string is interpreted by the shell, which handles
/// quoted arguments and paths with spaces correctly.
/// Returns stdout on success, or a formatted exit-code + stderr string on failure
/// so the LLM can read the error and adjust its next action.
pub fn run_command_unchecked(command: &str) -> Result<String> {
    log_info!("run_command_unchecked — executing: {:?}", command);
    let output = Command::new("sh").arg("-c").arg(command).output()?;
    // `from_utf8_lossy` replaces any non-UTF-8 bytes with `?` rather than returning
    // an error — important for commands that may output binary or mixed-encoding text.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // `status.success()` returns true only if the exit code is 0.
    if !output.status.success() {
        log_warn!(
            "run_command_unchecked — non-zero exit: {} stdout_len={} stderr_len={} stderr={:?}",
            output.status,
            stdout.len(),
            stderr.len(),
            stderr.chars().take(200).collect::<String>()
        );
        Ok(format!(
            "exit {}\nstdout: {}\nstderr: {}",
            output.status, stdout, stderr
        ))
    } else {
        log_debug!(
            "run_command_unchecked — success, stdout_len={} stderr_len={}",
            stdout.len(), stderr.len()
        );
        Ok(stdout.to_string())
    }
}

pub struct ToolExecutor;

impl ToolExecutor {
    pub fn execute(call: crate::schema::ToolCall) -> Result<String> {
        match call {
            // `fs::read_to_string` reads the entire file into a heap String.
            crate::schema::ToolCall::ReadFile { path } => {
                log_debug!("ToolExecutor::execute — ReadFile path={:?}", path);
                let content = fs::read_to_string(&path)?;
                log_debug!("ToolExecutor::execute — ReadFile read {} bytes from {:?}", content.len(), path);
                Ok(content)
            }
            crate::schema::ToolCall::WriteFile { path, content } => {
                log_debug!(
                    "ToolExecutor::execute — WriteFile path={:?} content_len={}",
                    path, content.len()
                );
                // `fs::write` creates the file if it doesn't exist and truncates if it does.
                fs::write(&path, &content)?;
                log_info!("ToolExecutor::execute — WriteFile wrote {} bytes to {:?}", content.len(), path);
                Ok("File written".into())
            }
            crate::schema::ToolCall::RunCommand { command } => {
                if !is_command_allowed(&command) {
                    return Ok("Command not allowed".into());
                }
                run_command_unchecked(&command)
            }
            crate::schema::ToolCall::ListDir { path } => {
                log_debug!("ToolExecutor::execute — ListDir path={:?}", path);
                // `read_dir` returns an iterator of `DirEntry` results; we unwrap each
                // entry individually so a single unreadable entry doesn't abort the whole listing.
                let entries = fs::read_dir(&path)?;
                let mut names = vec![];
                for e in entries {
                    match e {
                        Ok(entry) => {
                            // `file_name()` returns an OsString; `to_string_lossy` converts it to a
                            // UTF-8 String, replacing any non-UTF-8 bytes with `?`.
                            names.push(entry.file_name().to_string_lossy().to_string());
                        }
                        Err(e) => {
                            // A single unreadable entry (permissions, broken symlink) should not
                            // abort the whole listing — log and skip.
                            log_warn!("ToolExecutor::execute — ListDir skipping unreadable entry in {:?}: {}", path, e);
                        }
                    }
                }
                log_debug!("ToolExecutor::execute — ListDir found {} entries in {:?}", names.len(), path);
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
    log_debug!("generate_diff — old_len={} new_len={}", old.len(), new.len());
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
    log_debug!("generate_diff — diff_len={}", output.len());
    output
}
