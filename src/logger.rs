use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

// Single global log file, initialized once at startup.
// `OnceLock` gives safe one-time initialization without a runtime lock on reads.
// `BufWriter` batches writes to reduce syscalls; we flush explicitly after each line
// so entries survive a crash or panic before the buffer would otherwise be flushed.
static LOG: OnceLock<Mutex<BufWriter<File>>> = OnceLock::new();

/// Open (or create) the log file at `path` and register it as the global logger.
/// Must be called once before any log macros are used. Silently no-ops on subsequent calls.
pub fn init(path: &std::path::Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        // Create ~/.racoder/ if it doesn't exist yet.
        std::fs::create_dir_all(parent)?;
    }
    // `append(true)` preserves existing log content across restarts instead of truncating.
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    // `OnceLock::set` fails if already initialized — treat that as a no-op.
    let _ = LOG.set(Mutex::new(BufWriter::new(file)));
    Ok(())
}

/// Returns the default log file path: ~/.racoder/debug.log
pub fn default_log_path() -> std::path::PathBuf {
    std::env::var("HOME")
        .map(|h| std::path::PathBuf::from(h).join(".racoder").join("debug.log"))
        .unwrap_or_else(|_| std::path::PathBuf::from("racoder_debug.log"))
}

/// Formats the current wall-clock time as HH:MM:SS.mmm (UTC).
/// Avoids pulling in a date/time crate by computing from the UNIX epoch manually.
fn timestamp() -> String {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    let ms   = d.subsec_millis();
    let h = (secs % 86400) / 3600;
    let m = (secs % 3600)  / 60;
    let s =  secs % 60;
    format!("{:02}:{:02}:{:02}.{:03}", h, m, s, ms)
}

/// Write one log line. Called exclusively by the log macros below.
/// If the logger was never initialized (e.g. in tests), the message is silently discarded.
pub fn write_log(level: &str, module: &str, msg: &str) {
    if let Some(lock) = LOG.get() {
        if let Ok(mut w) = lock.lock() {
            let _ = writeln!(w, "[{}] {:<5} {} — {}", timestamp(), level, module, msg);
            // Flush immediately so the entry is on disk before any subsequent panic.
            let _ = w.flush();
        }
    }
}

// ── Macros ────────────────────────────────────────────────────────────────────
// `#[macro_export]` places these at the crate root so they can be used anywhere
// in the crate without a `use` statement, just like built-in macros.
// `$crate` refers to the crate where the macro is defined (racoder), ensuring the
// path resolves correctly regardless of where the macro is invoked.
// `module_path!()` in the caller produces e.g. "racoder::agent".

#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)*) => {
        $crate::logger::write_log("DEBUG", module_path!(), &format!($($arg)*))
    };
}

#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        $crate::logger::write_log("INFO", module_path!(), &format!($($arg)*))
    };
}

#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {
        $crate::logger::write_log("WARN", module_path!(), &format!($($arg)*))
    };
}

#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {
        $crate::logger::write_log("ERROR", module_path!(), &format!($($arg)*))
    };
}
