use ratatui::{
    backend::CrosstermBackend,
    Terminal,
    widgets::{Block, Borders, Paragraph, Wrap},
    style::{Style, Color, Modifier},
    layout::{Layout, Constraint, Direction},
    text::{Line, Span},
};
use std::time::{SystemTime, Instant};
use crossterm::{
    terminal::{enable_raw_mode, disable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    execute,
    event::{self, Event, KeyCode},
};
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use tokio::task::JoinHandle;

use crate::agent::{Agent, StepResult};
use crate::schema::ToolCall;
use crate::{log_error, log_info, log_warn};

// Dracula color palette — defined as constants so any change here propagates everywhere.
const PURPLE: Color = Color::Rgb(189, 147, 249);
const PINK:   Color = Color::Rgb(255, 121, 198);
const CYAN:   Color = Color::Rgb(139, 233, 253);
const GREEN:  Color = Color::Rgb(80,  250, 123);
const ORANGE: Color = Color::Rgb(255, 184, 108);
const RED:    Color = Color::Rgb(255, 85,  85);
const YELLOW: Color = Color::Rgb(241, 250, 140);
const FG:     Color = Color::Rgb(248, 248, 242);
const COMMENT:Color = Color::Rgb(98,  114, 164);

// Global frame counter for the braille spinner animation. AtomicUsize lets both the
// event loop and the render closure (same thread, different stack frames) read it
// without a mutex.
static SPINNER_FRAME: AtomicUsize = AtomicUsize::new(0);

// The message type sent from the background LLM task back to the main event loop.
// Wraps the step result together with the agent's updated history snapshot so the
// main thread can merge it back into the authoritative agent without shared state.
type PlanResultMsg = anyhow::Result<(StepResult, Vec<String>)>;

pub struct App {
    pub logs: Vec<String>,
    pub input: String,
    // The action the LLM has proposed but the user has not yet approved or rejected.
    pub pending: Option<ToolCall>,
    pub processing: bool,
    // Byte offset into `input` for the cursor (not char count — needed for multi-byte chars).
    cursor_pos: usize,
    // Timestamp of the last cursor blink toggle, used to compute blink phase from elapsed time.
    blink_start: SystemTime,
    // Handle to the spawned tokio task; kept so we can call `.abort()` on Esc.
    processing_task: Option<JoinHandle<()>>,
    // Receives the final plan result (StepResult + updated history) from the background task.
    result_rx: Option<mpsc::Receiver<PlanResultMsg>>,
    // Receives individual token strings as the LLM streams its response.
    stream_rx: Option<mpsc::Receiver<String>>,
    // Accumulates streamed tokens for the live preview in the Action panel.
    streaming_text: String,
    log_scroll: u16,
    // When true, the scroll position snaps to the bottom on every frame.
    // Set to false when the user manually scrolls up; reset on any new log entry.
    scroll_to_bottom: bool,
    // The goal currently being pursued. Kept so the agent can re-plan automatically
    // after each approved action without the user re-typing the goal.
    current_goal: Option<String>,
    // When set, the Input panel is in "path confirmation" mode — holds the file content
    // that will be written once the user confirms the path and presses Enter.
    path_request: Option<String>,
    // When set, the Input panel is in "command override" mode — holds a blocked command
    // that the user is being asked to approve running unchecked.
    command_override: Option<String>,
    // Wall-clock time when the current LLM task was spawned, used for the elapsed-time display.
    processing_start: Option<Instant>,
}

impl App {
    pub fn new() -> Self {
        Self {
            logs: vec!["Welcome. Type a goal and press Enter.".into()],
            input: String::new(),
            pending: None,
            processing: false,
            cursor_pos: 0,
            blink_start: SystemTime::now(),
            processing_task: None,
            result_rx: None,
            stream_rx: None,
            streaming_text: String::new(),
            log_scroll: 0,
            scroll_to_bottom: true,
            current_goal: None,
            path_request: None,
            command_override: None,
            processing_start: None,
        }
    }

    fn push_log(&mut self, msg: String) {
        self.logs.push(msg);
        // Any new log entry resets scroll to the bottom so the latest message is visible.
        self.scroll_to_bottom = true;
    }

    pub fn push_restored_notice(&mut self, history_len: usize, has_summary: bool) {
        let summary_note = if has_summary { ", summary: yes" } else { "" };
        self.push_log(format!(
            "Session restored ({} history entries{}).",
            history_len, summary_note
        ));
    }

    /// Returns whether the cursor block should be visible based on a 500ms blink period.
    fn is_cursor_visible(&self, now: SystemTime) -> bool {
        let elapsed = now.duration_since(self.blink_start).unwrap_or_default().as_millis();
        // Integer division gives the number of completed 500ms intervals; even = visible.
        (elapsed / 500) % 2 == 0
    }

    /// Clone the agent, create both channels, and spawn the async LLM task.
    /// The agent is cloned because tokio tasks require 'static ownership; history is
    /// merged back from the clone once the task completes via `result_rx`.
    fn spawn_plan_task(&mut self, agent: &Agent, goal: String) {
        self.processing = true;
        self.processing_start = Some(Instant::now());
        self.streaming_text = String::new();

        // Two separate channels:
        // - result_rx: receives the final (StepResult, history) pair when the task finishes
        // - stream_rx: receives individual token strings as the LLM streams them
        let (result_tx, result_rx) = mpsc::channel();
        let (stream_tx, stream_rx) = mpsc::channel();
        self.result_rx = Some(result_rx);
        self.stream_rx = Some(stream_rx);

        log_info!("spawn_plan_task — goal={:?}", goal);

        let mut agent_clone = agent.clone();
        // `tokio::spawn` runs the async block as an independent task on the tokio runtime,
        // freeing the main thread (and the TUI event loop) to continue drawing and
        // processing input while the LLM call is in progress.
        let task = tokio::spawn(async move {
            let result = agent_clone.plan_step(&goal, Some(stream_tx)).await;
            // Send both the result and the updated history in a single message so the
            // main thread can atomically merge them without a partial update window.
            let history = agent_clone.history.clone();
            let _ = result_tx.send(result.map(|r| (r, history)));
        });
        self.processing_task = Some(task);
    }

    pub async fn run(&mut self, agent: &mut Agent, session_path: &std::path::Path) -> anyhow::Result<()> {
        // `enable_raw_mode` disables the terminal's line buffering and echo so each
        // keypress is delivered immediately as a crossterm event rather than waiting
        // for a newline. Must be paired with `disable_raw_mode` on exit.
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        // `EnterAlternateScreen` switches to a separate terminal buffer, preserving the
        // user's existing shell output. `LeaveAlternateScreen` restores it on exit.
        execute!(stdout, EnterAlternateScreen)?;

        // `CrosstermBackend` is the ratatui backend that writes to a `Write` implementor.
        // `Terminal` wraps it and manages double-buffering: it diffs the previous frame
        // against the new one and only redraws changed cells, minimising flicker.
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        loop {
            let now = SystemTime::now();
            let cursor_visible = self.is_cursor_visible(now);

            // `try_recv` is non-blocking: it returns immediately with `Empty` if the
            // background task hasn't finished yet, letting the loop keep drawing.
            if let Some(rx) = &self.result_rx {
                match rx.try_recv() {
                // ── task completed normally ──────────────────────────────
                Ok(msg) => {
                    // Drop all task-related state before handling the result.
                    self.result_rx = None;
                    self.stream_rx = None;
                    self.streaming_text = String::new();
                    self.processing_task = None;
                    self.processing_start = None;
                    self.processing = false;
                    match msg {
                        Ok((StepResult::Proposed(action), history)) => {
                            log_info!("task result — Proposed({:?}), merging history_entries={}", action, history.len());
                            // Merge the background task's history back into the authoritative agent.
                            agent.history = history;
                            match &action {
                                ToolCall::WriteFile { path, content } => {
                                    if path.is_empty() {
                                        // LLM omitted the path field; the path_request flow
                                        // (triggered on 'y' approval) will prompt the user for it.
                                        self.push_log("Proposed WRITE (no path given — you will be asked)".into());
                                    } else {
                                        // Show a diff of what would change before asking for approval.
                                        let old = std::fs::read_to_string(&path).unwrap_or_default();
                                        let diff = crate::tools::generate_diff(&old, &content);
                                        let diff = if diff.len() > 2000 {
                                            let truncated: String = diff.chars().take(2000).collect();
                                            format!("{}...\n[truncated]", truncated)
                                        } else {
                                            diff
                                        };
                                        self.push_log(format!("Proposed WRITE to: {}", path));
                                        self.push_log("Diff:".into());
                                        self.push_log(diff);
                                    }
                                }
                                _ => {
                                    self.push_log(format!("Proposed: {:?}", action));
                                }
                            }
                            self.push_log("Approve? (y/n)".into());
                            self.pending = Some(action);
                        }
                        Ok((StepResult::Final(msg), history)) => {
                            log_info!("task result — Final, merging history_entries={}", history.len());
                            agent.history = history;
                            self.push_log(format!("DONE: {}", msg));
                            self.current_goal = None;
                            // Save on Final so context is preserved even if the user quits
                            // without pressing Esc.
                            if let Err(e) = agent.save(session_path) {
                                self.push_log(format!("Warning: session save failed: {}", e));
                            }
                        }
                        Ok((StepResult::Error(err), history)) => {
                            log_warn!("task result — StepResult::Error: {}", err);
                            agent.history = history;
                            self.push_log(format!("Error: {}", err));
                        }
                        Err(e) => {
                            log_error!("task result — anyhow error from plan_step: {}", e);
                            self.push_log(format!("Error: {}", e));
                        }
                    }
                }
                // The sender was dropped without sending — the task panicked or was aborted
                // externally. Reset state so the user can start a new request.
                Err(mpsc::TryRecvError::Disconnected) => {
                    log_error!("task result — Disconnected (task panicked or was aborted externally)");
                    self.result_rx = None;
                    self.stream_rx = None;
                    self.streaming_text = String::new();
                    self.processing_task = None;
                    self.processing_start = None;
                    self.processing = false;
                    self.push_log("Error: LLM task ended unexpectedly.".into());
                }
                // Task is still running — nothing to do this frame.
                Err(mpsc::TryRecvError::Empty) => {}
                }
            }

            // Drain all available streaming tokens into the preview buffer. `try_recv`
            // in a loop takes everything queued since the last frame without blocking.
            if let Some(rx) = &self.stream_rx {
                while let Ok(token) = rx.try_recv() {
                    self.streaming_text.push_str(&token);
                }
            }

            // Increment the spinner frame counter every loop iteration. The modulo in the
            // render code maps this to one of 6 braille characters for a smooth animation.
            SPINNER_FRAME.fetch_add(1, Ordering::SeqCst);

            // Compute the scroll offset needed to keep the latest log entry visible.
            // Only recalculated when `scroll_to_bottom` is true; manual scrolling clears it.
            if self.scroll_to_bottom {
                let term_size = terminal.size()?;
                let log_height = (term_size.height * 75 / 100).saturating_sub(2);
                // Subtract 2 for the left and right borders of the Logs panel to get the
                // usable character width that ratatui's wrap logic will use.
                let panel_width = term_size.width.saturating_sub(2) as usize;
                // Count visual rows rather than raw newline-delimited lines. A long line
                // that wraps across N terminal rows must be counted as N rows so the scroll
                // offset lands precisely at the bottom of the last entry.
                let visual_rows: u16 = self.logs.iter()
                    .flat_map(|entry| entry.split('\n'))
                    .map(|line| {
                        let chars = line.chars().count();
                        if panel_width == 0 || chars == 0 {
                            1u16
                        } else {
                            ((chars + panel_width - 1) / panel_width) as u16
                        }
                    })
                    .sum();
                self.log_scroll = visual_rows.saturating_sub(log_height);
            }

            // `terminal.draw` renders exactly one frame. The closure receives a `Frame`
            // which abstracts over the terminal size and the cell buffer. Ratatui computes
            // a diff against the previous frame and writes only changed cells to stdout.
            terminal.draw(|f| {
                // `Layout` divides the screen into a stack of rectangles using the
                // Cassowary constraint solver (the same algorithm used by Apple's Auto Layout).
                // `split(f.size())` returns the computed rectangles for each constraint.
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Percentage(75), // Logs
                        Constraint::Percentage(15), // Action
                        Constraint::Percentage(10), // Input
                    ])
                    .split(f.size());

                // --- Logs panel ---
                let log_lines = build_log_lines(&self.logs);
                // `Paragraph` is a ratatui widget that renders styled text with optional
                // scrolling. `.scroll((row, col))` offsets the view by that many lines/columns.
                // `Wrap { trim: false }` soft-wraps long lines at the panel boundary without
                // stripping leading spaces, so indented content (diffs, paths) stays readable.
                let logs = Paragraph::new(log_lines)
                    .block(
                        // `Block` draws a border frame with an optional title around any widget.
                        Block::default()
                            .title(Span::styled(
                                " Logs (↑/↓) ",
                                Style::default().fg(PURPLE).add_modifier(Modifier::BOLD),
                            ))
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(PURPLE)),
                    )
                    .wrap(Wrap { trim: false })
                    .scroll((self.log_scroll, 0));

                // --- Action panel ---
                // `SPINNER_FRAME % 6` cycles through the 6 braille spinner characters.
                let frame = SPINNER_FRAME.load(Ordering::SeqCst) % 6;
                let action_lines: Vec<Line> = if self.processing {
                    let icon = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴"][frame];
                    let elapsed = self.processing_start
                        .map(|s| s.elapsed().as_secs())
                        .unwrap_or(0);
                    // Show the last 200 characters of streamed tokens. Slicing by char
                    // (not bytes) avoids splitting multi-byte UTF-8 sequences.
                    let preview: String = if self.streaming_text.is_empty() {
                        String::new()
                    } else {
                        let chars: Vec<char> = self.streaming_text.chars().collect();
                        let start = chars.len().saturating_sub(200);
                        chars[start..].iter().collect()
                    };
                    // `Line::from(vec![...])` composes multiple styled `Span`s into one
                    // horizontal line of text with mixed styles.
                    let mut lines = vec![Line::from(vec![
                        Span::styled(
                            format!("{} ", icon),
                            Style::default().fg(GREEN).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("Processing... ", Style::default().fg(ORANGE)),
                        Span::styled(
                            format!("({}s)", elapsed),
                            Style::default().fg(COMMENT),
                        ),
                        Span::styled("  Esc to cancel", Style::default().fg(COMMENT)),
                    ])];
                    if !preview.is_empty() {
                        lines.push(Line::from(Span::styled(preview, Style::default().fg(COMMENT))));
                    }
                    lines
                } else {
                    vec![Line::from(Span::styled(" Idle", Style::default().fg(COMMENT)))]
                };

                let action_border_color = if self.processing { ORANGE } else { CYAN };
                let action_title_color  = if self.processing { ORANGE } else { CYAN };
                let action_block = Block::default()
                    .title(Span::styled(
                        " Action ",
                        Style::default().fg(action_title_color).add_modifier(Modifier::BOLD),
                    ))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(action_border_color));

                // `Wrap { trim: false }` enables soft-wrapping without stripping leading spaces,
                // which preserves the indentation of streamed token preview lines.
                let status = Paragraph::new(action_lines)
                    .block(action_block)
                    .wrap(Wrap { trim: false });

                // --- Input panel ---
                // When a y/n prompt is active (pending action or command override), the input
                // box is hidden — those keystrokes are handled separately in the event loop.
                let input_line: Line = if self.pending.is_some() || self.command_override.is_some() {
                    Line::from("")
                } else {
                    // Split `input` at `cursor_pos` (a byte index) to render text on both sides
                    // of the blinking cursor block. The cursor renders as a bold yellow █ or a
                    // space depending on the current blink phase.
                    let before = &self.input[..self.cursor_pos];
                    let after  = &self.input[self.cursor_pos..];
                    let cursor_char = if cursor_visible { "█" } else { " " };
                    Line::from(vec![
                        Span::styled(before.to_string(), Style::default().fg(FG)),
                        Span::styled(
                            cursor_char,
                            Style::default().fg(YELLOW).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(after.to_string(), Style::default().fg(FG)),
                    ])
                };

                // The Input panel border color and title change to communicate the current mode:
                // pink = normal input, green = confirming a file path, red = command override.
                let (input_border_color, input_title_text) = if self.path_request.is_some() {
                    (GREEN, " File path (Enter to confirm) ")
                } else if self.command_override.is_some() {
                    (RED, " Override blocked command? (y/n) ")
                } else {
                    (PINK, " Input ")
                };
                let input = Paragraph::new(input_line).block(
                    Block::default()
                        .title(Span::styled(
                            input_title_text,
                            Style::default().fg(input_border_color).add_modifier(Modifier::BOLD),
                        ))
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(input_border_color)),
                ).wrap(Wrap { trim: true });

                // `render_widget` draws each widget into its assigned rectangle.
                f.render_widget(logs, chunks[0]);
                f.render_widget(status, chunks[1]);
                f.render_widget(input, chunks[2]);
            })?;

            // `event::poll` returns true if an event is ready within the given timeout,
            // without blocking the loop. 100ms keeps the UI responsive while allowing the
            // spinner and streaming preview to update ~10 times per second.
            if event::poll(std::time::Duration::from_millis(100))? {
                // `event::read` is guaranteed non-blocking here because poll returned true.
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Up => {
                            self.log_scroll = self.log_scroll.saturating_sub(1);
                            self.scroll_to_bottom = false;
                        }
                        KeyCode::Down => {
                            self.log_scroll = self.log_scroll.saturating_add(1);
                            self.scroll_to_bottom = false;
                        }
                        KeyCode::Char(c) => {
                            if self.command_override.is_some() {
                                // Command override mode: y runs the blocked command unchecked,
                                // n records it as declined and re-plans so the LLM can try
                                // a different approach.
                                match c {
                                    'y' => {
                                        let cmd = self.command_override.take().unwrap();
                                        log_info!("command override — user approved: {:?}", cmd);
                                        match crate::tools::run_command_unchecked(&cmd) {
                                            Ok(result) => {
                                                agent.history.push(format!("Tool result: {}", result));
                                                self.push_log(format!("Override executed: {}", result));
                                                if let Some(goal) = self.current_goal.clone() {
                                                    self.spawn_plan_task(agent, goal);
                                                }
                                            }
                                            Err(e) => {
                                                self.push_log(format!("Execution error: {}", e));
                                            }
                                        }
                                        if self.result_rx.is_none() {
                                            self.processing = false;
                                        }
                                    }
                                    'n' => {
                                        let cmd = self.command_override.take().unwrap();
                                        log_info!("command override — user declined: {:?}", cmd);
                                        // Inject the refusal into history so the LLM knows the
                                        // command was intentionally blocked and can adapt.
                                        agent.history.push(format!(
                                            "Tool result: Command `{}` blocked — user declined override",
                                            cmd
                                        ));
                                        self.push_log("Override declined — command not executed".into());
                                        if let Some(goal) = self.current_goal.clone() {
                                            self.spawn_plan_task(agent, goal);
                                        } else if self.result_rx.is_none() {
                                            self.processing = false;
                                        }
                                    }
                                    _ => {}
                                }
                            } else if self.pending.is_none() {
                                // Normal typing: advance cursor by the UTF-8 byte length of the
                                // character, not by 1, so multi-byte characters don't corrupt the offset.
                                self.cursor_pos += c.len_utf8();
                                self.input.push(c);
                            } else {
                                // Action approval mode: y/n decides whether to execute the pending action.
                                match c {
                                    'y' => {
                                        if let Some(action) = self.pending.take() {
                                            log_info!("action approved — {:?}", action);
                                            match action {
                                                ToolCall::WriteFile { path, content } => {
                                                    // Sanitize before pre-filling: if the LLM proposed
                                                    // an absolute path outside the CWD (e.g. /usr/local/bin/),
                                                    // strip it to just the filename so the user only has to
                                                    // confirm a safe default rather than manually fixing it.
                                                    let safe_path = crate::tools::sanitize_write_path(&path);
                                                    if safe_path != path {
                                                        self.push_log(format!(
                                                            "Path restricted: {} → {} (outside CWD)",
                                                            path, safe_path
                                                        ));
                                                    }
                                                    self.cursor_pos = safe_path.len();
                                                    self.input = safe_path;
                                                    self.path_request = Some(content);
                                                    self.push_log(
                                                        "Confirm file path (edit if needed, then Enter):".into(),
                                                    );
                                                }
                                                ToolCall::RunCommand { command } => {
                                                    if crate::tools::is_command_allowed(&command) {
                                                        match agent.execute_step(ToolCall::RunCommand { command }) {
                                                            Ok(result) => {
                                                                self.push_log(format!("Executed: {}", result));
                                                                if let Some(goal) = self.current_goal.clone() {
                                                                    self.spawn_plan_task(agent, goal);
                                                                }
                                                            }
                                                            Err(e) => {
                                                                self.push_log(format!("Execution error: {}", e));
                                                            }
                                                        }
                                                    } else {
                                                        // Command is outside the allowlist; escalate to
                                                        // the override flow rather than silently refusing.
                                                        self.push_log(format!("Command blocked: `{}`", command));
                                                        self.push_log("Override and run anyway? (y/n)".into());
                                                        self.command_override = Some(command);
                                                    }
                                                }
                                                other => {
                                                    match agent.execute_step(other) {
                                                        Ok(result) => {
                                                            self.push_log(format!("Executed: {}", result));
                                                            if let Some(goal) = self.current_goal.clone() {
                                                                self.spawn_plan_task(agent, goal);
                                                            }
                                                        }
                                                        Err(e) => {
                                                            self.push_log(format!("Execution error: {}", e));
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        if self.result_rx.is_none() {
                                            self.processing = false;
                                        }
                                    }
                                    'n' => {
                                        log_info!("action rejected by user");
                                        self.pending = None;
                                        self.processing = false;
                                        self.current_goal = None;
                                        self.push_log("Action rejected".into());
                                    }
                                    _ => {}
                                }
                            }
                        }
                        KeyCode::Backspace => {
                            if self.pending.is_none() && !self.input.is_empty() {
                                // `pop` removes the last character (not byte), so cursor_pos
                                // must be recomputed from the new string length in bytes.
                                self.input.pop();
                                self.cursor_pos = self.input.len();
                            }
                        }
                        KeyCode::Enter => {
                            if self.pending.is_some() {
                                continue;
                            }
                            if self.result_rx.is_some() || self.processing_task.is_some() {
                                continue;
                            }

                            if let Some(content) = self.path_request.take() {
                                // Path confirmation: the user has typed (or accepted) a file path.
                                // Put path_request back and wait if the field is still empty.
                                let path = self.input.trim().to_string();
                                if path.is_empty() {
                                    self.path_request = Some(content);
                                    continue;
                                }
                                // Final safety check before writing. The user may have
                                // manually typed an absolute path outside the CWD; block it
                                // and re-prompt rather than silently writing to a system dir.
                                if !crate::tools::is_path_within_cwd(&path) {
                                    log_warn!("write blocked — {:?} is outside CWD", path);
                                    self.push_log(format!(
                                        "Blocked: {} is outside the working directory. Use a relative path.",
                                        path
                                    ));
                                    self.path_request = Some(content);
                                    continue;
                                }
                                self.input.clear();
                                self.cursor_pos = 0;
                                log_info!("path confirmed — writing to {:?} content_len={}", path, content.len());
                                match std::fs::write(&path, &content) {
                                    Ok(_) => {
                                        log_info!("file write succeeded — {:?}", path);
                                        agent.history.push(format!("Tool result: File written to {}", path));
                                        self.push_log(format!("Executed: File written to {}", path));
                                        if let Some(goal) = self.current_goal.clone() {
                                            self.spawn_plan_task(agent, goal);
                                        }
                                    }
                                    Err(e) => {
                                        // Write failed (e.g. bad path, permissions). Re-engage
                                        // the path prompt with the error shown so the user can correct it.
                                        log_warn!("file write failed — path={:?} error={}", path, e);
                                        self.push_log(format!("Write failed ({}). Enter file path:", e));
                                        self.path_request = Some(content);
                                    }
                                }
                                continue;
                            }

                            // Normal goal submission: clear input, record the goal, and
                            // kick off the first plan step.
                            let goal = self.input.trim().to_string();
                            if goal.is_empty() {
                                continue;
                            }

                            log_info!("new goal submitted — {:?}", goal);
                            self.push_log(format!("> {}", goal));
                            self.input.clear();
                            self.cursor_pos = 0;
                            self.current_goal = Some(goal.clone());
                            self.spawn_plan_task(agent, goal);
                        }
                        KeyCode::Esc => {
                            if self.processing {
                                // `task.abort()` sends a cancellation signal; the task is dropped
                                // at its next `.await` point. We don't wait for confirmation.
                                log_info!("Esc — cancelling in-progress task");
                                if let Some(task) = self.processing_task.take() {
                                    task.abort();
                                }
                                self.result_rx = None;
                                self.stream_rx = None;
                                self.streaming_text = String::new();
                                self.processing = false;
                                self.processing_start = None;
                                self.current_goal = None;
                                self.push_log("Request cancelled.".into());
                            } else {
                                // Clean exit: save the session before restoring the terminal.
                                log_info!("Esc — clean exit, saving session");
                                if let Err(e) = agent.save(session_path) {
                                    log_error!("clean-exit save failed: {}", e);
                                }
                                break;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Restore the terminal to its original state. Order matters:
        // disable raw mode before leaving the alternate screen so the shell prompt
        // reappears correctly.
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;

        Ok(())
    }
}

/// Converts raw log strings into styled ratatui `Line` values for rendering.
/// Multi-line entries (e.g. diffs) are split on `\n` so each line can be
/// styled individually by `color_log_line`.
fn build_log_lines(logs: &[String]) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for entry in logs {
        for raw in entry.split('\n') {
            lines.push(color_log_line(raw));
        }
    }
    lines
}

/// Maps a single log line to a colored ratatui `Line` based on its prefix.
/// `Span::styled` wraps a string with a `Style` (fg color + optional modifiers).
/// `Line::from(span)` creates a single-span line, which ratatui renders as one row.
fn color_log_line(line: &str) -> Line<'static> {
    let style = if line.starts_with("> ") {
        Style::default().fg(CYAN).add_modifier(Modifier::BOLD)    // user goal input
    } else if line.starts_with("DONE:") {
        Style::default().fg(GREEN).add_modifier(Modifier::BOLD)   // task complete
    } else if line.starts_with("Error:")
        || line.starts_with("Execution error:")
        || line.starts_with("Write failed")
        || line == "Action rejected"
    {
        Style::default().fg(RED)                                   // errors and rejections
    } else if line.starts_with("Proposed") {
        Style::default().fg(YELLOW)                                // proposed action description
    } else if line.starts_with("Approve?") {
        Style::default().fg(YELLOW).add_modifier(Modifier::BOLD)  // approval prompt
    } else if line.starts_with("Executed:") || line.starts_with("Override executed:") {
        Style::default().fg(GREEN)                                 // successful execution
    } else if line.starts_with("Confirm file path")
        || line.starts_with("No path")
        || line.starts_with("Directory for")
    {
        Style::default().fg(ORANGE)                                // path prompts
    } else if line.starts_with("Command blocked") || line.starts_with("Override") {
        Style::default().fg(ORANGE)                                // security / override notices
    } else if line.starts_with("Diff:") || line.ends_with("[truncated]") {
        Style::default().fg(COMMENT)                               // diff header
    } else if line.starts_with('+') && !line.starts_with("+++") {
        Style::default().fg(GREEN)                                 // diff: added line
    } else if line.starts_with('-') && !line.starts_with("---") {
        Style::default().fg(RED)                                   // diff: removed line
    } else if line.starts_with(' ') && line.len() > 1 {
        Style::default().fg(COMMENT)                               // diff: context line
    } else if line.starts_with("Welcome.") {
        Style::default().fg(PURPLE).add_modifier(Modifier::BOLD)  // startup banner
    } else {
        Style::default().fg(FG)                                    // default text
    };

    Line::from(Span::styled(line.to_string(), style))
}
