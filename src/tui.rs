use ratatui::{
    backend::CrosstermBackend,
    Terminal,
    widgets::{Block, Borders, Paragraph, Wrap},
    style::{Style, Color, Modifier},
    layout::{Layout, Constraint, Direction},
    text::{Line, Span},
};
use std::time::SystemTime;
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

// Dracula palette
const PURPLE: Color = Color::Rgb(189, 147, 249);
const PINK:   Color = Color::Rgb(255, 121, 198);
const CYAN:   Color = Color::Rgb(139, 233, 253);
const GREEN:  Color = Color::Rgb(80,  250, 123);
const ORANGE: Color = Color::Rgb(255, 184, 108);
const RED:    Color = Color::Rgb(255, 85,  85);
const YELLOW: Color = Color::Rgb(241, 250, 140);
const FG:     Color = Color::Rgb(248, 248, 242);
const COMMENT:Color = Color::Rgb(98,  114, 164);

static SPINNER_FRAME: AtomicUsize = AtomicUsize::new(0);

type PlanResultMsg = anyhow::Result<(StepResult, Vec<String>)>;

pub struct App {
    pub logs: Vec<String>,
    pub input: String,
    pub pending: Option<ToolCall>,
    pub processing: bool,
    cursor_pos: usize,
    blink_start: SystemTime,
    processing_task: Option<JoinHandle<()>>,
    result_rx: Option<mpsc::Receiver<PlanResultMsg>>,
    log_scroll: u16,
    scroll_to_bottom: bool,
    current_goal: Option<String>,
    path_request: Option<String>,
    stream_rx: Option<mpsc::Receiver<String>>,
    streaming_text: String,
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
            log_scroll: 0,
            scroll_to_bottom: true,
            current_goal: None,
            path_request: None,
            stream_rx: None,
            streaming_text: String::new(),
        }
    }

    fn push_log(&mut self, msg: String) {
        self.logs.push(msg);
        self.scroll_to_bottom = true;
    }

    fn is_cursor_visible(&self, now: SystemTime) -> bool {
        let elapsed = now.duration_since(self.blink_start).unwrap_or_default().as_millis();
        (elapsed / 500) % 2 == 0
    }

    fn spawn_plan_task(&mut self, agent: &Agent, goal: String) {
        self.processing = true;
        self.streaming_text.clear();

        let (result_tx, result_rx) = mpsc::channel();
        let (stream_tx, stream_rx) = mpsc::channel::<String>();
        self.result_rx = Some(result_rx);
        self.stream_rx = Some(stream_rx);

        let mut agent_clone = agent.clone();
        let task = tokio::spawn(async move {
            let result = tokio::time::timeout(
                std::time::Duration::from_secs(60),
                agent_clone.plan_step(&goal, Some(stream_tx)),
            )
            .await;
            let result = match result {
                Ok(r) => r,
                Err(_) => Err(anyhow::anyhow!("LLM request timed out after 60s")),
            };
            let history = agent_clone.history.clone();
            let _ = result_tx.send(result.map(|r| (r, history)));
        });
        self.processing_task = Some(task);
    }

    pub async fn run(&mut self, agent: &mut Agent) -> anyhow::Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;

        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        loop {
            let now = SystemTime::now();
            let cursor_visible = self.is_cursor_visible(now);

            // Drain incoming stream tokens into streaming_text
            if let Some(rx) = &self.stream_rx {
                while let Ok(token) = rx.try_recv() {
                    self.streaming_text.push_str(&token);
                }
            }

            // Poll for completed background LLM task
            if let Some(rx) = &self.result_rx {
                if let Ok(msg) = rx.try_recv() {
                    self.result_rx = None;
                    self.processing_task = None;
                    self.stream_rx = None;
                    self.streaming_text.clear();
                    self.processing = false;
                    match msg {
                        Ok((StepResult::Proposed(action), history)) => {
                            agent.history = history;
                            match &action {
                                ToolCall::WriteFile { path, content } => {
                                    let old = std::fs::read_to_string(path).unwrap_or_default();
                                    let diff = crate::tools::generate_diff(&old, content);
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
                                _ => {
                                    self.push_log(format!("Proposed: {:?}", action));
                                }
                            }
                            self.push_log("Approve? (y/n)".into());
                            self.pending = Some(action);
                        }
                        Ok((StepResult::Final(msg), history)) => {
                            agent.history = history;
                            self.push_log(format!("DONE: {}", msg));
                            self.current_goal = None;
                        }
                        Ok((StepResult::Error(err), history)) => {
                            agent.history = history;
                            self.push_log(format!("Error: {}", err));
                        }
                        Err(e) => {
                            self.push_log(format!("Error: {}", e));
                        }
                    }
                }
            }

            SPINNER_FRAME.fetch_add(1, Ordering::SeqCst);

            // Compute log scroll to stay at bottom unless user has scrolled up
            if self.scroll_to_bottom {
                let term_size = terminal.size()?;
                let log_height = (term_size.height * 75 / 100).saturating_sub(2);
                let log_lines: u16 = self.logs.iter()
                    .map(|l| l.lines().count().max(1) as u16)
                    .sum();
                self.log_scroll = log_lines.saturating_sub(log_height);
            }

            terminal.draw(|f| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Percentage(75),
                        Constraint::Percentage(15),
                        Constraint::Percentage(10),
                    ])
                    .split(f.size());

                // --- Logs panel ---
                let log_lines = build_log_lines(&self.logs);
                let logs = Paragraph::new(log_lines)
                    .block(
                        Block::default()
                            .title(Span::styled(
                                " Logs (↑/↓) ",
                                Style::default().fg(PURPLE).add_modifier(Modifier::BOLD),
                            ))
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(PURPLE)),
                    )
                    .scroll((self.log_scroll, 0));

                // --- Action panel ---
                let frame = SPINNER_FRAME.load(Ordering::SeqCst) % 6;
                let action_lines: Vec<Line> = if self.processing {
                    let icon = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴"][frame];
                    let mut lines = vec![Line::from(vec![
                        Span::styled(
                            format!("{} ", icon),
                            Style::default().fg(GREEN).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("Processing...", Style::default().fg(ORANGE)),
                    ])];
                    if !self.streaming_text.is_empty() {
                        let s = &self.streaming_text;
                        let start = s
                            .char_indices()
                            .rev()
                            .nth(199)
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        for tl in s[start..].lines() {
                            lines.push(Line::from(Span::styled(
                                tl.to_string(),
                                Style::default().fg(CYAN),
                            )));
                        }
                    }
                    lines
                } else {
                    vec![Line::from(Span::styled(
                        " Idle",
                        Style::default().fg(COMMENT),
                    ))]
                };

                let action_border_color = if self.processing { ORANGE } else { CYAN };
                let action_title_color = if self.processing { ORANGE } else { CYAN };
                let action_block = Block::default()
                    .title(Span::styled(
                        " Action ",
                        Style::default().fg(action_title_color).add_modifier(Modifier::BOLD),
                    ))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(action_border_color));

                let status = Paragraph::new(action_lines)
                    .block(action_block)
                    .wrap(Wrap { trim: false });

                // --- Input panel ---
                let input_line: Line = if self.pending.is_some() {
                    Line::from("")
                } else {
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

                let (input_border_color, input_title_text) = if self.path_request.is_some() {
                    (GREEN, " File path (Enter to confirm) ")
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
                );

                f.render_widget(logs, chunks[0]);
                f.render_widget(status, chunks[1]);
                f.render_widget(input, chunks[2]);
            })?;

            if event::poll(std::time::Duration::from_millis(100))? {
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
                            if self.pending.is_none() {
                                self.cursor_pos += c.len_utf8();
                                self.input.push(c);
                            } else {
                                match c {
                                    'y' => {
                                        if let Some(action) = self.pending.take() {
                                            if let ToolCall::WriteFile { path, content } = action {
                                                // Always ask the user to confirm or correct the path.
                                                // Pre-fill input with the LLM's suggestion so they can
                                                // just press Enter if it looks right.
                                                self.input = path.clone();
                                                self.cursor_pos = path.len();
                                                self.path_request = Some(content);
                                                self.push_log(
                                                    "Confirm file path (edit if needed, then Enter):"
                                                        .into(),
                                                );
                                            } else {
                                                match agent.execute_step(action) {
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
                                        if self.result_rx.is_none() {
                                            self.processing = false;
                                        }
                                    }
                                    'n' => {
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
                                let path = self.input.trim().to_string();
                                if path.is_empty() {
                                    self.path_request = Some(content);
                                    continue;
                                }
                                self.input.clear();
                                self.cursor_pos = 0;
                                match std::fs::write(&path, &content) {
                                    Ok(_) => {
                                        agent.history.push(format!("Tool result: File written to {}", path));
                                        self.push_log(format!("Executed: File written to {}", path));
                                        if let Some(goal) = self.current_goal.clone() {
                                            self.spawn_plan_task(agent, goal);
                                        }
                                    }
                                    Err(e) => {
                                        self.push_log(format!("Write failed ({}). Enter file path:", e));
                                        self.path_request = Some(content);
                                    }
                                }
                                continue;
                            }

                            let goal = self.input.trim().to_string();
                            if goal.is_empty() {
                                continue;
                            }

                            self.push_log(format!("> {}", goal));
                            self.input.clear();
                            self.cursor_pos = 0;
                            self.current_goal = Some(goal.clone());
                            self.spawn_plan_task(agent, goal);
                        }
                        KeyCode::Esc => break,
                        _ => {}
                    }
                }
            }
        }

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;

        Ok(())
    }
}

fn build_log_lines(logs: &[String]) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for entry in logs {
        for raw in entry.split('\n') {
            lines.push(color_log_line(raw));
        }
    }
    lines
}

fn color_log_line(line: &str) -> Line<'static> {
    let style = if line.starts_with("> ") {
        Style::default().fg(CYAN).add_modifier(Modifier::BOLD)
    } else if line.starts_with("DONE:") {
        Style::default().fg(GREEN).add_modifier(Modifier::BOLD)
    } else if line.starts_with("Error:")
        || line.starts_with("Execution error:")
        || line.starts_with("Write failed")
        || line == "Action rejected"
    {
        Style::default().fg(RED)
    } else if line.starts_with("Proposed") {
        Style::default().fg(YELLOW)
    } else if line.starts_with("Approve?") {
        Style::default().fg(YELLOW).add_modifier(Modifier::BOLD)
    } else if line.starts_with("Executed:") {
        Style::default().fg(GREEN)
    } else if line.starts_with("Confirm file path")
        || line.starts_with("No path")
        || line.starts_with("Directory for")
    {
        Style::default().fg(ORANGE)
    } else if line.starts_with("Diff:") || line.ends_with("[truncated]") {
        Style::default().fg(COMMENT)
    } else if line.starts_with('+') && !line.starts_with("+++") {
        Style::default().fg(GREEN)
    } else if line.starts_with('-') && !line.starts_with("---") {
        Style::default().fg(RED)
    } else if line.starts_with(' ') && line.len() > 1 {
        // unchanged diff context lines
        Style::default().fg(COMMENT)
    } else if line.starts_with("Welcome.") {
        Style::default().fg(PURPLE).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(FG)
    };

    Line::from(Span::styled(line.to_string(), style))
}
