use ratatui::{
    backend::CrosstermBackend,
    Terminal,
    widgets::{Block, Borders, Paragraph},
    style::{Style, Color, Modifier},
    layout::{Layout, Constraint, Direction},
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

static SPINNER_FRAME: AtomicUsize = AtomicUsize::new(0);

pub struct App {
    pub logs: Vec<String>,
    pub input: String,
    pub pending: Option<ToolCall>,
    pub processing: bool,
    cursor_pos: usize,
    blink_start: SystemTime,
    // Handle for the background task
    processing_task: Option<JoinHandle<StepResult>>,
    // Channel to receive results
    result_rx: Option<mpsc::Receiver<StepResult>>,
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
        }
    }

    fn is_cursor_visible(&self, now: SystemTime) -> bool {
        let elapsed = now.duration_since(self.blink_start).unwrap_or_default().as_secs();
        elapsed % 2 == 0
    }

    pub async fn run(&mut self, agent: &mut Agent) -> anyhow::Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;

        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let now = SystemTime::now();

        loop {
            let cursor_visible = self.is_cursor_visible(now);

            // Update spinner frame for animation
            SPINNER_FRAME.fetch_add(1, Ordering::SeqCst);

            terminal.draw(|f| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Percentage(75),
                        Constraint::Percentage(15),
                        Constraint::Percentage(10),
                    ])
                    .split(f.size());

                let log_text = self.logs.join("\n");

                let logs = Paragraph::new(log_text)
                    .block(Block::default().title("Logs").borders(Borders::ALL));

                // Status gauge for processing state
                let (status_icon, status_text, status_style) = if self.processing {
                    let frame = SPINNER_FRAME.load(Ordering::SeqCst) % 6;
                    let icon = match frame {
                        0 => "⠋",
                        1 => "⠙",
                        2 => "⠹",
                        3 => "⠸",
                        4 => "⠼",
                        5 => "⠴",
                        _ => "⠋",
                    };
                    (icon, "Processing...", Color::Cyan)
                } else {
                    (" ", "Idle", Color::Gray)
                };

                // Create status gauge with dark background for better visibility
                let status_block = Block::default()
                    .title("Action")
                    .borders(Borders::ALL)
                    .border_style(Style::default())
                    .style(Style::default());

                let status_text_style = Style::default()
                    .fg(status_style)
                    .add_modifier(Modifier::BOLD);

                let status_content = format!("{} {}", status_icon, status_text);
                let status = Paragraph::new(status_content)
                    .style(status_text_style)
                    .block(status_block);

                // Build input text with cursor embedded at cursor_pos
                let input_text = if self.pending.is_some() {
                    String::new()
                } else {
                    let mut input_text = String::new();
                    input_text.push_str(&self.input[..self.cursor_pos]);
                    input_text.push_str(if cursor_visible { "█" } else { " " });
                    input_text.push_str(&self.input[self.cursor_pos..]);
                    input_text
                };

                let input = Paragraph::new(input_text.as_str())
                    .block(Block::default().title("Input").borders(Borders::ALL));

                f.render_widget(logs, chunks[0]);
                f.render_widget(status, chunks[1]);
                f.render_widget(input, chunks[2]);
            })?;

            if event::poll(std::time::Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char(c) => {
                            // if approving/rejecting, don't append to input
                            if self.pending.is_none() {
                                self.input.push(c);
                                self.cursor_pos += 1;
                            } else {
                                match c {
                                    'y' => {
                                        if let Some(action) = self.pending.take() {
                                            match agent.execute_step(action) {
                                                Ok(result) => {
                                                    self.logs.push(format!("Executed: {}", result));
                                                }
                                                Err(e) => {
                                                    self.logs.push(format!("Execution error: {}", e));
                                                }
                                            }
                                        }
                                    }
                                    'n' => {
                                        self.pending = None;
                                        self.processing = false;
                                        self.logs.push("Action rejected".into());
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
                                // ignore enter if awaiting approval
                                continue;
                            }
                            if self.result_rx.is_some() || self.processing_task.is_some() {
                                // already processing, ignore
                                continue;
                            }

                            self.processing = true;

                            let goal = self.input.trim().to_string();
                            if goal.is_empty() {
                                self.processing = false;
                                continue;
                            }

                            self.cursor_pos = 0;
                            self.logs.push(format!("> {}", goal));

                            match agent.plan_step(&goal).await? {
                                StepResult::Proposed(action) => {
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

                                            self.logs.push(format!("Proposed WRITE to: {}", path));
                                            self.logs.push("Diff:".into());
                                            self.logs.push(diff);
                                        }
                                        _ => {
                                            self.logs.push(format!("Proposed: {:?}", action));
                                        }
                                    }
                                    self.logs.push("Approve? (y/n)".into());
                                    self.pending = Some(action);
                                }
                                StepResult::Final(msg) => {
                                    self.logs.push(format!("DONE: {}", msg));
                                }
                                StepResult::Error(err) => {
                                    self.logs.push(format!("Error: {}", err));
                                }
                            }

                            self.input.clear();
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
