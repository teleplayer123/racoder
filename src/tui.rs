use ratatui::{
    backend::CrosstermBackend,
    Terminal,
    widgets::{Block, Borders, Paragraph},
    layout::{Layout, Constraint, Direction},
};
use crossterm::{
    terminal::{enable_raw_mode, disable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    execute,
    event::{self, Event, KeyCode},
};
use std::io;

use crate::agent::{Agent, StepResult};
use crate::schema::ToolCall;

pub struct App {
    pub logs: Vec<String>,
    pub input: String,
    pub pending: Option<ToolCall>,
}

impl App {
    pub fn new() -> Self {
        Self {
            logs: vec!["Welcome. Type a goal and press Enter.".into()],
            input: String::new(),
            pending: None,
        }
    }

    pub async fn run(&mut self, agent: &mut Agent) -> anyhow::Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;

        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        loop {
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

                let pending_text = match &self.pending {
                    Some(p) => format!("Pending: {:?} (y=approve / n=reject)", p),
                    None => "No pending action".into(),
                };

                let pending = Paragraph::new(pending_text)
                    .block(Block::default().title("Action").borders(Borders::ALL));

                let input = Paragraph::new(self.input.as_str())
                    .block(Block::default().title("Input").borders(Borders::ALL));

                f.render_widget(logs, chunks[0]);
                f.render_widget(pending, chunks[1]);
                f.render_widget(input, chunks[2]);
            })?;

            if event::poll(std::time::Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char(c) => {
                            // if approving/rejecting, don't append to input
                            if self.pending.is_none() {
                                self.input.push(c);
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
                                        self.logs.push("Action rejected".into());
                                    }
                                    _ => {}
                                }
                            }
                        }
                        KeyCode::Backspace => {
                            if self.pending.is_none() {
                                self.input.pop();
                            }
                        }
                        KeyCode::Enter => {
                            if self.pending.is_some() {
                                // ignore enter if awaiting approval
                                continue;
                            }

                            let goal = self.input.trim().to_string();
                            if goal.is_empty() {
                                continue;
                            }

                            self.logs.push(format!("> {}", goal));

                            match agent.plan_step(&goal).await? {
                                StepResult::Proposed(action) => {
                                    self.logs.push(format!("Proposed: {:?}", action));
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
