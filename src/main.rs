mod agent;
mod llm;
mod tools;
mod schema;
mod tui;

use agent::Agent;
use tui::App;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let session_path = Agent::default_session_path();
    let (mut agent, restored) = Agent::load_or_new("http://localhost:8080/v1", &session_path);

    let mut app = App::new();
    if restored {
        app.push_restored_notice(agent.history.len(), agent.summary.is_some());
    }

    app.run(&mut agent, &session_path).await?;

    Ok(())
}
