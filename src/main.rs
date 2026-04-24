mod agent;
mod llm;
mod tools;
mod schema;
mod tui;

use agent::Agent;
use tui::App;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut app = App::new();
    let mut agent = Agent::new("http://localhost:8080/v1");

    app.run(&mut agent).await?;

    Ok(())
}