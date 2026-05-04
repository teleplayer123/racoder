mod agent;
mod llm;
mod logger;
mod tools;
mod schema;
mod tui;

use agent::Agent;
use tui::App;
use clap::Parser;

#[derive(Parser)]
struct Cli {
    #[arg(short, long, default_value = "http://localhost:8080/v1")]
    server_url: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let server_url = cli.server_url;
    let log_path = logger::default_log_path();
    // Logger init failure is non-fatal — the app still works, just without file logging.
    if let Err(e) = logger::init(&log_path) {
        eprintln!("Warning: could not open log file {:?}: {}", log_path, e);
    }

    let session_path = Agent::default_session_path();
    log_info!("startup — session_path={:?}, log_path={:?}", session_path, log_path);
    log_info!("LLM server found: {}", &server_url);

    let (mut agent, restored) = Agent::load_or_new(&server_url, &session_path);

    let mut app = App::new();
    if restored {
        log_info!(
            "session restored — history_entries={}, summary={}",
            agent.history.len(),
            agent.summary.is_some()
        );
        app.push_restored_notice(agent.history.len(), agent.summary.is_some());
    } else {
        log_info!("fresh session started");
    }

    app.run(&mut agent, &session_path).await?;

    log_info!("clean exit");
    Ok(())
}
