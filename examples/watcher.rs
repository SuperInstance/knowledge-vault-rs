//! File watching: monitor a directory and stream index commands as files change.
//!
//! Run with: `cargo run --example watcher`
//!
//! `FileWatcher` watches directories (checksum-deduplicated, debounced) and
//! sends `IndexCommand`s over a tokio mpsc channel. Wire the receiving end to
//! `DocumentIndexer` in production; this example just prints the commands.

use std::time::Duration;

use tokio::sync::mpsc;

use knowledge_vault::{IndexCommand, WatchConfig, FileWatcher};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = std::env::temp_dir().join("kv-watcher-example");
    std::fs::create_dir_all(&tmp)?;

    let config = WatchConfig {
        directories: vec![tmp.clone()],
        // Only react to markdown/txt/rs files; None would watch everything.
        extensions: Some(vec!["md".into(), "txt".into(), "rs".into()]),
        // Substring-matched against each path; e.g. "target/" or ".git/".
        exclude_patterns: vec![],
        // Wait for edits to settle before emitting a command.
        debounce: Duration::from_millis(200),
        recursive: false,
    };

    // The channel consumer is the ingestion hook: in a real service this is
    // where DocumentIndexer::index_file(path) gets called.
    let (tx, mut rx) = mpsc::channel::<IndexCommand>(64);

    let mut watcher = FileWatcher::with_auto_index(config, tx)?;
    watcher.start().await?;
    println!("watching {} (existing files scanned for checksums)", tmp.display());

    // Touch two files: the first write indexes; the second write to the SAME
    // file inside the debounce window collapses into one command (dedup by
    // checksum happens for unchanged content).
    std::fs::write(tmp.join("hello.md"), "# Hello\n\nFirst version.")?;
    tokio::time::sleep(Duration::from_millis(50)).await;
    std::fs::write(tmp.join("hello.md"), "# Hello\n\nSecond version with more text.")?;
    std::fs::write(tmp.join("notes.txt"), "watcher demo note")?;

    // Consume commands for a moment, then shut the watcher down cleanly.
    let deadline = tokio::time::sleep(Duration::from_millis(1200));
    tokio::pin!(deadline);
    let mut received = 0;
    loop {
        tokio::select! {
            cmd = rx.recv() => {
                match cmd {
                    Some(IndexCommand::IndexFile(p)) => {
                        received += 1;
                        println!("index command #{received}: {}", p.display());
                    }
                    Some(other) => println!("other command: {other:?}"),
                    None => break,
                }
            }
            _ = &mut deadline => break,
        }
    }

    watcher.stop();
    println!("stopped; received {received} index commands");
    let _ = std::fs::remove_dir_all(&tmp);
    Ok(())
}
