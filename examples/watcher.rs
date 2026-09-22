//! File Watcher Example: Auto-indexing documents on file changes
//!
//! This example demonstrates:
//! 1. Creating a file watcher configuration
//! 2. Watching directories for changes
//! 3. Auto-indexing modified files
//! 4. Debouncing to avoid redundant indexing
//! 5. Extension filtering
//!
//! To test this example:
//! 1. Run the program: cargo run --example watcher
//! 2. In another terminal, modify files in examples/watched_docs/
//! 3. Watch the indexer react to changes
//!
//! Run with:
//! ```bash
//! cargo run --example watcher
//! ```

use knowledge_vault::{
    KnowledgeVault, DocumentIndexer, LocalEmbedder, FileWatcher, WatchConfig,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("👁️  knowledge-vault-rs - File Watcher Example\n");

    // Step 1: Create watched directory
    let watch_dir = PathBuf::from("examples/watched_docs");

    println!("📁 Creating watched directory: {:?}\n", watch_dir);

    std::fs::create_dir_all(&watch_dir)?;

    // Step 2: Create sample documents
    println!("📝 Creating sample documents...\n");

    let sample_files = vec![
        ("rust_intro.md", "# Introduction to Rust\n\nRust is a systems programming language that guarantees memory safety."),
        ("python_basics.md", "# Python Basics\n\nPython is known for its simplicity and readability."),
        ("ml_guide.md", "# Machine Learning Guide\n\nML enables computers to learn from data without explicit programming."),
    ];

    for (filename, content) in &sample_files {
        let path = watch_dir.join(filename);
        std::fs::write(&path, content)?;
        println!("  Created: {}", filename);
    }
    println!();

    // Step 3: Open vault and create embedder
    println!("🔧 Setting up vault and embedder...\n");

    let vault = KnowledgeVault::open("examples/watcher_vault.db", 384)?;
    let embedder = Arc::new(Mutex::new(LocalEmbedder::new(384)?));

    let vault_arc = Arc::new(Mutex::new(vault.clone()));

    // Step 4: Create indexer
    println!("📥 Creating document indexer...\n");

    let (indexer, _handle) = DocumentIndexer::new(
        vault_arc.clone(),
        embedder.clone(),
        Default::default(),
    );

    // Step 5: Configure watcher
    println!("⚙️  Configuring file watcher...\n");

    let config = WatchConfig {
        directories: vec![watch_dir.clone()],
        extensions: Some(vec![
            "md".to_string(),
            "txt".to_string(),
            "rs".to_string(),
            "py".to_string(),
        ]),
        exclude_patterns: vec![
            ".git".to_string(),
            "node_modules".to_string(),
            "target".to_string(),
        ],
        debounce: Duration::from_secs(2), // Wait 2s for changes to settle
        recursive: true,
    };

    println!("  Watch directory: {:?}", watch_dir);
    println!("  Extensions: {:?}", config.extensions);
    println!("  Debounce: 2 seconds");
    println!("  Recursive: yes\n");

    // Step 6: Create watcher with auto-indexing
    println!("👀 Creating watcher with auto-indexing...\n");

    let command_tx = indexer.command_sender();

    let mut watcher = FileWatcher::with_auto_index(config, command_tx)?;

    // Step 7: Pre-index existing files
    println!("📊 Initial indexing...\n");

    tokio::time::sleep(Duration::from_millis(100)).await;

    let stats = vault.stats()?;
    println!("  Documents in vault: {}", stats.document_count);
    println!("  Chunks: {}", stats.chunk_count);
    println!();

    // Step 8: Start watching (in background task)
    println!("🚀 Starting file watcher...\n");
    println!("  The watcher is now monitoring {:?} for changes.\n", watch_dir);
    println!("  Try these experiments in another terminal:\n");

    println!("  1. Modify a file:");
    println!("     echo 'New content' >> examples/watched_docs/rust_intro.md\n");

    println!("  2. Create a new file:");
    println!("     echo '# New Document' > examples/watched_docs/new.md\n");

    println!("  3. Rename a file:");
    println!("     mv examples/watched_docs/python_basics.md examples/watched_docs/python_advanced.md\n");

    println!("\n  Watch for indexing messages below...\n");
    println!("  {}\n", "=".repeat(60));

    // Spawn watcher in background
    let watch_dir_clone = watch_dir.clone();
    let watcher_task = tokio::spawn(async move {
        // Watch for changes (this blocks)
        if let Err(e) = watcher.start() {
            eprintln!("Watcher error: {}", e);
        }
    });

    // Step 9: Simulate some changes for demonstration
    println!("🎭 Simulating file changes...\n");

    // Wait a bit for watcher to start
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Modify a file
    println!("  ➕ Modifying rust_intro.md...");
    let rust_path = watch_dir.join("rust_intro.md");
    let existing_content = std::fs::read_to_string(&rust_path)?;
    let new_content = format!("{}\n\n## Memory Safety\nRust's ownership system prevents data races at compile time.", existing_content);
    std::fs::write(&rust_path, new_content)?;
    println!("  ✓ File modified\n");

    // Wait for watcher to process
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Create a new file
    println!("  ➕ Creating javascript_guide.md...");
    let js_path = watch_dir.join("javascript_guide.md");
    std::fs::write(&js_path, "# JavaScript Guide\n\nJavaScript powers interactive web content.")?;
    println!("  ✓ File created\n");

    // Wait for watcher to process
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Show updated stats
    println!("📊 Updated vault statistics:\n");

    let stats = vault.stats()?;
    println!("  Documents: {} (+{})", stats.document_count, stats.document_count - 3);
    println!("  Chunks: {}", stats.chunk_count);
    println!("  Embeddings: {}", stats.embedding_count);
    println!();

    // Step 10: Search the vault
    println!("🔍 Testing search...\n");

    let query = "memory safety in rust";
    println!("  Query: \"{}\"\n", query);

    let query_emb = {
        let embedder_guard = embedder.lock().await;
        embedder_guard.embed(query).await?
    };

    let results = vault.search(&query_emb, 3)?;

    if !results.is_empty() {
        println!("  Found {} relevant chunks:\n", results.len());

        for (i, result) in results.iter().enumerate() {
            println!("  {}. Score: {:.4}", i + 1, result.score);
            println!("     Document: {}", result.document_title);

            let preview = if result.content.len() > 80 {
                format!("{}...", &result.content[..80])
            } else {
                result.content.clone()
            };
            println!("     Content: \"{}\"\n", preview);
        }
    } else {
        println!("  No results found (embeddings may still be processing)\n");
    }

    // Step 11: List all documents
    println!("📚 All documents in vault:\n");

    let docs = vault.list_documents(100)?;
    for doc in docs {
        println!("  • {} ({})", doc.title, doc.doc_type);
        println!("    Path: {:?}", doc.path);
        println!("    Chunks: {}, Size: {} bytes", doc.chunk_count, doc.size_bytes);
        println!("    Indexed: {}", doc.indexed_at.format("%Y-%m-%d %H:%M:%S"));
        println!();
    }

    println!("\n{}", "=".repeat(60));
    println!("\n✅ Watcher is still running! Press Ctrl+C to stop.\n");

    println!("💡 The watcher will continue to monitor for changes.");
    println!("   Any new, modified, or deleted files will be automatically indexed.\n");

    // Keep the watcher running
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            println!("\n🛑 Received Ctrl+C, stopping watcher...\n");
            watcher_task.abort();
        }
        _ = tokio::time::sleep(Duration::from_secs(30)) => {
            println!("\n⏱️  Demo timeout (30s), stopping watcher...\n");
            watcher_task.abort();
        }
    }

    println!("✅ Example complete!");
    println!("\n💡 File Watcher Features:");
    println!("   - Auto-indexes new files");
    println!("   - Re-indexes modified files (SHA256 diff detection)");
    println!("   - Removes deleted files from vault");
    println!("   - Updates paths for renamed files");
    println!("   - Debounced to avoid redundant indexing");
    println!("   - Extension filtering to ignore irrelevant files");
    println!("   - Recursive directory watching");

    Ok(())
}

/// Example: Manual file watching (without auto-indexer)
#[allow(dead_code)]
async fn manual_watching_example() -> Result<(), Box<dyn std::error::Error>> {
    use knowledge_vault::FileChange;
    use tokio::sync::mpsc;

    println!("🔧 Manual File Watching Example\n");

    // Create watcher without auto-indexing
    let config = WatchConfig {
        directories: vec![PathBuf::from("./docs")],
        ..Default::default()
    };

    // Create channel for file change events
    let (change_tx, mut change_rx) = mpsc::channel(100);

    // Note: You would need to implement FileWatcher::with_channel
    // to manually receive change events instead of auto-indexing
    //
    // let mut watcher = FileWatcher::with_channel(config, change_tx)?;
    // watcher.start()?;

    // Process change events manually
    while let Some(change) = change_rx.recv().await {
        match change {
            FileChange::Modified(path) => {
                println!("File modified: {:?}", path);
                // Manually trigger indexing
            }
            FileChange::Deleted(path) => {
                println!("File deleted: {:?}", path);
                // Manually remove from vault
            }
            FileChange::Renamed(old_path, new_path) => {
                println!("File renamed: {:?} -> {:?}", old_path, new_path);
                // Manually update path in vault
            }
        }
    }

    Ok(())
}

/// Example: Custom debounce strategy
#[allow(dead_code)]
fn custom_debounce_config() -> WatchConfig {
    // For frequently changing files (logs, temp files)
    WatchConfig {
        debounce: Duration::from_secs(5), // Longer debounce
        ..Default::default()
    }
}

/// Example: Watching multiple directories
#[allow(dead_code)]
fn multi_directory_config() -> WatchConfig {
    WatchConfig {
        directories: vec![
            PathBuf::from("./src"),
            PathBuf::from("./docs"),
            PathBuf::from("./examples"),
        ],
        extensions: Some(vec![
            "rs".to_string(),
            "md".to_string(),
            "toml".to_string(),
        ]),
        exclude_patterns: vec![
            "target".to_string(),
            ".git".to_string(),
        ],
        debounce: Duration::from_secs(2),
        recursive: true,
    }
}

/// Example: Watch-specific file types only
#[allow(dead_code)]
fn code_only_config() -> WatchConfig {
    WatchConfig {
        directories: vec![PathBuf::from("./src")],
        extensions: Some(vec![
            "rs".to_string(),
            "toml".to_string(),
        ]),
        ..Default::default()
    }
}
