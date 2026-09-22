//! Basic Example: Index a file and search for similar content
//!
//! This example demonstrates the core workflow:
//! 1. Open a knowledge vault
//! 2. Create an embedder (BGE-Micro, 384 dimensions)
//! 3. Index a document (auto-chunking)
//! 4. Search for similar content
//! 5. Display results with similarity scores
//!
//! Run with:
//! ```bash
//! cargo run --example basic
//! ```

use knowledge_vault::{KnowledgeVault, DocumentIndexer, LocalEmbedder};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🔍 knowledge-vault-rs - Basic Example\n");

    // Step 1: Open or create a knowledge vault
    // The vault stores documents, chunks, and embeddings in SQLite
    println!("📂 Opening vault...");
    let vault = KnowledgeVault::open("examples/basic_vault.db", 384)?;
    println!("✓ Vault opened\n");

    // Step 2: Create an embedder
    // BGE-Micro generates 384-dimensional embeddings
    // Falls back to SHA256 placeholder if model unavailable
    println!("🧠 Initializing embedder (BGE-Micro, 384 dims)...");
    let embedder = Arc::new(Mutex::new(LocalEmbedder::new(384)?));
    println!("✓ Embedder ready\n");

    // Step 3: Create sample documents
    println!("📝 Creating sample documents...");
    let sample_docs = vec![
        (
            "doc1.txt",
            "Machine learning is a subset of artificial intelligence that \
             focuses on algorithms that can learn from data. It enables \
             computers to improve their performance on a specific task \
             through experience.",
        ),
        (
            "doc2.txt",
            "Deep learning is a type of machine learning that uses neural \
             networks with many layers. These networks can automatically \
             discover patterns in data without explicit programming.",
        ),
        (
            "doc3.txt",
            "Natural language processing (NLP) is a branch of AI that helps \
             computers understand, interpret, and manipulate human language. \
             It powers chatbots, translation, and sentiment analysis.",
        ),
        (
            "doc4.txt",
            "Rust is a systems programming language that guarantees memory \
             safety and prevents data races. It's designed for performance \
             and reliability, making it ideal for critical systems.",
        ),
    ];

    // Step 4: Index documents
    println!("📥 Indexing documents...\n");

    // Create indexer (spawns background task)
    let vault_arc = Arc::new(Mutex::new(vault.clone()));
    let (indexer, _handle) = DocumentIndexer::new(
        vault_arc.clone(),
        embedder.clone(),
        Default::default(),
    );

    for (filename, content) in &sample_docs {
        println!("  - Indexing {}...", filename);

        // Index content directly (in-memory, no file needed)
        indexer
            .index_content(
                content.to_string(),
                filename.to_string(),
                "text".to_string(),
                None, // No file path
            )
            .await?;

        // Small delay to let indexer process
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    println!("✓ All documents indexed\n");

    // Step 5: Perform semantic search
    println!("🔎 Performing semantic searches...\n");

    // Define test queries
    let queries = vec![
        "What is machine learning?",
        "How do neural networks work?",
        "Tell me about programming languages",
    ];

    for query in &queries {
        println!("Query: \"{}\"\n", query);

        // Generate embedding for the query
        let query_emb = {
            let embedder_guard = embedder.lock().await;
            embedder_guard.embed(query).await?
        };

        // Search for similar chunks
        let results = vault.search(&query_emb, 3)?;

        if results.is_empty() {
            println!("  No results found.\n");
            continue;
        }

        // Display results
        for (i, result) in results.iter().enumerate() {
            println!(
                "  Result #{} (Score: {:.2})",
                i + 1,
                result.score
            );

            // Truncate content for display
            let content_preview = if result.content.len() > 100 {
                format!("{}...", &result.content[..100])
            } else {
                result.content.clone()
            };

            println!("    Document: {}", result.document_title);
            println!("    Content: {}", content_preview);
            println!();
        }
    }

    // Step 6: Display vault statistics
    println!("📊 Vault Statistics:");
    let stats = vault.stats()?;

    println!("  Documents: {}", stats.document_count);
    println!("  Chunks: {}", stats.chunk_count);
    println!("  Embeddings: {}", stats.embedding_count);
    println!("  Total size: {} bytes", stats.total_size_bytes);
    println!("  DB size: {} bytes", stats.database_size_bytes);
    println!("  Dimensions: {}", stats.embedding_dimensions);

    println!("\n✅ Example complete!");
    println!("💾 Vault saved to: examples/basic_vault.db");

    Ok(())
}

/// Helper function to calculate similarity between two texts
#[allow(dead_code)]
async fn similarity_between(
    embedder: Arc<Mutex<LocalEmbedder>>,
    text1: &str,
    text2: &str,
) -> Result<f32, Box<dyn std::error::Error>> {
    let embedder_guard = embedder.lock().await;

    let emb1 = embedder_guard.embed(text1).await?;
    let emb2 = embedder_guard.embed(text2).await?;

    // Calculate cosine similarity
    let dot_product: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = emb2.iter().map(|x| x * x).sum::<f32>().sqrt();

    Ok(dot_product / (norm1 * norm2))
}
