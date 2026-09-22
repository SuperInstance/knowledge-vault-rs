//! Search Example: Advanced vector similarity search with filtering
//!
//! This example demonstrates:
//! 1. Basic similarity search
//! 2. Advanced search with options (threshold, filters)
//! 3. Filtering by document type
//! 4. Filtering by document IDs
//! 5. Custom search strategies
//! 6. Result ranking and relevance
//!
//! Run with:
//! ```bash
//! cargo run --example search
//! ```

use knowledge_vault::{
    KnowledgeVault, DocumentIndexer, LocalEmbedder, SearchOptions, VectorSearch,
};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN) // Reduce noise
        .init();

    println!("🔍 knowledge-vault-rs - Search Example\n");

    // Setup: Create vault and index sample documents
    println!("📚 Setting up knowledge vault...\n");

    let vault = KnowledgeVault::open("examples/search_vault.db", 384)?;
    let embedder = Arc::new(Mutex::new(LocalEmbedder::new(384)?));

    let vault_arc = Arc::new(Mutex::new(vault.clone()));
    let (indexer, _handle) = DocumentIndexer::new(
        vault_arc.clone(),
        embedder.clone(),
        Default::default(),
    );

    // Index diverse documents
    let documents = vec![
        // Programming languages
        ("doc1", "code", "Rust is a systems programming language that guarantees memory safety and prevents data races. It's designed for performance and concurrency."),
        ("doc2", "code", "Python is a high-level programming language known for its simplicity and readability. It's widely used in data science, AI, and web development."),
        ("doc3", "code", "JavaScript is the language of the web, enabling interactive and dynamic content in browsers. Node.js allows server-side JavaScript development."),

        // Machine learning
        ("doc4", "text", "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming."),
        ("doc5", "text", "Deep learning uses neural networks with many layers to automatically discover patterns in data. It's achieved remarkable success in computer vision and NLP."),
        ("doc6", "text", "Natural language processing (NLP) enables computers to understand, interpret, and generate human language. Applications include translation, sentiment analysis, and chatbots."),

        // Databases
        ("doc7", "markdown", "SQLite is a lightweight, serverless database engine that stores data in a single file. It's ideal for embedded systems and mobile applications."),
        ("doc8", "markdown", "PostgreSQL is a powerful, open-source relational database with advanced features like ACID compliance, complex queries, and extensibility."),
        ("doc9", "markdown", "Vector databases specialize in storing and querying high-dimensional vectors. They're essential for AI applications requiring semantic search."),

        // Web development
        ("doc10", "code", "React is a JavaScript library for building user interfaces with component-based architecture. It enables efficient updates through virtual DOM diffing."),
        ("doc11", "code", "HTML provides the structure and content of web pages, while CSS handles styling and layout. Together with JavaScript, they form the core of web development."),
    ];

    for (id, doc_type, content) in &documents {
        indexer
            .index_content(
                content.to_string(),
                id.to_string(),
                doc_type.to_string(),
                None,
            )
            .await?;

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    }

    println!("✓ Indexed {} documents\n", documents.len());

    // Example 1: Basic search
    println!("1️⃣  Basic Search (Top 5 Results)\n");

    let query = "programming languages";
    println!("  Query: \"{}\"\n", query);

    let query_emb = {
        let embedder_guard = embedder.lock().await;
        embedder_guard.embed(query).await?
    };

    let results = vault.search(&query_emb, 5)?;

    println!("  Results:");
    for (i, result) in results.iter().enumerate() {
        println!("  {}. Score: {:.4}", i + 1, result.score);
        println!("     Document: {}", result.document_title);
        println!("     Type: {}", result.document_path.as_deref().unwrap_or("N/A"));

        let preview = if result.content.len() > 80 {
            format!("{}...", &result.content[..80])
        } else {
            result.content.clone()
        };
        println!("     Content: \"{}\"\n", preview);
    }

    // Example 2: Threshold filtering
    println!("2️⃣  Search with Threshold (Score ≥ 0.70)\n");

    let query = "artificial intelligence";
    println!("  Query: \"{}\"\n", query);

    let query_emb = {
        let embedder_guard = embedder.lock().await;
        embedder_guard.embed(query).await?
    };

    let options = SearchOptions {
        limit: 10,
        threshold: 0.70, // Only high-quality results
        ..Default::default()
    };

    let search = VectorSearch::new(&vault);
    let results = search.search(&query_emb, &options).await?;

    println!("  Found {} results with score ≥ 0.70:\n", results.len());

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

    // Example 3: Filter by document type
    println!("3️⃣  Filter by Document Type\n");

    let query = "web development";
    println!("  Query: \"{}\"\n", query);

    let query_emb = {
        let embedder_guard = embedder.lock().await;
        embedder_guard.embed(query).await?
    };

    // Filter for code documents only
    let options = SearchOptions {
        limit: 5,
        threshold: 0.5,
        doc_types: Some(vec!["code".to_string()]),
        ..Default::default()
    };

    let results = search.search(&query_emb, &options).await?;

    println!("  Results (code documents only):\n");

    for (i, result) in results.iter().enumerate() {
        println!("  {}. Score: {:.4}", i + 1, result.score);
        println!("     Document: {}", result.document_title);
        println!("     Content: \"{}\"\n", result.content);
    }

    // Example 4: Filter by document IDs
    println!("4️⃣  Filter by Document IDs\n");

    let query = "data storage";
    println!("  Query: \"{}\"\n", query);

    let query_emb = {
        let embedder_guard = embedder.lock().await;
        embedder_guard.embed(query).await?
    };

    // Only search specific documents
    let options = SearchOptions {
        limit: 10,
        threshold: 0.3, // Lower threshold for limited search
        doc_ids: Some(vec!["doc7".to_string(), "doc8".to_string(), "doc9".to_string()]),
        ..Default::default()
    };

    let results = search.search(&query_emb, &options).await?;

    println!("  Results (doc7, doc8, doc9 only):\n");

    for (i, result) in results.iter().enumerate() {
        println!("  {}. Score: {:.4}", i + 1, result.score);
        println!("     Document: {}", result.document_title);
        println!("     Content: \"{}\"\n", result.content);
    }

    // Example 5: Combined filters
    println!("5️⃣  Combined Filters (Type + Threshold)\n");

    let query = "machine learning";
    println!("  Query: \"{}\"\n", query);

    let query_emb = {
        let embedder_guard = embedder.lock().await;
        embedder_guard.embed(query).await?
    };

    let options = SearchOptions {
        limit: 5,
        threshold: 0.65,
        doc_types: Some(vec!["text".to_string()]),
        ..Default::default()
    };

    let results = search.search(&query_emb, &options).await?;

    println!("  Results (text documents, score ≥ 0.65):\n");

    for (i, result) in results.iter().enumerate() {
        println!("  {}. Score: {:.4}", i + 1, result.score);
        println!("     Document: {}", result.document_title);
        println!("     Content: \"{}\"\n", result.content);
    }

    // Example 6: Exclude content from results
    println!("6️⃣  Search Without Content (Metadata Only)\n");

    let query = "rust programming";
    println!("  Query: \"{}\"\n", query);

    let query_emb = {
        let embedder_guard = embedder.lock().await;
        embedder_guard.embed(query).await?
    };

    let options = SearchOptions {
        limit: 5,
        threshold: 0.5,
        include_content: false, // Don't include chunk content
        ..Default::default()
    };

    let results = search.search(&query_emb, &options).await?;

    println!("  Results (metadata only):\n");

    for (i, result) in results.iter().enumerate() {
        println!("  {}. Score: {:.4}", i + 1, result.score);
        println!("     Document: {}", result.document_title);
        println!("     Chunk: {} (offset: {}-{})",
            result.chunk_index,
            result.start_offset,
            result.end_offset
        );
        println!("     Content: <excluded>\n");
    }

    // Example 7: Comparison across queries
    println!("7️⃣  Query Comparison\n");

    let queries = vec![
        "database technology",
        "web framework",
        "AI algorithms",
    ];

    println!("  Comparing results across different queries:\n");

    for query in queries {
        let query_emb = {
            let embedder_guard = embedder.lock().await;
            embedder_guard.embed(query).await?
        };

        let results = vault.search(&query_emb, 2)?;

        println!("  Query: \"{}\"", query);
        for (i, result) in results.iter().enumerate() {
            println!("    {}. {} (Score: {:.4})", i + 1, result.document_title, result.score);
        }
        println!();
    }

    // Example 8: Relevance distribution
    println!("8️⃣  Relevance Score Distribution\n");

    let query = "programming language";
    println!("  Query: \"{}\"\n", query);

    let query_emb = {
        let embedder_guard = embedder.lock().await;
        embedder_guard.embed(query).await?
    };

    let results = vault.search(&query_emb, 100)?; // Get many results

    if !results.is_empty() {
        // Calculate statistics
        let scores: Vec<f32> = results.iter().map(|r| r.score).collect();
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_score = scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let avg_score = scores.iter().sum::<f32>() / scores.len() as f32;

        println!("  Score statistics ({} results):", results.len());
        println!("    Max: {:.4}", max_score);
        println!("    Min: {:.4}", min_score);
        println!("    Avg: {:.4}", avg_score);

        // Show distribution
        let high = scores.iter().filter(|&&s| s >= 0.8).count();
        let medium = scores.iter().filter(|&&s| s >= 0.6 && s < 0.8).count();
        let low = scores.iter().filter(|&&s| s < 0.6).count();

        println!("\n  Distribution:");
        println!("    High (≥ 0.8): {}", high);
        println!("    Medium (0.6-0.8): {}", medium);
        println!("    Low (< 0.6): {}", low);
        println!();
    }

    // Example 9: Practical RAG pipeline
    println!("9️⃣  Practical RAG Pipeline Example\n");

    let user_question = "What is Rust used for?";
    println!("  User Question: \"{}\"\n", user_question);

    // Step 1: Retrieve relevant context
    let query_emb = {
        let embedder_guard = embedder.lock().await;
        embedder_guard.embed(user_question).await?
    };

    let options = SearchOptions {
        limit: 2,
        threshold: 0.6,
        ..Default::default()
    };

    let results = search.search(&query_emb, &options).await?;

    // Step 2: Build context from retrieved chunks
    let context = if results.is_empty() {
        "No relevant information found.".to_string()
    } else {
        results
            .iter()
            .map(|r| format!("Document: {}\n{}", r.document_title, r.content))
            .collect::<Vec<_>>()
            .join("\n\n")
    };

    println!("  Retrieved Context:\n");
    println!("  {}\n", context);

    // Step 3: In real RAG, you would send this to an LLM
    println!("  ✅ In a real RAG system, this context would be");
    println!("     sent to an LLM along with the user question.");
    println!("     The LLM would use the retrieved context to");
    println!("     generate a more accurate and grounded answer.\n");

    // Display vault statistics
    println!("📊 Vault Statistics:");
    let stats = vault.stats()?;

    println!("  Documents: {}", stats.document_count);
    println!("  Chunks: {}", stats.chunk_count);
    println!("  Embeddings: {}", stats.embedding_count);
    println!("  Database size: {:.2} KB", stats.database_size_bytes as f64 / 1024.0);

    println!("\n✅ Example complete!");
    println!("\n💡 Search Tips:");
    println!("   - Use threshold (0.6-0.8) to filter low-quality results");
    println!("   - Filter by doc_type to search specific document categories");
    println!("   - Use doc_ids to limit search to known relevant documents");
    println!("   - Set include_content=false if you only need metadata");
    println!("   - Top 3-5 results usually provide sufficient context for RAG");

    Ok(())
}

/// Helper: Format score as percentage
#[allow(dead_code)]
fn score_to_percentage(score: f32) -> String {
    format!("{:.1}%", score * 100.0)
}

/// Helper: Get relevance label
#[allow(dead_code)]
fn relevance_label(score: f32) -> &'static str {
    match score {
        s if s >= 0.8 => "High",
        s if s >= 0.6 => "Medium",
        _ => "Low",
    }
}
