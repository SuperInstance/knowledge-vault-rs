//! The full RAG loop, end to end: chunk -> embed -> store -> vector search.
//!
//! Run with: `cargo run --example search`
//!
//! Uses a temporary on-disk vault and `PlaceholderEmbedder` (no model file
//! needed), so the whole pipeline runs in milliseconds. Swap in
//! `LocalEmbedder::load` for real semantic retrieval; every call shape stays
//! the same.

use knowledge_vault::embeddings::EmbeddingProvider;
use knowledge_vault::{Chunker, PlaceholderEmbedder, KnowledgeVault};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = std::env::temp_dir().join("kv-search-example.db");
    let vault = KnowledgeVault::open(&db, 384)?;
    let embedder = PlaceholderEmbedder::new(384);
    let chunker = Chunker::new();

    // 1. Ingest documents: chunk each one, register chunks, store embeddings.
    let docs = [
        ("doc/redis.md", "Redis keeps the whole dataset in memory. Persistence is optional via snapshots.",
         "RocksDB is an embedded key-value store built on log-structured merge trees."),
        ("doc/postgres.md", "Postgres stores rows on disk with WAL. Vector search ships via the pgvector extension.",
         "SQLite is a serverless embedded database; rusqlite is its Rust binding."),
    ];

    for (path, para_a, para_b) in docs {
        // Register the document row (hash-deduplicated).
        let doc_id = vault.add_document(path, &format!("{para_a}\n\n{para_b}"), "markdown")?;

        // Chunk and persist each chunk, then embed + store its vector.
        for chunk in chunker.chunk(&format!("{para_a}\n\n{para_b}"))? {
            let chunk_id = uuid::Uuid::new_v4().to_string();
            vault.insert_chunk(
                &chunk_id,
                &doc_id,
                chunk.index,
                &chunk.content,
                chunk.start_offset,
                chunk.end_offset,
                chunk.token_count,
            )?;
            let embedding = embedder.embed(&chunk.content).await?;
            vault.insert_embedding(&chunk_id, &embedding)?;
        }
    }

    // 2. Query: embed the query with the SAME embedder, then search top-k.
    let query = "embedded key-value store";
    let q_vec = embedder.embed(query).await?;
    let results = vault.search(&q_vec, 3)?;

    println!("query: {query:?}\ntop {} results:", results.len());
    for r in &results {
        println!("  score={:.3} doc={:?} chunk={}", r.score, r.document_title, &r.chunk_id[..8]);
        let preview: String = r.content.chars().take(64).collect();
        println!("          {preview}...");
    }

    // 3. Inspect where a hit came from.
    if let Some(best) = results.first() {
        let chunks = vault.get_chunks(&best.document_id)?;
        println!("\nbest hit's document holds {} chunks total", chunks.len());
    }

    let _ = std::fs::remove_file(&db);
    Ok(())
}
