//! Basic vault operations: open, add documents, list, inspect stats.
//!
//! Run with: `cargo run --example basic`
//!
//! Creates a temporary on-disk vault (deleted on exit) and does NOT need an
//! embedding model — it exercises the document/metadata side of the API.

use knowledge_vault::KnowledgeVault;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // On-disk SQLite vault. The second argument is the embedding dimensionality
    // used by the vector side (384 for BGE-micro); it only matters once you
    // insert embeddings (see the `search` example).
    let db = std::env::temp_dir().join("kv-basic-example.db");
    let vault = KnowledgeVault::open(&db, 384)?;

    // add_document stores the document row with a SHA256 content hash for
    // deduplication. Chunking + embeddings are separate steps (see the
    // `search` example for the full RAG loop).
    let id_a = vault.add_document(
        "notes/rust-lifetimes.md",
        "# Lifetimes\n\nLifetimes describe how long references remain valid. \
         The compiler checks every reference against these bounds.",
        "markdown",
    )?;
    let id_b = vault.add_document(
        "notes/rust-traits.md",
        "# Traits\n\nA trait defines shared behavior. Types opt in with `impl Trait for Type`.",
        "markdown",
    )?;
    println!("inserted docs: {id_a}, {id_b}");

    // Re-adding identical content is a no-op: the content hash matches and the
    // existing document id is returned.
    let id_again = vault.add_document(
        "notes/rust-lifetimes.md",
        "# Lifetimes\n\nLifetimes describe how long references remain valid. \
         The compiler checks every reference against these bounds.",
        "markdown",
    )?;
    println!("re-added same content -> id {id_again} (deduplicated: {})", id_again == id_a);

    // Fetch one document.
    if let Some(doc) = vault.get_document(&id_a)? {
        println!("\nfetched: {} [{}] hash={}", doc.title, doc.doc_type, &doc.content_hash[..12]);
        println!("  path={:?} size={}B", doc.path, doc.size_bytes);
    }

    // List everything and show aggregate stats.
    let docs = vault.list_documents(10)?;
    println!("\nall documents ({}):", docs.len());
    for d in &docs {
        println!("  - {} ({})", d.title, d.doc_type);
    }

    let stats = vault.stats()?;
    println!("\nstats: {} docs, {} chunks, {} embeddings, db={}B",
        stats.document_count, stats.chunk_count, stats.embedding_count, stats.database_size_bytes);

    // Delete one and confirm.
    vault.delete_document(&id_b)?;
    println!("\nafter delete: {} docs remain", vault.list_documents(10)?.len());

    let _ = std::fs::remove_file(&db);
    Ok(())
}
