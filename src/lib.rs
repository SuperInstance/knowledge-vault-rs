//! # Knowledge Vault - Vector Database & RAG
//!
//! A standalone Rust library for managing document storage, chunking, embedding generation,
//! and vector similarity search. Extracted from the SuperInstance AI project.
//!
//! ## Architecture
//!
//! The knowledge system is built around several components:
//!
//! - **Vault** ([`KnowledgeVault`]): SQLite-based storage with vector similarity search
//! - **Chunker** ([`Chunker`]): Splits documents into optimal-sized chunks for embedding
//! - **Embeddings** ([`LocalEmbedder`]): Generates vector embeddings for text
//! - **Indexer** ([`DocumentIndexer`]): Automates document ingestion and indexing
//! - **Watcher** ([`FileWatcher`]): Monitors files for changes and auto-reindexes
//! - **Search** ([`VectorSearch`]): Performs semantic similarity queries
//!
//! ## Usage Example
//!
//! For a no-model quick start use [`PlaceholderEmbedder`] (deterministic
//! SHA256-derived vectors) with [`LegacyDocumentIndexer`]; runnable versions
//! of every stage live in `examples/` (`basic`, `chunking`, `embeddings`,
//! `search`, `watcher`).
//!
//! ```rust,no_run,ignore
//! use knowledge_vault::{KnowledgeVault, PlaceholderEmbedder, EmbeddingProvider};
//! use knowledge_vault::indexer::LegacyDocumentIndexer;
//!
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! // Open vault with 384-dimensional embeddings (BGE-micro)
//! let vault = KnowledgeVault::open("knowledge.db".as_ref(), 384)?;
//!
//! // Index content through the indexer (chunk -> persist -> embed)
//! let embedder = PlaceholderEmbedder::new(384); // or LocalEmbedder::load(model_path)?
//! let indexer = LegacyDocumentIndexer::new(&vault, &embedder);
//! indexer.index_file(std::path::Path::new("README.md")).await?;
//!
//! // Search for similar content (same embedder for the query!)
//! let query_embedding = embedder.embed("How do I search documents?").await?;
//! let results = vault.search(&query_embedding, 5)?;
//!
//! for result in results {
//!     println!("Score: {:.2} | Content: {}", result.score, result.content);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Document Storage
//!
//! Documents are stored with the following metadata:
//!
//! - **Content Hash**: SHA256 for deduplication
//! - **Chunks**: Document split into optimal pieces (default: 512 tokens)
//! - **Embeddings**: Vector representations for each chunk
//! - **Type Detection**: Automatic classification (code, markdown, text)
//!
//! ## Vector Search
//!
//! The vault supports two search modes:
//!
//! 1. **VSS (Virtual Table)**: Fast approximate nearest neighbor search
//!    - Requires SQLite-VSS extension
//!    - Best for large datasets (>10k chunks)
//!
//! 2. **Cosine Similarity**: Exact similarity calculation
//!    - Pure Rust implementation
//!    - Fallback when VSS unavailable
//!    - Suitable for smaller datasets
//!
//! ## File Watching
//!
//! Enable automatic reindexing when files change:
//!
//! ```rust,no_run,ignore
//! use knowledge_vault::{FileWatcher, WatchConfig};
//! use notify::RecursiveMode;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = WatchConfig {
//!     paths: vec!["./docs".into()],
//!     recursive: RecursiveMode::Recursive,
//!     ..Default::default()
//! };
//!
//! let watcher = FileWatcher::new(config)?.await?;
//! watcher.start().await?;
//! # Ok(())
//! # }
//! ```

pub mod chunker;
pub mod embeddings;
pub mod indexer;
pub mod search;
pub mod vault;
pub mod watcher;

pub use chunker::{Chunk, ChunkOptions, Chunker};
pub use embeddings::{
    cosine_similarity, euclidean_distance, normalize_embedding, BatchOptions, ChunkMetadata,
    DocType, DocumentChunker, EmbeddingPipeline, EmbeddingProvider, LocalEmbedder,
    PlaceholderEmbedder,
};
pub use indexer::{DocumentIndexer, IndexCommand, IndexResult, IndexerConfig, IndexerHandle};
pub use search::{HybridSearch, SearchOptions, SearchResult, VectorSearch};
pub use vault::{ChunkRecord, ChunkResult, Document, KnowledgeVault, VaultStats};
pub use watcher::{FileChange, FileWatcher, WatchConfig};

/// Result type for knowledge operations
pub type KnowledgeResult<T> = std::result::Result<T, KnowledgeError>;

/// Knowledge error types
#[derive(Debug, thiserror::Error)]
pub enum KnowledgeError {
    #[error("Document not found: {0}")]
    NotFound(String),

    #[error("Invalid document format: {0}")]
    InvalidFormat(String),

    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("SQLite error: {0}")]
    SqliteError(#[from] rusqlite::Error),

    #[error("Watch error: {0}")]
    WatchError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}
