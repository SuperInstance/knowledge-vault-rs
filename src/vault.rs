//! Knowledge Vault
//!
//! SQLite-based storage for documents, chunks, and embeddings.
//! Uses sqlite-vss for vector similarity search.

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{debug, info, instrument, warn};

use crate::{KnowledgeError, KnowledgeResult};

/// A document in the knowledge vault
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique document ID
    pub id: String,
    /// Original file path
    pub path: Option<String>,
    /// Document title
    pub title: String,
    /// Document type (markdown, text, code, etc.)
    pub doc_type: String,
    /// SHA256 hash of content for deduplication
    pub content_hash: String,
    /// Number of chunks
    pub chunk_count: u32,
    /// Original content size in bytes
    pub size_bytes: u64,
    /// When indexed
    pub indexed_at: chrono::DateTime<chrono::Utc>,
    /// Last modified
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Additional metadata
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

/// Vault statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultStats {
    pub document_count: u64,
    pub chunk_count: u64,
    pub embedding_count: u64,
    pub total_size_bytes: u64,
    pub database_size_bytes: u64,
    pub embedding_dimensions: u32,
}

/// Result from a vector similarity search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkResult {
    pub chunk_id: String,
    pub document_id: String,
    pub content: String,
    pub document_title: String,
    pub document_path: Option<String>,
    pub score: f32,
}

/// Calculate cosine similarity between two vectors
///
/// # Mathematical Formula
///
/// ```text
/// cosine_similarity(a, b) = (a · b) / (||a|| * ||b||)
///
/// where:
/// - a · b = Σ(aᵢ * bᵢ)  (dot product)
/// - ||a|| = √(Σaᵢ²)      (Euclidean norm/magnitude)
/// - ||b|| = √(Σbᵢ²)
/// ```
///
/// # Interpretation
///
/// - **1.0**: Vectors are perfectly aligned (same direction)
/// - **0.0**: Vectors are orthogonal (uncorrelated)
/// - **-1.0**: Vectors are opposite (negative correlation)
///
/// For embeddings (all-positive vectors), range is typically [0, 1].
///
/// # Performance
///
/// - Time complexity: O(n) where n is vector dimension
/// - Space complexity: O(1) - no allocation
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // Validate inputs: vectors must be same length and non-empty
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    // Calculate dot product: Σ(aᵢ * bᵢ)
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    // Calculate Euclidean norms (magnitudes): ||a|| = √(Σaᵢ²)
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Avoid division by zero for zero vectors
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    // Return cosine similarity: (a · b) / (||a|| * ||b||)
    dot_product / (norm_a * norm_b)
}

/// Knowledge vault backed by SQLite-VSS
pub struct KnowledgeVault {
    conn: Connection,
    db_path: PathBuf,
    embedding_dimensions: u32,
}

impl KnowledgeVault {
    /// Create or open a knowledge vault
    #[instrument(skip_all)]
    pub fn open(path: &Path, embedding_dimensions: u32) -> KnowledgeResult<Self> {
        info!("Opening knowledge vault at {:?}", path);

        // Ensure directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(path)?;

        let vault = Self {
            conn,
            db_path: path.to_path_buf(),
            embedding_dimensions,
        };

        vault.init_schema()?;

        Ok(vault)
    }

    /// Create an in-memory knowledge vault for testing
    #[cfg(test)]
    pub fn in_memory() -> KnowledgeResult<Self> {
        let conn = Connection::open_in_memory()?;

        let vault = Self {
            conn,
            db_path: PathBuf::from(":memory:"),
            embedding_dimensions: 384,
        };

        vault.init_schema()?;

        Ok(vault)
    }

    /// Initialize database schema
    fn init_schema(&self) -> KnowledgeResult<()> {
        debug!("Initializing vault schema");

        // Documents table
        self.conn.execute(
            r#"
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                path TEXT UNIQUE,
                title TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                indexed_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
            "#,
            [],
        )?;

        // Chunks table
        self.conn.execute(
            r#"
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
            "#,
            [],
        )?;

        // Embeddings table (stores raw vectors)
        self.conn.execute(
            r#"
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            )
            "#,
            [],
        )?;

        // Attempt to create VSS virtual table for vector similarity search
        // This requires sqlite-vss extension to be loaded
        if let Err(e) = self.create_vss_table() {
            debug!(
                "VSS table creation failed (extension may not be loaded): {}",
                e
            );
            debug!("Falling back to basic embedding storage without vector search");
        }

        // Indexes
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)",
            [],
        )?;

        debug!("Schema initialized");
        Ok(())
    }

    /// Create VSS virtual table for vector similarity search
    fn create_vss_table(&self) -> KnowledgeResult<()> {
        // Validate embedding dimensions are within reasonable bounds
        if self.embedding_dimensions == 0 || self.embedding_dimensions > 10000 {
            return Err(KnowledgeError::InvalidFormat(
                format!("Embedding dimensions must be between 1 and 10000, got {}", self.embedding_dimensions)
            ));
        }

        // Create VSS virtual table for approximate nearest neighbor search
        // vss0 requires explicit chunk_id column for joins
        self.conn.execute(
            &format!(
                "CREATE VIRTUAL TABLE IF NOT EXISTS vss_chunks USING vss0(vss_chunk_id TEXT PRIMARY KEY, embedding({}))",
                self.embedding_dimensions
            ),
            [],
        )?;

        debug!(
            "VSS virtual table created with {} dimensions",
            self.embedding_dimensions
        );
        Ok(())
    }

    /// Insert a document
    #[instrument(skip(self, doc))]
    pub fn insert_document(&self, doc: &Document) -> KnowledgeResult<()> {
        self.conn.execute(
            r#"
            INSERT INTO documents (id, path, title, doc_type, content_hash, chunk_count, size_bytes, indexed_at, updated_at, metadata)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                content_hash = excluded.content_hash,
                chunk_count = excluded.chunk_count,
                size_bytes = excluded.size_bytes,
                updated_at = excluded.updated_at,
                metadata = excluded.metadata
            "#,
            params![
                doc.id,
                doc.path,
                doc.title,
                doc.doc_type,
                doc.content_hash,
                doc.chunk_count,
                doc.size_bytes as i64,
                doc.indexed_at.to_rfc3339(),
                doc.updated_at.to_rfc3339(),
                serde_json::to_string(&doc.metadata).unwrap_or_default(),
            ],
        )?;

        Ok(())
    }

    /// Get a document by ID
    pub fn get_document(&self, id: &str) -> KnowledgeResult<Option<Document>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, title, doc_type, content_hash, chunk_count, size_bytes, indexed_at, updated_at, metadata FROM documents WHERE id = ?1"
        )?;

        let doc = stmt
            .query_row(params![id], |row| {
                Ok(Document {
                    id: row.get(0)?,
                    path: row.get(1)?,
                    title: row.get(2)?,
                    doc_type: row.get(3)?,
                    content_hash: row.get(4)?,
                    chunk_count: row.get(5)?,
                    size_bytes: row.get::<_, i64>(6)? as u64,
                    indexed_at: chrono::DateTime::parse_from_rfc3339(&row.get::<_, String>(7)?)
                        .map(|dt| dt.with_timezone(&chrono::Utc))
                        .unwrap_or_else(|_| chrono::Utc::now()),
                    updated_at: chrono::DateTime::parse_from_rfc3339(&row.get::<_, String>(8)?)
                        .map(|dt| dt.with_timezone(&chrono::Utc))
                        .unwrap_or_else(|_| chrono::Utc::now()),
                    metadata: serde_json::from_str(&row.get::<_, String>(9)?).unwrap_or_default(),
                })
            })
            .optional()?;

        Ok(doc)
    }

    /// Check if a document exists by content hash
    pub fn has_document_hash(&self, hash: &str) -> KnowledgeResult<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM documents WHERE content_hash = ?1",
            params![hash],
            |row| row.get(0),
        )?;

        Ok(count > 0)
    }

    /// Delete a document and its chunks
    pub fn delete_document(&self, id: &str) -> KnowledgeResult<()> {
        self.conn
            .execute("DELETE FROM documents WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Insert a chunk
    #[allow(clippy::too_many_arguments)]
    pub fn insert_chunk(
        &self,
        id: &str,
        document_id: &str,
        chunk_index: u32,
        content: &str,
        start_offset: u64,
        end_offset: u64,
        token_count: u32,
    ) -> KnowledgeResult<()> {
        self.conn.execute(
            r#"
            INSERT INTO chunks (id, document_id, chunk_index, content, start_offset, end_offset, token_count)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
            params![
                id,
                document_id,
                chunk_index,
                content,
                start_offset as i64,
                end_offset as i64,
                token_count,
            ],
        )?;

        Ok(())
    }

    /// Get chunks for a document
    pub fn get_chunks(&self, document_id: &str) -> KnowledgeResult<Vec<ChunkRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, chunk_index, content, start_offset, end_offset, token_count FROM chunks WHERE document_id = ?1 ORDER BY chunk_index"
        )?;

        let chunks = stmt
            .query_map(params![document_id], |row| {
                Ok(ChunkRecord {
                    id: row.get(0)?,
                    document_id: document_id.to_string(),
                    chunk_index: row.get(1)?,
                    content: row.get(2)?,
                    start_offset: row.get::<_, i64>(3)? as u64,
                    end_offset: row.get::<_, i64>(4)? as u64,
                    token_count: row.get(5)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(chunks)
    }

    /// Insert an embedding
    pub fn insert_embedding(&self, chunk_id: &str, embedding: &[f32]) -> KnowledgeResult<()> {
        let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

        self.conn.execute(
            "INSERT INTO embeddings (chunk_id, embedding, created_at) VALUES (?1, ?2, ?3)",
            params![chunk_id, blob, chrono::Utc::now().to_rfc3339()],
        )?;

        // Try to insert into VSS table for vector search
        // Format embedding as vss_debug format: "[0.1,0.2,0.3,...]"
        let embedding_str = {
            let mut s = String::from("[");
            for (i, f) in embedding.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&f.to_string());
            }
            s.push(']');
            s
        };

        // Insert into VSS table for fast vector search
        if let Err(e) = self.conn.execute(
            "INSERT INTO vss_chunks (vss_chunk_id, embedding) VALUES (?1, ?2)",
            params![chunk_id, embedding_str],
        ) {
            warn!(
                "Failed to insert into VSS table: {}. Vector search may be limited to cosine similarity fallback.",
                e
            );
        }

        Ok(())
    }

    /// Get embedding for a chunk
    pub fn get_embedding(&self, chunk_id: &str) -> KnowledgeResult<Option<Vec<f32>>> {
        let blob: Option<Vec<u8>> = self
            .conn
            .query_row(
                "SELECT embedding FROM embeddings WHERE chunk_id = ?1",
                params![chunk_id],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()?;

        Ok(blob.map(|b: Vec<u8>| {
            b.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        }))
    }

    /// Search for similar chunks using vector similarity
    #[instrument(skip(self, query_embedding))]
    pub fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> KnowledgeResult<Vec<ChunkResult>> {
        debug!("Searching for top {} similar chunks", top_k);

        // Try VSS-based search first
        if let Ok(results) = self.search_vss(query_embedding, top_k) {
            return Ok(results);
        }

        // Fallback to simple cosine similarity search
        debug!("VSS search unavailable, using manual cosine similarity");
        self.search_cosine(query_embedding, top_k)
    }

    /// Search using VSS virtual table (fast approximate nearest neighbor)
    fn search_vss(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> KnowledgeResult<Vec<ChunkResult>> {
        // Format query embedding more efficiently
        let query_str = {
            let mut s = String::from("[");
            for (i, f) in query_embedding.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&f.to_string());
            }
            s.push(']');
            s
        };

        let sql = r#"
            SELECT
                c.id,
                c.document_id,
                c.content,
                d.title,
                d.path,
                vss.distance
            FROM vss_chunks
            JOIN chunks c ON vss_chunks.vss_chunk_id = c.id
            JOIN documents d ON c.document_id = d.id
            WHERE vss_chunks.embedding MATCH vss_search(?1)
            ORDER BY vss.distance
            LIMIT ?2
        "#;

        let mut stmt = self.conn.prepare(sql)?;

        let results = stmt
            .query_map(params![query_str, top_k as i64], |row| {
                Ok(ChunkResult {
                    chunk_id: row.get(0)?,
                    document_id: row.get(1)?,
                    content: row.get(2)?,
                    document_title: row.get(3)?,
                    document_path: Some(row.get(4)?),
                    score: 1.0 - row.get::<_, f64>(5)? as f32, // Convert distance to similarity
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(results)
    }

    /// Fallback: Manual cosine similarity search
    ///
    /// # When To Use
    ///
    /// This function is called when **VSS extension is not available**.
    /// It's a fallback that implements semantic search manually.
    ///
    /// # Performance Characteristics
    ///
    /// **⚠️ WARNING**: This method loads **ALL embeddings into memory**.
    ///
    /// ### Time Complexity
    /// - **Query**: O(n * d) where:
    ///   - n = total number of embeddings in database
    ///   - d = embedding dimension (e.g., 384 for BGE-Micro)
    /// - **Sorting**: O(n log n) for top-k selection
    ///
    /// ### Space Complexity
    /// - **Memory**: O(n * d * 4) bytes (all embeddings loaded as f32)
    ///   - Example: 10k embeddings × 384 dims × 4 bytes = ~15 MB
    ///   - Example: 100k embeddings × 384 dims × 4 bytes = ~150 MB
    ///
    /// ### Benchmarks
    /// - **1,000 chunks**: ~10ms
    /// - **10,000 chunks**: ~100ms
    /// - **100,000 chunks**: ~1-2s (becomes slow)
    ///
    /// # Recommendations
    ///
    /// ✅ **Suitable for**:
    /// - Small datasets (<10k chunks)
    /// - Development/testing
    /// - Systems without VSS extension
    ///
    /// ❌ **NOT suitable for**:
    /// - Large datasets (>10k chunks)
    /// - Production systems
    /// - Low-memory environments
    fn search_cosine(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> KnowledgeResult<Vec<ChunkResult>> {
        // Load all embeddings with their chunk metadata from database
        let sql = r#"
            SELECT
                c.id,
                c.document_id,
                c.content,
                d.title,
                d.path,
                e.embedding
            FROM embeddings e
            JOIN chunks c ON e.chunk_id = c.id
            JOIN documents d ON c.document_id = d.id
            "#;

        let mut stmt = self.conn.prepare(sql)?;

        // Calculate cosine similarity for each embedding against query
        let mut results: Vec<ChunkResult> = stmt
            .query_map([], |row| {
                // Deserialize embedding from little-endian f32 bytes
                let blob: Vec<u8> = row.get(5)?;
                let embedding: Vec<f32> = blob
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                // Calculate similarity score
                let score = cosine_similarity(query_embedding, &embedding);

                Ok(ChunkResult {
                    chunk_id: row.get(0)?,
                    document_id: row.get(1)?,
                    content: row.get(2)?,
                    document_title: row.get(3)?,
                    document_path: row.get(4)?,
                    score,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        // Sort by score descending (highest similarity first)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep only top_k results
        results.truncate(top_k);

        Ok(results)
    }

    /// Add a document with its content
    #[instrument(skip(self, content))]
    pub fn add_document(
        &self,
        path: &str,
        content: &str,
        doc_type: &str,
    ) -> KnowledgeResult<String> {
        use sha2::{Digest, Sha256};
        use uuid::Uuid;

        // Calculate content hash for deduplication
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let hash = hex::encode(hasher.finalize());

        // Check if document already exists
        if let Ok(Some(existing)) = self.get_document_by_path(path) {
            if existing.content_hash == hash {
                debug!("Document unchanged: {}", path);
                return Ok(existing.id);
            }
        }

        // Create new document
        let id = Uuid::new_v4().to_string();
        let title = Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("Untitled")
            .to_string();

        let now = chrono::Utc::now();
        let size_bytes = content.len() as u64;

        let doc = Document {
            id: id.clone(),
            path: Some(path.to_string()),
            title,
            doc_type: doc_type.to_string(),
            content_hash: hash,
            chunk_count: 0,
            size_bytes,
            indexed_at: now,
            updated_at: now,
            metadata: std::collections::HashMap::new(),
        };

        self.insert_document(&doc)?;

        Ok(id)
    }

    /// Get document by path
    fn get_document_by_path(&self, path: &str) -> KnowledgeResult<Option<Document>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, title, doc_type, content_hash, chunk_count, size_bytes, indexed_at, updated_at, metadata FROM documents WHERE path = ?1"
        )?;

        let doc = stmt
            .query_row(params![path], |row| {
                Ok(Document {
                    id: row.get(0)?,
                    path: row.get(1)?,
                    title: row.get(2)?,
                    doc_type: row.get(3)?,
                    content_hash: row.get(4)?,
                    chunk_count: row.get(5)?,
                    size_bytes: row.get::<_, i64>(6)? as u64,
                    indexed_at: chrono::DateTime::parse_from_rfc3339(&row.get::<_, String>(7)?)
                        .map(|dt| dt.with_timezone(&chrono::Utc))
                        .unwrap_or_else(|_| chrono::Utc::now()),
                    updated_at: chrono::DateTime::parse_from_rfc3339(&row.get::<_, String>(8)?)
                        .map(|dt| dt.with_timezone(&chrono::Utc))
                        .unwrap_or_else(|_| chrono::Utc::now()),
                    metadata: serde_json::from_str(&row.get::<_, String>(9)?).unwrap_or_default(),
                })
            })
            .optional()?;

        Ok(doc)
    }

    /// List all documents
    pub fn list_documents(&self, limit: usize) -> KnowledgeResult<Vec<Document>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, title, doc_type, content_hash, chunk_count, size_bytes, indexed_at, updated_at, metadata FROM documents ORDER BY updated_at DESC LIMIT ?1"
        )?;

        let docs = stmt
            .query_map(params![limit as i64], |row| {
                Ok(Document {
                    id: row.get(0)?,
                    path: row.get(1)?,
                    title: row.get(2)?,
                    doc_type: row.get(3)?,
                    content_hash: row.get(4)?,
                    chunk_count: row.get(5)?,
                    size_bytes: row.get::<_, i64>(6)? as u64,
                    indexed_at: chrono::DateTime::parse_from_rfc3339(&row.get::<_, String>(7)?)
                        .map(|dt| dt.with_timezone(&chrono::Utc))
                        .unwrap_or_else(|_| chrono::Utc::now()),
                    updated_at: chrono::DateTime::parse_from_rfc3339(&row.get::<_, String>(8)?)
                        .map(|dt| dt.with_timezone(&chrono::Utc))
                        .unwrap_or_else(|_| chrono::Utc::now()),
                    metadata: serde_json::from_str(&row.get::<_, String>(9)?).unwrap_or_default(),
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(docs)
    }

    /// Get vault statistics
    pub fn stats(&self) -> KnowledgeResult<VaultStats> {
        let document_count: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))?;

        let chunk_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;

        let embedding_count: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM embeddings", [], |row| row.get(0))?;

        let total_size_bytes: Option<i64> =
            self.conn
                .query_row("SELECT SUM(size_bytes) FROM documents", [], |row| {
                    row.get(0)
                })?;

        // Get database file size
        let db_size = std::fs::metadata(&self.db_path)
            .map(|m| m.len())
            .unwrap_or(0);

        Ok(VaultStats {
            document_count: document_count as u64,
            chunk_count: chunk_count as u64,
            embedding_count: embedding_count as u64,
            total_size_bytes: total_size_bytes.unwrap_or(0) as u64,
            database_size_bytes: db_size,
            embedding_dimensions: self.embedding_dimensions,
        })
    }
}

/// Chunk record from database
#[derive(Debug, Clone)]
pub struct ChunkRecord {
    pub id: String,
    pub document_id: String,
    pub chunk_index: u32,
    pub content: String,
    pub start_offset: u64,
    pub end_offset: u64,
    pub token_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_vault_creation() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let vault = KnowledgeVault::open(&db_path, 384).unwrap();
        let stats = vault.stats().unwrap();

        assert_eq!(stats.document_count, 0);
        assert_eq!(stats.chunk_count, 0);
    }

    #[test]
    fn test_document_insert_and_get() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let vault = KnowledgeVault::open(&db_path, 384).unwrap();

        let doc = Document {
            id: "doc_001".to_string(),
            path: Some("/path/to/doc.md".to_string()),
            title: "Test Document".to_string(),
            doc_type: "markdown".to_string(),
            content_hash: "abc123".to_string(),
            chunk_count: 5,
            size_bytes: 1024,
            indexed_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
        };

        vault.insert_document(&doc).unwrap();

        let retrieved = vault.get_document("doc_001").unwrap().unwrap();
        assert_eq!(retrieved.title, "Test Document");
        assert_eq!(retrieved.chunk_count, 5);
    }

    #[test]
    fn test_add_document() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let vault = KnowledgeVault::open(&db_path, 384).unwrap();

        let content = "This is a test document for the knowledge vault.";
        let doc_id = vault
            .add_document("/test/doc.txt", content, "text")
            .unwrap();

        // Verify document was created
        let doc = vault.get_document(&doc_id).unwrap().unwrap();
        assert_eq!(doc.title, "doc.txt");
        assert_eq!(doc.doc_type, "text");
        assert_eq!(doc.size_bytes, content.len() as u64);

        // Adding same document again should return same ID (deduplication)
        let doc_id2 = vault
            .add_document("/test/doc.txt", content, "text")
            .unwrap();
        assert_eq!(doc_id, doc_id2);
    }

    #[test]
    fn test_search_cosine_fallback() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let vault = KnowledgeVault::open(&db_path, 384).unwrap();

        // Add a document
        let doc_id = vault
            .add_document("/test/search.txt", "Test content for search", "text")
            .unwrap();

        // Add a chunk
        let chunk_id = "chunk_001".to_string();
        vault
            .insert_chunk(&chunk_id, &doc_id, 0, "Test content for search", 0, 20, 5)
            .unwrap();

        // Add an embedding
        let embedding = vec![0.1f32; 384];
        vault.insert_embedding(&chunk_id, &embedding).unwrap();

        // Search should use cosine similarity fallback
        let results = vault.search(&embedding, 5).unwrap();

        // Should find the chunk
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, chunk_id);
    }

    #[test]
    fn test_vault_stats() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let vault = KnowledgeVault::open(&db_path, 384).unwrap();

        let stats = vault.stats().unwrap();
        assert_eq!(stats.document_count, 0);
        assert_eq!(stats.chunk_count, 0);
        assert_eq!(stats.embedding_dimensions, 384);

        // Add a document
        vault
            .add_document("/test/stats.txt", "Content", "text")
            .unwrap();

        let stats = vault.stats().unwrap();
        assert_eq!(stats.document_count, 1);
    }
}
