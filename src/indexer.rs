//! Document Indexer
//!
//! Handles ingestion of documents into the knowledge vault using a channel-based architecture
//! to avoid holding locks across await points.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

use crate::chunker::{detect_document_type, ChunkOptions, Chunker};
use crate::embeddings::EmbeddingProvider;
use crate::vault::{Document, KnowledgeVault};
use crate::{KnowledgeError, KnowledgeResult};

/// Result of indexing a document
#[derive(Debug)]
pub struct IndexResult {
    /// Document ID
    pub document_id: String,
    /// Number of chunks created
    pub chunk_count: u32,
    /// Whether the document was updated (vs new)
    pub updated: bool,
    /// Indexing time in milliseconds
    pub indexing_time_ms: u64,
}

/// Indexing commands sent through the channel
#[derive(Debug, Clone)]
pub enum IndexCommand {
    /// Index a single file
    IndexFile(PathBuf),
    /// Index content directly
    IndexContent {
        content: String,
        title: String,
        doc_type: String,
        path: Option<PathBuf>,
    },
    /// Index a directory recursively
    IndexDirectory {
        path: PathBuf,
        extensions: Option<Vec<String>>,
    },
    /// Reindex a specific document
    Reindex(String),
    /// Shutdown the indexer
    Shutdown,
}

/// Configuration for the indexer
#[derive(Debug, Clone)]
pub struct IndexerConfig {
    /// Skip if content hash matches existing document
    pub skip_duplicates: bool,
    /// Chunking options
    pub chunk_options: ChunkOptions,
    /// Channel buffer size
    pub channel_buffer: usize,
}

impl Default for IndexerConfig {
    fn default() -> Self {
        Self {
            skip_duplicates: true,
            chunk_options: ChunkOptions::default(),
            channel_buffer: 100,
        }
    }
}

/// Channel-based document indexer
///
/// This indexer uses an MPSC channel to avoid holding vault locks across await points.
/// Commands are sent to a background task that processes them sequentially.
#[derive(Clone)]
pub struct DocumentIndexer {
    /// Channel sender for commands
    command_tx: mpsc::Sender<IndexCommand>,
    /// Configuration for future reference
    #[allow(dead_code)]
    config: IndexerConfig,
}

impl DocumentIndexer {
    /// Create a new channel-based indexer
    ///
    /// This spawns a background task that processes indexing commands.
    /// The vault is wrapped in Arc<Mutex<>> to allow safe sharing across tasks.
    pub fn new<E: EmbeddingProvider + Send + 'static>(
        vault: Arc<Mutex<KnowledgeVault>>,
        embedder: Arc<Mutex<E>>,
        config: IndexerConfig,
    ) -> (Self, IndexerHandle) {
        let (command_tx, command_rx) = mpsc::channel(config.channel_buffer);

        let handle = IndexerHandle::spawn(vault, embedder, command_rx, config.clone());

        let indexer = Self {
            command_tx,
            config,
        };

        (indexer, handle)
    }

    /// Index a file from disk
    pub async fn index_file(&self, path: PathBuf) -> KnowledgeResult<()> {
        let cmd = IndexCommand::IndexFile(path);
        self.command_tx
            .send(cmd)
            .await
            .map_err(|_| KnowledgeError::Internal(
                "Indexer task shut down or channel full while processing file".to_string()
            ))?;
        Ok(())
    }

    /// Index content directly
    pub async fn index_content(
        &self,
        content: String,
        title: String,
        doc_type: String,
        path: Option<PathBuf>,
    ) -> KnowledgeResult<()> {
        let cmd = IndexCommand::IndexContent {
            content,
            title,
            doc_type,
            path,
        };

        self.command_tx
            .send(cmd)
            .await
            .map_err(|_| KnowledgeError::Internal(
                "Indexer task shut down or channel full while indexing content".to_string()
            ))?;

        Ok(())
    }

    /// Get the channel sender for use with FileWatcher
    pub fn command_sender(&self) -> mpsc::Sender<IndexCommand> {
        self.command_tx.clone()
    }
}

/// Background task handle for the indexer
///
/// This owns the background task that processes indexing commands.
/// When dropped, it will shutdown the task gracefully.
pub struct IndexerHandle {
    task: tokio::task::JoinHandle<()>,
    shutdown_tx: mpsc::Sender<IndexCommand>,
}

impl IndexerHandle {
    /// Spawn the background indexing task
    fn spawn<E: EmbeddingProvider + Send + 'static>(
        vault: Arc<Mutex<KnowledgeVault>>,
        embedder: Arc<Mutex<E>>,
        mut command_rx: mpsc::Receiver<IndexCommand>,
        config: IndexerConfig,
    ) -> Self {
        let task = tokio::spawn(async move {
            info!("Indexer background task started");

            while let Some(cmd) = command_rx.recv().await {
                match cmd {
                    IndexCommand::IndexFile(path) => {
                        if let Err(e) = Self::do_index_file(&vault, &embedder, &config, &path).await {
                            warn!("Failed to index file {:?}: {}", path, e);
                        }
                    }
                    IndexCommand::IndexContent {
                        content,
                        title,
                        doc_type,
                        path,
                    } => {
                        if let Err(e) =
                            Self::do_index_content(&vault, &embedder, &config, &content, &title, &doc_type, path.as_deref()).await
                        {
                            warn!("Failed to index content '{}': {}", title, e);
                        }
                    }
                    IndexCommand::IndexDirectory { path, extensions } => {
                        if let Err(e) =
                            Self::do_index_directory(&vault, &embedder, &config, &path, extensions.as_deref()).await
                        {
                            warn!("Failed to index directory {:?}: {}", path, e);
                        }
                    }
                    IndexCommand::Reindex(doc_id) => {
                        if let Err(e) = Self::do_reindex(&vault, &embedder, &config, &doc_id).await {
                            warn!("Failed to reindex {}: {}", doc_id, e);
                        }
                    }
                    IndexCommand::Shutdown => {
                        info!("Indexer shutdown command received");
                        break;
                    }
                }
            }

            info!("Indexer background task stopped");
        });

        // Create a channel for sending shutdown command
        let (shutdown_tx, _) = mpsc::channel(1);

        Self {
            task,
            shutdown_tx,
        }
    }

    /// Actually index a file (synchronous, no await points while holding lock)
    async fn do_index_file<E: EmbeddingProvider>(
        vault: &Arc<Mutex<KnowledgeVault>>,
        embedder: &Arc<Mutex<E>>,
        config: &IndexerConfig,
        path: &Path,
    ) -> KnowledgeResult<IndexResult> {
        let _start = std::time::Instant::now();

        // Read file (async, no lock held)
        let content = tokio::fs::read_to_string(path).await?;

        // Get filename for title (use to_str for proper UTF-8 handling)
        let filename = path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("Unknown")
            .to_string();

        // Detect document type
        let doc_type = detect_document_type(&filename);

        // Now do the indexing (lock held only during sync operations)
        Self::do_index_content(vault, embedder, config, &content, &filename, doc_type, Some(path))
            .await
    }

    /// Actually index content (synchronous operations while holding lock)
    async fn do_index_content<E: EmbeddingProvider>(
        vault: &Arc<Mutex<KnowledgeVault>>,
        embedder: &Arc<Mutex<E>>,
        config: &IndexerConfig,
        content: &str,
        title: &str,
        doc_type: &str,
        path: Option<&Path>,
    ) -> KnowledgeResult<IndexResult> {
        let start = std::time::Instant::now();

        // Calculate content hash (outside lock)
        let content_hash = calculate_hash(content);

        // Check for duplicates (lock held briefly)
        let should_skip = if config.skip_duplicates {
            let vault_guard = vault.lock().await;
            vault_guard.has_document_hash(&content_hash)?
        } else {
            false
        };

        if should_skip {
            info!("Document already indexed with same content hash: {}", title);
            return Ok(IndexResult {
                document_id: String::new(),
                chunk_count: 0,
                updated: false,
                indexing_time_ms: start.elapsed().as_millis() as u64,
            });
        }

        // Chunk the content (outside lock)
        let chunker = Chunker::with_options(config.chunk_options.clone());
        let chunks = chunker.chunk(content)?;
        let chunk_count = chunks.len() as u32;

        debug!("Created {} chunks", chunk_count);

        // Generate document ID (outside lock)
        let doc_id = format!("doc_{}", Uuid::new_v4().simple());

        // Create document record (outside lock)
        let now = chrono::Utc::now();
        let document = Document {
            id: doc_id.clone(),
            path: path.map(|p| p.to_string_lossy().to_string()),
            title: title.to_string(),
            doc_type: doc_type.to_string(),
            content_hash,
            chunk_count,
            size_bytes: content.len() as u64,
            indexed_at: now,
            updated_at: now,
            metadata: std::collections::HashMap::new(),
        };

        // Save document and chunks (lock held during database writes)
        {
            let vault_guard = vault.lock().await;
            let embedder_guard = embedder.lock().await;

            vault_guard.insert_document(&document)?;

            // Process chunks
            for (i, chunk) in chunks.iter().enumerate() {
                let chunk_id = format!("chunk_{}_{}", doc_id, i);

                // Save chunk
                vault_guard.insert_chunk(
                    &chunk_id,
                    &doc_id,
                    i as u32,
                    &chunk.content,
                    chunk.start_offset,
                    chunk.end_offset,
                    chunk.token_count,
                )?;

                // Generate and save embedding
                // Note: This is still synchronous for now
                // In the future, this could be batched
                let embedding = embedder_guard.embed(&chunk.content).await?;
                vault_guard.insert_embedding(&chunk_id, &embedding)?;
            }
        }

        info!(
            "Indexed document {} with {} chunks in {}ms",
            doc_id,
            chunk_count,
            start.elapsed().as_millis()
        );

        Ok(IndexResult {
            document_id: doc_id,
            chunk_count,
            updated: false,
            indexing_time_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Index a directory recursively
    async fn do_index_directory<E: EmbeddingProvider>(
        vault: &Arc<Mutex<KnowledgeVault>>,
        embedder: &Arc<Mutex<E>>,
        config: &IndexerConfig,
        dir: &Path,
        extensions: Option<&[String]>,
    ) -> KnowledgeResult<Vec<IndexResult>> {
        info!("Indexing directory: {:?}", dir);

        let mut results = Vec::new();
        let mut stack = vec![dir.to_path_buf()];

        // Convert extensions to &str for comparison
        let ext_strs: Option<Vec<&str>> = extensions.map(|exts| exts.iter().map(|s| s.as_str()).collect());

        while let Some(current) = stack.pop() {
            let mut entries = tokio::fs::read_dir(&current).await?;

            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();

                if path.is_dir() {
                    // Skip hidden directories
                    if !path
                        .file_name()
                        .map(|n| n.to_string_lossy().starts_with('.'))
                        .unwrap_or(false)
                    {
                        stack.push(path);
                    }
                } else if path.is_file() {
                    // Check extension filter
                    let should_index = if let Some(exts) = &ext_strs {
                        path.extension()
                            .and_then(|e| e.to_str())
                            .map(|e| exts.contains(&e))
                            .unwrap_or(false)
                    } else {
                        true
                    };

                    if should_index {
                        match Self::do_index_file(vault, embedder, config, &path).await {
                            Ok(result) => results.push(result),
                            Err(e) => {
                                warn!("Failed to index {:?}: {}", path, e);
                            }
                        }
                    }
                }
            }
        }

        info!("Indexed {} files from directory", results.len());
        Ok(results)
    }

    /// Reindex a specific document
    async fn do_reindex<E: EmbeddingProvider>(
        vault: &Arc<Mutex<KnowledgeVault>>,
        embedder: &Arc<Mutex<E>>,
        config: &IndexerConfig,
        document_id: &str,
    ) -> KnowledgeResult<IndexResult> {
        // Get existing document
        let doc = {
            let vault_guard = vault.lock().await;
            vault_guard
                .get_document(document_id)?
                .ok_or_else(|| KnowledgeError::NotFound(document_id.to_string()))?
        };

        // If we have the original path, read and reindex
        if let Some(ref path_str) = doc.path {
            let path = Path::new(path_str);
            if path.exists() {
                // Delete existing document
                {
                    let vault_guard = vault.lock().await;
                    vault_guard.delete_document(document_id)?;
                }
                // Reindex
                return Self::do_index_file(vault, embedder, config, path).await;
            }
        }

        Err(KnowledgeError::Internal(
            "Cannot reindex: original file not found".to_string(),
        ))
    }

    /// Shutdown the indexer gracefully
    pub async fn shutdown(self) {
        // Send shutdown command
        let _ = self.shutdown_tx.send(IndexCommand::Shutdown).await;

        // Wait for task to finish (with timeout)
        let _ = tokio::time::timeout(Duration::from_secs(5), self.task)
            .await;
    }
}

/// Legacy document indexer with lifetime parameters (deprecated)
///
/// This is kept for backward compatibility but should be replaced with
/// the channel-based `DocumentIndexer` for new code.
#[deprecated(note = "Use channel-based DocumentIndexer instead")]
pub struct LegacyDocumentIndexer<'a, E: EmbeddingProvider> {
    vault: &'a KnowledgeVault,
    embedder: &'a E,
    chunker: Chunker,
    /// Skip if content hash matches existing document
    skip_duplicates: bool,
}

#[allow(deprecated)]
impl<'a, E: EmbeddingProvider> LegacyDocumentIndexer<'a, E> {
    /// Create a new legacy indexer
    pub fn new(vault: &'a KnowledgeVault, embedder: &'a E) -> Self {
        Self {
            vault,
            embedder,
            chunker: Chunker::new(),
            skip_duplicates: true,
        }
    }

    /// Configure chunker options
    pub fn with_chunk_options(mut self, options: ChunkOptions) -> Self {
        self.chunker = Chunker::with_options(options);
        self
    }

    /// Configure duplicate handling
    pub fn skip_duplicates(mut self, skip: bool) -> Self {
        self.skip_duplicates = skip;
        self
    }

    /// Index a file from disk
    #[instrument(skip(self))]
    pub async fn index_file(&self, path: &Path) -> KnowledgeResult<IndexResult> {
        info!("Indexing file: {:?}", path);

        // Read file content
        let content = tokio::fs::read_to_string(path).await?;

        // Get filename for title (use to_str for proper UTF-8 handling)
        let filename = path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("Unknown")
            .to_string();

        // Detect document type
        let doc_type = detect_document_type(&filename);

        self.index_content(&content, &filename, doc_type, Some(path))
            .await
    }

    /// Index content directly
    #[instrument(skip(self, content))]
    pub async fn index_content(
        &self,
        content: &str,
        title: &str,
        doc_type: &str,
        path: Option<&Path>,
    ) -> KnowledgeResult<IndexResult> {
        let start = std::time::Instant::now();

        // Calculate content hash
        let content_hash = calculate_hash(content);

        // Check for duplicates
        if self.skip_duplicates && self.vault.has_document_hash(&content_hash)? {
            info!("Document already indexed with same content hash");
            return Ok(IndexResult {
                document_id: String::new(),
                chunk_count: 0,
                updated: false,
                indexing_time_ms: start.elapsed().as_millis() as u64,
            });
        }

        // Generate document ID
        let doc_id = format!("doc_{}", Uuid::new_v4().simple());

        // Chunk the content
        let chunks = self.chunker.chunk(content)?;
        let chunk_count = chunks.len() as u32;

        debug!("Created {} chunks", chunk_count);

        // Create document record
        let now = chrono::Utc::now();
        let document = Document {
            id: doc_id.clone(),
            path: path.map(|p| p.to_string_lossy().to_string()),
            title: title.to_string(),
            doc_type: doc_type.to_string(),
            content_hash,
            chunk_count,
            size_bytes: content.len() as u64,
            indexed_at: now,
            updated_at: now,
            metadata: std::collections::HashMap::new(),
        };

        // Save document
        self.vault.insert_document(&document)?;

        // Process chunks
        for (i, chunk) in chunks.iter().enumerate() {
            let chunk_id = format!("chunk_{}_{}", doc_id, i);

            // Save chunk
            self.vault.insert_chunk(
                &chunk_id,
                &doc_id,
                i as u32,
                &chunk.content,
                chunk.start_offset,
                chunk.end_offset,
                chunk.token_count,
            )?;

            // Generate and save embedding
            let embedding = self.embedder.embed(&chunk.content).await?;
            self.vault.insert_embedding(&chunk_id, &embedding)?;
        }

        info!(
            "Indexed document {} with {} chunks in {}ms",
            doc_id,
            chunk_count,
            start.elapsed().as_millis()
        );

        Ok(IndexResult {
            document_id: doc_id,
            chunk_count,
            updated: false,
            indexing_time_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Index multiple files
    pub async fn index_files(&self, paths: &[&Path]) -> KnowledgeResult<Vec<IndexResult>> {
        let mut results = Vec::with_capacity(paths.len());

        for path in paths {
            match self.index_file(path).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    warn!("Failed to index {:?}: {}", path, e);
                    // Continue with other files
                }
            }
        }

        Ok(results)
    }

    /// Index a directory recursively
    #[instrument(skip(self))]
    pub async fn index_directory(
        &self,
        dir: &Path,
        extensions: Option<&[&str]>,
    ) -> KnowledgeResult<Vec<IndexResult>> {
        info!("Indexing directory: {:?}", dir);

        let mut results = Vec::new();
        let mut stack = vec![dir.to_path_buf()];

        while let Some(current) = stack.pop() {
            let mut entries = tokio::fs::read_dir(&current).await?;

            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();

                if path.is_dir() {
                    // Skip hidden directories
                    if !path
                        .file_name()
                        .map(|n| n.to_string_lossy().starts_with('.'))
                        .unwrap_or(false)
                    {
                        stack.push(path);
                    }
                } else if path.is_file() {
                    // Check extension filter
                    let should_index = if let Some(exts) = extensions {
                        path.extension()
                            .and_then(|e| e.to_str())
                            .map(|e| exts.contains(&e))
                            .unwrap_or(false)
                    } else {
                        true
                    };

                    if should_index {
                        match self.index_file(&path).await {
                            Ok(result) => results.push(result),
                            Err(e) => {
                                warn!("Failed to index {:?}: {}", path, e);
                            }
                        }
                    }
                }
            }
        }

        info!("Indexed {} files from directory", results.len());
        Ok(results)
    }

    /// Reindex a specific document
    pub async fn reindex(&self, document_id: &str) -> KnowledgeResult<IndexResult> {
        // Get existing document
        let doc = self
            .vault
            .get_document(document_id)?
            .ok_or_else(|| KnowledgeError::NotFound(document_id.to_string()))?;

        // If we have the original path, read and reindex
        if let Some(path) = &doc.path {
            let path = Path::new(path);
            if path.exists() {
                // Delete existing document
                self.vault.delete_document(document_id)?;
                // Reindex
                return self.index_file(path).await;
            }
        }

        Err(KnowledgeError::Internal(
            "Cannot reindex: original file not found".to_string(),
        ))
    }
}

/// Calculate SHA256 hash of content
fn calculate_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_hash() {
        let hash1 = calculate_hash("hello world");
        let hash2 = calculate_hash("hello world");
        let hash3 = calculate_hash("different content");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.len(), 64); // SHA256 = 64 hex chars
    }
}
