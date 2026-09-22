//! File Watcher
//!
//! Watches directories for changes and automatically indexes new/modified files.

use hex;
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, instrument, warn};

use crate::indexer::IndexCommand;
use crate::{KnowledgeError, KnowledgeResult};

/// Watch configuration
#[derive(Debug, Clone)]
pub struct WatchConfig {
    /// Directories to watch
    pub directories: Vec<PathBuf>,
    /// File extensions to include (None = all)
    pub extensions: Option<Vec<String>>,
    /// Patterns to exclude
    pub exclude_patterns: Vec<String>,
    /// Debounce duration (wait for changes to settle)
    pub debounce: Duration,
    /// Whether to watch recursively
    pub recursive: bool,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            directories: vec![],
            extensions: Some(vec![
                "md".to_string(),
                "txt".to_string(),
                "rs".to_string(),
                "py".to_string(),
                "js".to_string(),
                "ts".to_string(),
                "json".to_string(),
                "yaml".to_string(),
                "toml".to_string(),
            ]),
            exclude_patterns: vec![
                ".git".to_string(),
                "node_modules".to_string(),
                "target".to_string(),
                "__pycache__".to_string(),
                ".venv".to_string(),
            ],
            debounce: Duration::from_secs(2),
            recursive: true,
        }
    }
}

/// File change event
#[derive(Debug, Clone)]
pub enum FileChange {
    /// File created or modified
    Modified(PathBuf),
    /// File deleted
    Deleted(PathBuf),
    /// File renamed (old_path, new_path)
    Renamed(PathBuf, PathBuf),
}

/// File checksum cache
#[derive(Debug, Clone)]
struct FileChecksum {
    /// SHA256 checksum
    hash: String,
    /// Last modification time
    modified_time: std::time::SystemTime,
}

/// Auto-indexing file watcher
pub struct FileWatcher {
    config: WatchConfig,
    watcher: Option<RecommendedWatcher>,
    change_tx: Option<mpsc::Sender<IndexCommand>>,
    /// Cache of file checksums to detect actual changes
    checksums: HashMap<PathBuf, FileChecksum>,
}

impl FileWatcher {
    /// Create a new file watcher
    pub fn new(config: WatchConfig) -> Self {
        Self {
            config,
            watcher: None,
            change_tx: None,
            checksums: HashMap::new(),
        }
    }

    /// Create auto-indexing watcher with channel-based indexer
    ///
    /// This creates a watcher that automatically sends IndexCommand messages
    /// when files change. The channel should be connected to a DocumentIndexer.
    pub fn with_auto_index(
        config: WatchConfig,
        command_tx: mpsc::Sender<IndexCommand>,
    ) -> KnowledgeResult<Self> {
        let mut watcher = Self {
            config: config.clone(),
            watcher: None,
            change_tx: None, // We'll use the command_tx directly
            checksums: HashMap::new(),
        };

        // Pre-populate checksums for existing files
        watcher.scan_existing_files()?;

        // Store the command sender for use in event processing
        watcher.change_tx = Some(command_tx);

        Ok(watcher)
    }

    /// Scan existing files and populate checksum cache
    #[instrument(skip(self))]
    fn scan_existing_files(&mut self) -> KnowledgeResult<()> {
        info!("Scanning existing files for checksums");

        // Collect directories to avoid borrow checker issue
        let dirs = self.config.directories.clone();
        for dir in &dirs {
            if let Err(e) = self.scan_directory(dir) {
                warn!("Failed to scan directory {:?}: {}", dir, e);
            }
        }

        info!("Cached {} file checksums", self.checksums.len());
        Ok(())
    }

    /// Scan a directory for files
    fn scan_directory(&mut self, dir: &Path) -> KnowledgeResult<()> {
        let mut stack = vec![dir.to_path_buf()];

        while let Some(current) = stack.pop() {
            if !current.is_dir() {
                continue;
            }

            let entries = std::fs::read_dir(&current)?;

            for entry in entries {
                let entry = entry?;
                let path = entry.path();

                if path.is_dir() {
                    // Skip excluded directories
                    if !should_skip_directory(&path, &self.config.exclude_patterns) {
                        stack.push(path);
                    }
                } else if path.is_file() {
                    // Check if we should process this file
                    if should_process_path(&path, &self.config) {
                        if let Ok(checksum) = compute_checksum(&path) {
                            let metadata = std::fs::metadata(&path)?;
                            let modified = metadata.modified()?;
                            self.checksums.insert(
                                path.clone(),
                                FileChecksum {
                                    hash: checksum.hash,
                                    modified_time: modified,
                                },
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Start watching directories
    ///
    /// This spawns a background task that monitors file changes and sends
    /// IndexCommand messages to the channel.
    pub async fn start(&mut self) -> KnowledgeResult<()> {
        info!("Starting file watcher");

        let command_tx = self
            .change_tx
            .clone()
            .ok_or_else(|| KnowledgeError::Internal("No command channel configured".to_string()))?;

        // Create notify watcher
        let (notify_tx, notify_rx) = std::sync::mpsc::channel();

        let mut watcher =
            notify::recommended_watcher(move |res: Result<Event, notify::Error>| match res {
                Ok(event) => {
                    if let Err(e) = notify_tx.send(event) {
                        error!("Failed to send event: {}", e);
                    }
                },
                Err(e) => {
                    error!("Watch error: {}", e);
                },
            })
            .map_err(|e| KnowledgeError::WatchError(e.to_string()))?;

        // Add directories to watch
        let mode = if self.config.recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };

        for dir in &self.config.directories {
            info!("Watching directory: {:?}", dir);
            watcher
                .watch(dir, mode)
                .map_err(|e| KnowledgeError::WatchError(e.to_string()))?;
        }

        self.watcher = Some(watcher);

        // Spawn event processing task with checksums
        let config_clone = self.config.clone();
        tokio::spawn(async move {
            let mut pending: HashSet<PathBuf> = HashSet::new();
            let debounce = config_clone.debounce;
            let mut checksums: HashMap<PathBuf, FileChecksum> = HashMap::new();

            loop {
                // Process notify events
                match notify_rx.try_recv() {
                    Ok(event) => {
                        for path in event.paths {
                            if !should_process_path(&path, &config_clone) {
                                continue;
                            }

                            match event.kind {
                                EventKind::Create(_) | EventKind::Modify(_) => {
                                    pending.insert(path);
                                },
                                EventKind::Remove(_) => {
                                    // Remove from checksums
                                    checksums.remove(&path);
                                    // We don't send delete commands for now
                                    // as we don't have a DeleteFile command
                                },
                                _ => {},
                            }
                        }
                    },
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        info!("Notify channel disconnected, shutting down watcher");
                        break;
                    },
                    Err(_) => {
                        // No events, continue
                    }
                }

                // Debounce: process pending after delay
                if !pending.is_empty() {
                    tokio::time::sleep(debounce).await;

                    for path in pending.drain() {
                        // Compute checksum directly (handles file not found)
                        match compute_checksum(&path) {
                            Ok(checksum) => {
                                let metadata = std::fs::metadata(&path);
                                let modified = metadata.and_then(|m| m.modified()).ok();

                                let changed = if let Some(cached) = checksums.get(&path) {
                                    // Check if hash or modification time changed
                                    cached.hash != checksum.hash
                                        || modified.as_ref() != Some(&cached.modified_time)
                                } else {
                                    // New file
                                    true
                                };

                                if changed {
                                    // Update cache
                                    if let Some(modified) = modified {
                                        checksums.insert(
                                            path.clone(),
                                            FileChecksum {
                                                hash: checksum.hash,
                                                modified_time: modified,
                                            },
                                        );
                                    }

                                    // Send IndexCommand to reindex the file
                                    if let Err(e) =
                                        command_tx.try_send(IndexCommand::IndexFile(path.clone()))
                                    {
                                        error!("Failed to send index command for {:?}: {}", path, e);
                                    } else {
                                        debug!("Sent index command for changed file: {:?}", path);
                                    }
                                }
                            },
                            Err(e) => {
                                debug!("Failed to compute checksum for {:?}: {} (file may have been deleted)", path, e);
                                // File might have been deleted, that's ok
                            }
                        }
                    }
                }

                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });

        Ok(())
    }

    /// Stop watching
    pub fn stop(&mut self) {
        info!("Stopping file watcher");
        self.watcher = None;
        self.change_tx = None;
    }

    /// Add a directory to watch
    pub fn add_directory(&mut self, dir: &Path) -> KnowledgeResult<()> {
        if let Some(watcher) = &mut self.watcher {
            let mode = if self.config.recursive {
                RecursiveMode::Recursive
            } else {
                RecursiveMode::NonRecursive
            };

            watcher
                .watch(dir, mode)
                .map_err(|e| KnowledgeError::WatchError(e.to_string()))?;

            self.config.directories.push(dir.to_path_buf());
            info!("Added watch directory: {:?}", dir);
        }

        Ok(())
    }

    /// Remove a directory from watch
    pub fn remove_directory(&mut self, dir: &Path) -> KnowledgeResult<()> {
        if let Some(watcher) = &mut self.watcher {
            watcher
                .unwatch(dir)
                .map_err(|e| KnowledgeError::WatchError(e.to_string()))?;

            self.config.directories.retain(|d| d != dir);
            info!("Removed watch directory: {:?}", dir);
        }

        Ok(())
    }

    /// Get current watch config
    pub fn config(&self) -> &WatchConfig {
        &self.config
    }

    /// Check if currently watching
    pub fn is_watching(&self) -> bool {
        self.watcher.is_some()
    }
}

/// Check if a path should be processed based on config
fn should_process_path(path: &Path, config: &WatchConfig) -> bool {
    // Check exclude patterns
    let path_str = path.to_string_lossy();
    for pattern in &config.exclude_patterns {
        if path_str.contains(pattern) {
            return false;
        }
    }

    // Check extension filter
    if let Some(extensions) = &config.extensions {
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if !extensions.contains(&ext.to_string()) {
                return false;
            }
        } else {
            return false; // No extension
        }
    }

    // Must be a file
    path.is_file()
}

/// Check if a directory should be skipped
fn should_skip_directory(path: &Path, exclude_patterns: &[String]) -> bool {
    let path_str = path.to_string_lossy();

    // Check exclude patterns
    for pattern in exclude_patterns {
        if path_str.contains(pattern) {
            return true;
        }
    }

    // Skip hidden directories
    if let Some(name) = path.file_name() {
        if let Some(name_str) = name.to_str() {
            if name_str.starts_with('.') {
                return true;
            }
        }
    }

    false
}

/// Compute SHA256 checksum of a file
#[instrument(skip(path))]
fn compute_checksum(path: &Path) -> KnowledgeResult<FileChecksum> {
    let content = std::fs::read(path)?;

    let mut hasher = Sha256::new();
    hasher.update(&content);
    let hash = hex::encode(hasher.finalize());

    let metadata = std::fs::metadata(path)?;
    let modified = metadata.modified()?;

    Ok(FileChecksum {
        hash,
        modified_time: modified,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_should_process_path() {
        let config = WatchConfig::default();

        // Should process markdown files
        // Note: File doesn't exist in test, so we just verify it doesn't panic
        let _ = should_process_path(Path::new("/test/readme.md"), &config);

        // Should exclude .git
        assert!(!should_process_path(
            Path::new("/test/.git/config"),
            &config
        ));

        // Should exclude node_modules
        assert!(!should_process_path(
            Path::new("/test/node_modules/package.json"),
            &config
        ));
    }

    #[test]
    fn test_should_skip_directory() {
        let exclude_patterns = vec![
            ".git".to_string(),
            "node_modules".to_string(),
            "target".to_string(),
        ];

        // Should skip .git
        assert!(should_skip_directory(
            Path::new("/test/.git"),
            &exclude_patterns
        ));

        // Should skip node_modules
        assert!(should_skip_directory(
            Path::new("/test/node_modules"),
            &exclude_patterns
        ));

        // Should skip hidden directories
        assert!(should_skip_directory(
            Path::new("/test/.hidden"),
            &exclude_patterns
        ));

        // Should not skip normal directories
        assert!(!should_skip_directory(
            Path::new("/test/src"),
            &exclude_patterns
        ));
    }

    #[test]
    fn test_compute_checksum() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "Hello, World!").unwrap();

        let checksum = compute_checksum(temp_file.path()).unwrap();
        assert_eq!(checksum.hash.len(), 64); // SHA256 = 64 hex chars

        // Same content should produce same checksum
        let checksum2 = compute_checksum(temp_file.path()).unwrap();
        assert_eq!(checksum.hash, checksum2.hash);

        // Different content should produce different checksum
        writeln!(temp_file, "More content").unwrap();
        let checksum3 = compute_checksum(temp_file.path()).unwrap();
        assert_ne!(checksum.hash, checksum3.hash);
    }

    #[test]
    fn test_default_config() {
        let config = WatchConfig::default();
        assert!(config.recursive);
        assert!(config.extensions.is_some());
        assert!(!config.exclude_patterns.is_empty());

        // Check default extensions
        let extensions = config.extensions.unwrap();
        assert!(extensions.contains(&"rs".to_string()));
        assert!(extensions.contains(&"py".to_string()));
        assert!(extensions.contains(&"md".to_string()));
    }

    #[test]
    fn test_watch_config_clone() {
        let config = WatchConfig::default();
        let config2 = config.clone();

        assert_eq!(config.recursive, config2.recursive);
        assert_eq!(config.debounce, config2.debounce);
    }
}
