//! Document Chunker
//!
//! Splits documents into chunks suitable for embedding and retrieval.
//! Supports various chunking strategies.

use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

use crate::KnowledgeResult;

/// A chunk of text from a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Chunk content
    pub content: String,
    /// Start offset in original document
    pub start_offset: u64,
    /// End offset in original document
    pub end_offset: u64,
    /// Estimated token count
    pub token_count: u32,
    /// Chunk index within document
    pub index: u32,
}

/// Chunking options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkOptions {
    /// Target chunk size in tokens
    pub chunk_size: u32,
    /// Overlap between chunks in tokens
    pub chunk_overlap: u32,
    /// Minimum chunk size (don't create smaller chunks)
    pub min_chunk_size: u32,
    /// Respect sentence boundaries
    pub respect_sentences: bool,
    /// Respect paragraph boundaries
    pub respect_paragraphs: bool,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 50,
            min_chunk_size: 100,
            respect_sentences: true,
            respect_paragraphs: true,
        }
    }
}

/// Document chunker
pub struct Chunker {
    options: ChunkOptions,
}

impl Chunker {
    /// Create a new chunker with default options
    pub fn new() -> Self {
        Self {
            options: ChunkOptions::default(),
        }
    }

    /// Create a new chunker with custom options
    pub fn with_options(options: ChunkOptions) -> Self {
        Self { options }
    }

    /// Chunk a document
    #[instrument(skip(self, text))]
    pub fn chunk(&self, text: &str) -> KnowledgeResult<Vec<Chunk>> {
        debug!("Chunking {} characters", text.len());

        if text.is_empty() {
            return Ok(vec![]);
        }

        let chunks = if self.options.respect_paragraphs {
            self.chunk_by_paragraphs(text)
        } else if self.options.respect_sentences {
            self.chunk_by_sentences(text)
        } else {
            self.chunk_by_size(text)
        };

        debug!("Created {} chunks", chunks.len());
        Ok(chunks)
    }

    /// Chunk by paragraphs first, then by size
    fn chunk_by_paragraphs(&self, text: &str) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_start: u64 = 0;
        let mut chunk_index: u32 = 0;

        for paragraph in text.split("\n\n") {
            let para_tokens = estimate_tokens(paragraph);

            // If adding this paragraph exceeds chunk size, save current and start new
            if !current_chunk.is_empty()
                && estimate_tokens(&current_chunk) + para_tokens > self.options.chunk_size
            {
                chunks.push(self.create_chunk(&current_chunk, current_start, chunk_index));
                chunk_index += 1;

                // Handle overlap
                if self.options.chunk_overlap > 0 {
                    let overlap_text = self.get_overlap_text(&current_chunk);
                    current_chunk = overlap_text;
                    current_start = chunks.last()
                        .map(|c| c.end_offset - current_chunk.len() as u64)
                        .unwrap_or(0);
                } else {
                    current_chunk = String::new();
                    current_start = chunks.last().map(|c| c.end_offset).unwrap_or(0);
                }
            }

            // Add paragraph to current chunk
            if !current_chunk.is_empty() {
                current_chunk.push_str("\n\n");
            }
            current_chunk.push_str(paragraph);
        }

        // Don't forget the last chunk
        // If we have no chunks yet, always create one even if small
        // Otherwise, only create if it meets min_chunk_size
        if !current_chunk.is_empty()
            && (chunks.is_empty() || estimate_tokens(&current_chunk) >= self.options.min_chunk_size)
        {
            chunks.push(self.create_chunk(&current_chunk, current_start, chunk_index));
        }

        chunks
    }

    /// Chunk by sentences
    fn chunk_by_sentences(&self, text: &str) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_start: u64 = 0;
        let mut chunk_index: u32 = 0;

        for sentence in split_sentences(text) {
            let sentence_tokens = estimate_tokens(sentence);

            if !current_chunk.is_empty()
                && estimate_tokens(&current_chunk) + sentence_tokens > self.options.chunk_size
            {
                chunks.push(self.create_chunk(&current_chunk, current_start, chunk_index));
                chunk_index += 1;

                if self.options.chunk_overlap > 0 {
                    let overlap_text = self.get_overlap_text(&current_chunk);
                    current_chunk = overlap_text;
                    current_start = chunks.last()
                        .map(|c| c.end_offset - current_chunk.len() as u64)
                        .unwrap_or(0);
                } else {
                    current_chunk = String::new();
                    current_start = chunks.last().map(|c| c.end_offset).unwrap_or(0);
                }
            }

            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(sentence);
        }

        // Same logic as paragraph chunking
        if !current_chunk.is_empty()
            && (chunks.is_empty() || estimate_tokens(&current_chunk) >= self.options.min_chunk_size)
        {
            chunks.push(self.create_chunk(&current_chunk, current_start, chunk_index));
        }

        chunks
    }

    /// Simple size-based chunking
    fn chunk_by_size(&self, text: &str) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let chunk_chars = self.options.chunk_size as usize * 4; // ~4 chars per token
        let overlap_chars = self.options.chunk_overlap as usize * 4;

        let mut start = 0;
        let mut chunk_index = 0;

        while start < chars.len() {
            let end = (start + chunk_chars).min(chars.len());
            let content: String = chars[start..end].iter().collect();

            chunks.push(self.create_chunk(&content, start as u64, chunk_index));
            chunk_index += 1;

            start = if end == chars.len() {
                end
            } else {
                end - overlap_chars
            };
        }

        chunks
    }

    /// Create a chunk from content
    fn create_chunk(&self, content: &str, start_offset: u64, index: u32) -> Chunk {
        let content = content.trim().to_string();
        let token_count = estimate_tokens(&content);

        Chunk {
            end_offset: start_offset + content.len() as u64,
            start_offset,
            content,
            token_count,
            index,
        }
    }

    /// Get overlap text from the end of a chunk
    fn get_overlap_text(&self, chunk: &str) -> String {
        let target_tokens = self.options.chunk_overlap;
        let sentences: Vec<&str> = split_sentences(chunk).collect();

        let mut overlap = String::new();
        let mut tokens = 0;

        for sentence in sentences.into_iter().rev() {
            let sentence_tokens = estimate_tokens(sentence);
            if tokens + sentence_tokens > target_tokens && !overlap.is_empty() {
                break;
            }
            overlap = if overlap.is_empty() {
                sentence.to_string()
            } else {
                format!("{} {}", sentence, overlap)
            };
            tokens += sentence_tokens;
        }

        overlap
    }
}

impl Default for Chunker {
    fn default() -> Self {
        Self::new()
    }
}

/// Estimate token count (rough approximation: ~4 chars per token for English)
pub fn estimate_tokens(text: &str) -> u32 {
    let tokens = text.len() / 4;
    if tokens == 0 {
        return 1;
    }
    // Use saturating conversion to avoid overflow for very large strings
    u32::try_from(tokens).unwrap_or(u32::MAX)
}

/// Split text into sentences (simple implementation)
fn split_sentences(text: &str) -> impl Iterator<Item = &str> {
    text.split(['.', '!', '?'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
}

/// Determine document type from file extension
pub fn detect_document_type(filename: &str) -> &'static str {
    let ext = std::path::Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "md" | "markdown" => "markdown",
        "txt" => "text",
        "rs" => "rust",
        "py" => "python",
        "js" | "ts" => "javascript",
        "json" => "json",
        "yaml" | "yml" => "yaml",
        "toml" => "toml",
        "html" | "htm" => "html",
        "css" => "css",
        "sql" => "sql",
        "sh" | "bash" => "shell",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("hello world"), 2); // 11 chars / 4 ≈ 2
        assert_eq!(estimate_tokens(""), 1); // min 1
    }

    #[test]
    fn test_detect_document_type() {
        assert_eq!(detect_document_type("readme.md"), "markdown");
        assert_eq!(detect_document_type("main.rs"), "rust");
        assert_eq!(detect_document_type("script.py"), "python");
        assert_eq!(detect_document_type("unknown.xyz"), "unknown");
    }

    #[test]
    fn test_chunking() {
        let chunker = Chunker::new();
        let text = "This is the first paragraph.\n\nThis is the second paragraph.\n\nThis is the third paragraph.";

        let chunks = chunker.chunk(text).unwrap();
        assert!(!chunks.is_empty());

        // All text should be covered
        let total_content: String = chunks.iter().map(|c| c.content.as_str()).collect();
        assert!(total_content.contains("first"));
        assert!(total_content.contains("second"));
        assert!(total_content.contains("third"));
    }

    #[test]
    fn test_empty_text() {
        let chunker = Chunker::new();
        let chunks = chunker.chunk("").unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_small_text() {
        let chunker = Chunker::with_options(ChunkOptions {
            min_chunk_size: 5,
            ..Default::default()
        });

        let chunks = chunker.chunk("Hello world.").unwrap();
        // Should still create a chunk even if small
        assert!(!chunks.is_empty());
    }
}
