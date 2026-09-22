//! Vector Search
//!
//! Provides similarity search over document embeddings.
//! Includes hybrid search combining vector similarity and keyword matching.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, instrument};

use crate::embeddings::{cosine_similarity, EmbeddingProvider};
use crate::vault::KnowledgeVault;
use crate::{KnowledgeError, KnowledgeResult};

/// Search options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchOptions {
    /// Maximum number of results
    pub limit: usize,
    /// Minimum similarity threshold (0.0-1.0)
    pub threshold: f32,
    /// Filter by document types
    pub doc_types: Option<Vec<String>>,
    /// Filter by document IDs
    pub doc_ids: Option<Vec<String>>,
    /// Include chunk content in results
    pub include_content: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: 10,
            threshold: 0.5,
            doc_types: None,
            doc_ids: None,
            include_content: true,
        }
    }
}

/// A search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Chunk ID
    pub chunk_id: String,
    /// Document ID
    pub document_id: String,
    /// Document title
    pub document_title: String,
    /// Similarity score (0.0-1.0)
    pub score: f32,
    /// Chunk content (if requested)
    pub content: Option<String>,
    /// Chunk index within document
    pub chunk_index: u32,
    /// Start offset in document
    pub start_offset: u64,
    /// End offset in document
    pub end_offset: u64,
}

/// Vector search engine
pub struct VectorSearch<'a> {
    vault: &'a KnowledgeVault,
}

impl<'a> VectorSearch<'a> {
    /// Create a new vector search instance
    pub fn new(vault: &'a KnowledgeVault) -> Self {
        Self { vault }
    }

    /// Search for similar chunks
    #[instrument(skip(self, query_embedding))]
    pub async fn search(
        &self,
        query_embedding: &[f32],
        options: &SearchOptions,
    ) -> KnowledgeResult<Vec<SearchResult>> {
        debug!(
            "Searching with {} dimensions, limit={}",
            query_embedding.len(),
            options.limit
        );

        // Get all documents (with optional filtering)
        let documents = self.vault.list_documents(1000)?;

        let mut results = Vec::new();

        for doc in documents {
            // Apply document type filter
            if let Some(ref types) = options.doc_types {
                if !types.contains(&doc.doc_type) {
                    continue;
                }
            }

            // Apply document ID filter
            if let Some(ref ids) = options.doc_ids {
                if !ids.contains(&doc.id) {
                    continue;
                }
            }

            // Get chunks for this document
            let chunks = self.vault.get_chunks(&doc.id)?;

            for chunk in chunks {
                // Get embedding for this chunk
                if let Some(embedding) = self.vault.get_embedding(&chunk.id)? {
                    let score = cosine_similarity(query_embedding, &embedding);

                    if score >= options.threshold {
                        results.push(SearchResult {
                            chunk_id: chunk.id.clone(),
                            document_id: doc.id.clone(),
                            document_title: doc.title.clone(),
                            score,
                            content: if options.include_content {
                                Some(chunk.content.clone())
                            } else {
                                None
                            },
                            chunk_index: chunk.chunk_index,
                            start_offset: chunk.start_offset,
                            end_offset: chunk.end_offset,
                        });
                    }
                }
            }
        }

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        results.truncate(options.limit);

        debug!("Found {} results above threshold", results.len());
        Ok(results)
    }

    /// Search with text query (generates embedding first)
    pub async fn search_text<E: EmbeddingProvider>(
        &self,
        query: &str,
        embedder: &E,
        options: &SearchOptions,
    ) -> KnowledgeResult<Vec<SearchResult>> {
        let query_embedding = embedder.embed(query).await?;
        self.search(&query_embedding, options).await
    }

    /// Get similar chunks to a specific chunk
    pub async fn find_similar(
        &self,
        chunk_id: &str,
        options: &SearchOptions,
    ) -> KnowledgeResult<Vec<SearchResult>> {
        // Get the embedding for the reference chunk
        let embedding = self
            .vault
            .get_embedding(chunk_id)?
            .ok_or_else(|| KnowledgeError::NotFound(format!("Chunk: {}", chunk_id)))?;

        // Exclude the reference chunk from results
        let modified_options = options.clone();

        let results = self.search(&embedding, &modified_options).await?;

        // Filter out the reference chunk
        Ok(results
            .into_iter()
            .filter(|r| r.chunk_id != chunk_id)
            .collect())
    }
}

/// Hybrid search combining vector and keyword search
pub struct HybridSearch<'a> {
    vector_search: VectorSearch<'a>,
    /// Weight for vector search (0.0-1.0)
    vector_weight: f32,
    /// Weight for keyword search (0.0-1.0)
    keyword_weight: f32,
}

impl<'a> HybridSearch<'a> {
    pub fn new(vault: &'a KnowledgeVault, vector_weight: f32, keyword_weight: f32) -> Self {
        Self {
            vector_search: VectorSearch::new(vault),
            vector_weight,
            keyword_weight,
        }
    }

    /// Hybrid search combining vector similarity and keyword matching
    ///
    /// Combines scores from both search methods using configured weights:
    /// final_score = (vector_score * vector_weight) + (keyword_score * keyword_weight)
    #[instrument(skip(self, query, embedder))]
    pub async fn search<E: EmbeddingProvider>(
        &self,
        query: &str,
        embedder: &E,
        options: &SearchOptions,
    ) -> KnowledgeResult<Vec<SearchResult>> {
        // Get vector search results
        let vector_results = self
            .vector_search
            .search_text(query, embedder, options)
            .await?;

        // Get keyword search results
        let keyword_results = self.keyword_search(query, options)?;

        // Combine scores from both methods
        let combined_results = self.combine_results(vector_results, keyword_results);

        Ok(combined_results)
    }

    /// Keyword-based search using BM25-like scoring
    fn keyword_search(&self, query: &str, options: &SearchOptions) -> KnowledgeResult<Vec<SearchResult>> {
        debug!("Performing keyword search for: {}", query);

        let query_terms: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .filter(|s| s.len() > 2) // Skip very short terms
            .map(|s| s.to_string())
            .collect();

        if query_terms.is_empty() {
            return Ok(vec![]);
        }

        // Get all documents
        let documents = self.vector_search.vault.list_documents(1000)?;

        let mut chunk_scores: HashMap<String, f32> = HashMap::new();
        let mut chunk_data: HashMap<String, SearchResult> = HashMap::new();

        for doc in documents {
            // Apply filters
            if let Some(ref types) = options.doc_types {
                if !types.contains(&doc.doc_type) {
                    continue;
                }
            }

            if let Some(ref ids) = options.doc_ids {
                if !ids.contains(&doc.id) {
                    continue;
                }
            }

            // Get chunks for this document
            let chunks = self.vector_search.vault.get_chunks(&doc.id)?;

            for chunk in chunks {
                let content_lower = chunk.content.to_lowercase();

                // Calculate BM25-like score
                let mut score = 0.0f32;
                for term in &query_terms {
                    // Count term frequency
                    let tf = content_lower.matches(term.as_str()).count() as f32;

                    if tf > 0.0 {
                        // Simple BM25: term frequency boost
                        score += (1.0 + tf).ln_1p();
                    }
                }

                if score > 0.0 {
                    // Normalize score to 0-1 range (approximate)
                    let normalized_score = (score / (query_terms.len() as f32)).min(1.0);

                    chunk_scores.insert(chunk.id.clone(), normalized_score);
                    chunk_data.insert(
                        chunk.id.clone(),
                        SearchResult {
                            chunk_id: chunk.id.clone(),
                            document_id: doc.id.clone(),
                            document_title: doc.title.clone(),
                            score: normalized_score,
                            content: if options.include_content {
                                Some(chunk.content.clone())
                            } else {
                                None
                            },
                            chunk_index: chunk.chunk_index,
                            start_offset: chunk.start_offset,
                            end_offset: chunk.end_offset,
                        },
                    );
                }
            }
        }

        // Convert to results
        let mut results: Vec<SearchResult> = chunk_scores
            .into_iter()
            .filter_map(|(id, score)| chunk_data.get(&id).map(|r| (id, r, score)))
            .map(|(id, result, score)| SearchResult {
                chunk_id: id,
                score,
                ..result.clone()
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        results.truncate(options.limit);

        debug!("Keyword search found {} results", results.len());
        Ok(results)
    }

    /// Combine vector and keyword search results
    fn combine_results(
        &self,
        vector_results: Vec<SearchResult>,
        keyword_results: Vec<SearchResult>,
    ) -> Vec<SearchResult> {
        let mut combined: HashMap<String, SearchResult> = HashMap::new();

        // Add vector results
        for result in vector_results {
            let id = result.chunk_id.clone();
            let score = result.score * self.vector_weight;
            combined.insert(id.clone(), result);
            if let Some(r) = combined.get_mut(&id) {
                r.score = score;
            }
        }

        // Add/update with keyword results
        for result in keyword_results {
            let id = result.chunk_id.clone();
            let keyword_score = result.score * self.keyword_weight;

            if let Some(existing) = combined.get_mut(&id) {
                // Combine scores
                existing.score += keyword_score;
            } else {
                // New result from keyword search only
                let mut new_result = result;
                new_result.score = keyword_score;
                combined.insert(id, new_result);
            }
        }

        // Convert to vec and sort
        let mut results: Vec<SearchResult> = combined.into_values().collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_options_default() {
        let options = SearchOptions::default();
        assert_eq!(options.limit, 10);
        assert_eq!(options.threshold, 0.5);
        assert!(options.include_content);
    }

    #[test]
    fn test_search_options_custom() {
        let options = SearchOptions {
            limit: 20,
            threshold: 0.7,
            doc_types: Some(vec!["code".to_string()]),
            doc_ids: None,
            include_content: false,
        };

        assert_eq!(options.limit, 20);
        assert_eq!(options.threshold, 0.7);
        assert!(options.doc_types.is_some());
        assert!(!options.include_content);
    }

    #[tokio::test]
    async fn test_hybrid_search_weights() {
        // Test that hybrid search properly weights results
        let vault = KnowledgeVault::in_memory().unwrap();
        let hybrid = HybridSearch::new(&vault, 0.7, 0.3);

        assert_eq!(hybrid.vector_weight, 0.7);
        assert_eq!(hybrid.keyword_weight, 0.3);
    }

    #[test]
    fn test_hybrid_search_equal_weights() {
        let vault = KnowledgeVault::in_memory().unwrap();
        let hybrid = HybridSearch::new(&vault, 0.5, 0.5);

        assert_eq!(hybrid.vector_weight, 0.5);
        assert_eq!(hybrid.keyword_weight, 0.5);
    }

    #[tokio::test]
    async fn test_keyword_search_empty_query() {
        let vault = KnowledgeVault::in_memory().unwrap();
        let hybrid = HybridSearch::new(&vault, 0.7, 0.3);

        let options = SearchOptions::default();
        let result = hybrid.keyword_search("", &options).unwrap();

        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_keyword_search_short_terms() {
        let vault = KnowledgeVault::in_memory().unwrap();
        let hybrid = HybridSearch::new(&vault, 0.7, 0.3);

        let options = SearchOptions::default();
        // Query with only short terms (< 3 chars) should return empty
        let result = hybrid.keyword_search("a b c", &options).unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn test_search_result_clone() {
        let result = SearchResult {
            chunk_id: "test-chunk".to_string(),
            document_id: "test-doc".to_string(),
            document_title: "Test Document".to_string(),
            score: 0.85,
            content: Some("Test content".to_string()),
            chunk_index: 0,
            start_offset: 0,
            end_offset: 12,
        };

        let cloned = result.clone();
        assert_eq!(cloned.chunk_id, result.chunk_id);
        assert_eq!(cloned.score, result.score);
    }
}
