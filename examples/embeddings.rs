//! Embeddings Example: Generate and compare text embeddings
//!
//! This example demonstrates:
//! 1. Creating embedders (BGE-Micro and placeholder)
//! 2. Generating embeddings for text
//! 3. Batch embedding for efficiency
//! 4. Calculating similarity scores
//! 5. Visualizing embedding relationships
//!
//! Run with:
//! ```bash
//! cargo run --example embeddings
//! ```

use knowledge_vault::{LocalEmbedder, PlaceholderEmbedder, EmbeddingProvider};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 knowledge-vault-rs - Embeddings Example\n");

    // Example 1: Create BGE-Micro embedder
    println!("1️⃣  Creating BGE-Micro Embedder (384 dimensions)\n");

    let embedder = LocalEmbedder::new(384)?;
    println!("✓ Embedder created");
    println!("  Dimensions: {}", embedder.dimensions());
    println!("  Type: BGE-Micro-v2 (or SHA256 placeholder if unavailable)\n");

    // Example 2: Generate single embedding
    println!("2️⃣  Generating Single Embedding\n");

    let text = "Machine learning is a subset of artificial intelligence";
    println!("  Text: \"{}\"\n", text);

    let start = std::time::Instant::now();
    let embedding = embedder.embed(text).await?;
    let elapsed = start.elapsed();

    println!("  Embedding generated in {:?}", elapsed);
    println!("  Shape: ({},)", embedding.len());
    println!("  First 10 values: {:.4?}", &embedding[..10]);
    println!("  Min: {:.4}, Max: {:.4}, Mean: {:.4}",
        embedding.iter().cloned().fold(f32::INFINITY, f32::min),
        embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        embedding.iter().sum::<f32>() / embedding.len() as f32
    );
    println!();

    // Example 3: Batch embedding (more efficient)
    println!("3️⃣  Batch Embedding\n");

    let texts = vec![
        "Rust is a systems programming language",
        "Python is a high-level programming language",
        "JavaScript is used for web development",
        "Machine learning requires training data",
    ];

    println!("  Embedding {} texts...\n", texts.len());

    let start = std::time::Instant::now();
    let embeddings = embedder.embed_batch(&texts.iter().map(|s| *s).collect::<Vec<_>>()).await?;
    let elapsed = start.elapsed();

    println!("  All embeddings generated in {:?}", elapsed);
    println!("  Average time per embedding: {:?}", elapsed / texts.len() as u32);
    println!("  Throughput: {:.2} texts/sec", texts.len() as f64 / elapsed.as_secs_f64());
    println!();

    // Example 4: Calculate similarity scores
    println!("4️⃣  Calculating Similarity Scores\n");

    let query = "programming languages";
    println!("  Query: \"{}\"\n", query);

    let query_emb = embedder.embed(query).await?;

    println!("  Similarities:");
    println!("  {:<40} | {}", "Text", "Score");
    println!("  {}", "-".repeat(55));

    for (i, text) in texts.iter().enumerate() {
        let similarity = cosine_similarity(&query_emb, &embeddings[i]);
        println!("  {:<40} | {:.4}", text, similarity);
    }
    println!();

    // Example 5: Semantic relationships
    println!("5️⃣  Exploring Semantic Relationships\n");

    let concepts = vec![
        "dog",
        "cat",
        "car",
        "bicycle",
        "computer",
        "software",
        "hardware",
        "animal",
    ];

    println!("  Computing embedding matrix...\n");

    // Generate embeddings for all concepts
    let concept_embeddings: Vec<Vec<f32>> = {
        let mut results = Vec::new();
        for concept in &concepts {
            results.push(embedder.embed(concept).await?);
        }
        results
    };

    // Create similarity matrix
    println!("  Similarity Matrix:");
    print!("          ");
    for concept in &concepts {
        print!("{:>10} ", &concept[..8.min(concept.len())]);
    }
    println!();

    for (i, concept_i) in concepts.iter().enumerate() {
        print!("{:>10} ", &concept_i[..8.min(concept_i.len())]);
        for (j, _) in concepts.iter().enumerate() {
            let similarity = cosine_similarity(&concept_embeddings[i], &concept_embeddings[j]);
            print!("{:>10.4} ", similarity);
        }
        println!();
    }
    println!();

    // Example 6: Find most similar pairs
    println!("6️⃣  Most Similar Concept Pairs\n");

    let mut pairs = Vec::new();
    for i in 0..concepts.len() {
        for j in (i + 1)..concepts.len() {
            let similarity = cosine_similarity(&concept_embeddings[i], &concept_embeddings[j]);
            pairs.push((concepts[i], concepts[j], similarity));
        }
    }

    // Sort by similarity descending
    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    println!("  Top 5 most similar pairs:");
    for (i, (c1, c2, score)) in pairs.iter().take(5).enumerate() {
        println!("  {}. {} ↔ {} : {:.4}", i + 1, c1, c2, score);
    }
    println!();

    // Example 7: Embedding properties
    println!("7️⃣  Embedding Properties\n");

    let text1 = "The cat sat on the mat";
    let text2 = "The cat sat on the mat"; // Identical
    let text3 = "The feline rested on the rug"; // Semantically similar
    let text4 = "The car drove down the street"; // Semantically different

    let emb1 = embedder.embed(text1).await?;
    let emb2 = embedder.embed(text2).await?;
    let emb3 = embedder.embed(text3).await?;
    let emb4 = embedder.embed(text4).await?;

    let sim_identical = cosine_similarity(&emb1, &emb2);
    let sim_similar = cosine_similarity(&emb1, &emb3);
    let sim_different = cosine_similarity(&emb1, &emb4);

    println!("  Identical texts:");
    println!("    \"{}\"", text1);
    println!("    \"{}\"", text2);
    println!("    Similarity: {:.4}\n", sim_identical);

    println!("  Semantically similar texts:");
    println!("    \"{}\"", text1);
    println!("    \"{}\"", text3);
    println!("    Similarity: {:.4}\n", sim_similar);

    println!("  Semantically different texts:");
    println!("    \"{}\"", text1);
    println!("    \"{}\"", text4);
    println!("    Similarity: {:.4}\n", sim_different);

    // Example 8: Placeholder embedder comparison
    println!("8️⃣  Placeholder Embedder (Deterministic)\n");

    let placeholder = PlaceholderEmbedder::new(384);

    let text_a = "hello world";
    let emb_a1 = placeholder.embed(text_a).await?;
    let emb_a2 = placeholder.embed(text_a).await?; // Same text

    let text_b = "hello universe"; // Different text
    let emb_b = placeholder.embed(text_b).await?;

    println!("  Placeholder embeddings are deterministic:");
    println!("    Same text, same embedding: {}", if emb_a1 == emb_a2 { "✓ Yes" } else { "✗ No" });
    println!("    Different text, different embedding: {}", if emb_a1 != emb_b { "✓ Yes" } else { "✗ No" });
    println!();

    let sim_placeholder = cosine_similarity(&emb_a1, &emb_b);
    println!("    Similarity of similar texts (placeholder): {:.4}", sim_placeholder);
    println!("    Note: Placeholder embeddings are NOT semantically meaningful!");
    println!("          Use only for development/testing.\n");

    // Example 9: Embedding dimensions comparison
    println!("9️⃣  Embedding Size Comparison\n");

    let text_sample = "This is a sample text for embedding";

    let dims_384 = LocalEmbedder::new(384)?;
    let emb_384 = dims_384.embed(text_sample).await?;

    println!("  Text: \"{}\"", text_sample);
    println!("  Length: {} characters\n", text_sample.len());

    println!("  384-dim embedding:");
    println!("    Float32 count: {}", emb_384.len());
    println!("    Memory size: {} bytes ({:.2} KB)", emb_384.len() * 4, emb_384.len() as f32 * 4.0 / 1024.0);
    println!();

    let comparison_1536 = emb_384.len() as f32 * 4.0 / (1536.0 * 4.0) * 100.0;
    println!("  Compared to OpenAI ada-002 (1536 dims):");
    println!("    Size ratio: {:.1}%", comparison_1536);
    println!("    Storage saved: {:.1}%", 100.0 - comparison_1536);
    println!();

    // Example 10: Practical use case - document ranking
    println!("🔟 Document Ranking Example\n");

    let query_doc = "How do I learn Rust programming?";
    let candidates = vec![
        "Rust is a systems programming language focused on safety",
        "Python is great for beginners and data science",
        "The Rust Book provides comprehensive tutorials",
        "Learning to code requires practice and patience",
        "Rust's ownership system prevents memory bugs",
    ];

    println!("  Query: \"{}\"\n", query_doc);

    let query_emb = embedder.embed(query_doc).await?;

    let mut rankings: Vec<(String, f32)> = Vec::new();
    for doc in &candidates {
        let doc_emb = embedder.embed(doc).await?;
        let score = cosine_similarity(&query_emb, &doc_emb);
        rankings.push((doc.to_string(), score));
    }

    rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("  Ranked documents:");
    for (i, (doc, score)) in rankings.iter().enumerate() {
        println!("  {}. (Score: {:.4})", i + 1, score);
        println!("     {}", doc);
    }
    println!();

    println!("✅ Example complete!");
    println!("\n💡 Key Takeaways:");
    println!("   - BGE-Micro generates 384-dimensional embeddings");
    println!("   - Batch embedding is more efficient than single embedding");
    println!("   - Similarity scores range from -1.0 to 1.0");
    println!("   - Scores closer to 1.0 indicate higher semantic similarity");
    println!("   - Placeholder embeddings are deterministic but NOT semantic");
    println!("   - Smaller dimensions (384 vs 1536) save storage and improve speed");

    Ok(())
}

/// Calculate cosine similarity between two vectors
///
/// Formula: (a · b) / (||a|| * ||b||)
/// where:
///   a · b = dot product (sum of element-wise products)
///   ||a|| = Euclidean norm (sqrt of sum of squares)
///
/// Returns: Similarity score in range [-1.0, 1.0]
///   1.0  = Perfectly aligned (same direction)
///   0.0  = Orthogonal (uncorrelated)
///   -1.0 = Opposite directions
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // Validate: vectors must have same length
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    // Calculate dot product
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    // Calculate magnitudes (Euclidean norms)
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Avoid division by zero
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Calculate Euclidean distance between two vectors
///
/// Formula: sqrt(Σ(aᵢ - bᵢ)²)
///
/// Returns: Distance in range [0.0, ∞)
///   0.0 = Identical vectors
#[allow(dead_code)]
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::INFINITY;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Calculate Manhattan distance between two vectors
///
/// Formula: Σ|aᵢ - bᵢ|
///
/// Returns: Distance in range [0.0, ∞)
///   0.0 = Identical vectors
#[allow(dead_code)]
fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::INFINITY;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum()
}
