//! Chunking Example: Test different chunking strategies
//!
//! This example demonstrates various chunking strategies:
//! 1. Paragraph-aware chunking (default, recommended)
//! 2. Sentence-aware chunking (good for narrative text)
//! 3. Size-based chunking (fastest, but breaks context)
//! 4. Custom chunk sizes and overlaps
//!
//! Run with:
//! ```bash
//! cargo run --example chunking
//! ```

use knowledge_vault::{Chunker, ChunkOptions};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📦 knowledge-vault-rs - Chunking Example\n");

    // Sample text with mixed content
    let sample_text = r#"
# Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables
computers to learn from data without being explicitly programmed. It
focuses on developing algorithms that can improve their performance
through experience.

## Types of Machine Learning

There are three main types of machine learning: supervised learning,
unsupervised learning, and reinforcement learning. Each type has its
own strengths and is suited for different types of problems.

Supervised learning uses labeled data to train models. The algorithm
learns to map inputs to outputs based on example pairs. Common
applications include image classification and spam detection.

Unsupervised learning finds patterns in unlabeled data. It's used for
clustering, dimensionality reduction, and anomaly detection.

## Deep Learning

Deep learning is a type of machine learning that uses neural networks
with many layers. These networks can automatically learn hierarchical
representations of data. They've achieved remarkable success in
computer vision, natural language processing, and game playing.

## Practical Applications

Machine learning powers many technologies we use daily: recommendation
systems, search engines, voice assistants, and autonomous vehicles.
As data availability increases, ML applications continue to expand
into new domains.
"#;

    // Example 1: Paragraph-aware chunking (RECOMMENDED)
    println!("1️⃣  Paragraph-Aware Chunking (Default)\n");
    println!("   Strategy: Split on paragraph boundaries first, then by size");
    println!("   Best for: Technical documentation, structured text\n");

    let options = ChunkOptions {
        chunk_size: 128, // Smaller for demo
        chunk_overlap: 20,
        respect_paragraphs: true,
        respect_sentences: true,
        min_chunk_size: 50,
    };

    let chunker = Chunker::with_options(options.clone());
    let chunks = chunker.chunk(sample_text)?;

    println!("   Chunks created: {}", chunks.len());
    for (i, chunk) in chunks.iter().take(3).enumerate() {
        println!(
            "\n   Chunk {} ({} tokens, {} chars):\n   \"{}\"",
            i + 1,
            chunk.token_count,
            chunk.content.len(),
            chunk.content.chars().take(80).collect::<String>()
        );
    }
    println!("   ...\n");

    // Example 2: Sentence-aware chunking
    println!("2️⃣  Sentence-Aware Chunking\n");
    println!("   Strategy: Split on sentence boundaries");
    println!("   Best for: Narrative text, stories, articles\n");

    let options = ChunkOptions {
        chunk_size: 128,
        chunk_overlap: 20,
        respect_paragraphs: false, // Override paragraph splitting
        respect_sentences: true,
        min_chunk_size: 50,
    };

    let chunker = Chunker::with_options(options.clone());
    let chunks = chunker.chunk(sample_text)?;

    println!("   Chunks created: {}", chunks.len());
    for (i, chunk) in chunks.iter().take(3).enumerate() {
        println!(
            "\n   Chunk {} ({} tokens, {} chars):\n   \"{}\"",
            i + 1,
            chunk.token_count,
            chunk.content.len(),
            chunk.content.chars().take(80).collect::<String>()
        );
    }
    println!("   ...\n");

    // Example 3: Size-based chunking (FASTEST)
    println!("3️⃣  Size-Based Chunking (Fastest)\n");
    println!("   Strategy: Split by character count (~4 chars per token)");
    println!("   Best for: Large files where speed matters most");
    println!("   Warning: May break sentences and paragraphs\n");

    let options = ChunkOptions {
        chunk_size: 128,
        chunk_overlap: 0, // No overlap in size-based
        respect_paragraphs: false,
        respect_sentences: false,
        min_chunk_size: 50,
    };

    let chunker = Chunker::with_options(options.clone());
    let chunks = chunker.chunk(sample_text)?;

    println!("   Chunks created: {}", chunks.len());
    for (i, chunk) in chunks.iter().take(3).enumerate() {
        println!(
            "\n   Chunk {} ({} tokens, {} chars):\n   \"{}\"",
            i + 1,
            chunk.token_count,
            chunk.content.len(),
            chunk.content.chars().take(80).collect::<String>()
        );
    }
    println!("   ...\n");

    // Example 4: Large chunk with overlap
    println!("4️⃣  Large Chunks with Overlap\n");
    println!("   Strategy: 512-token chunks with 50-token overlap");
    println!("   Best for: Long documents where context matters\n");

    let options = ChunkOptions {
        chunk_size: 512,
        chunk_overlap: 50,
        respect_paragraphs: true,
        respect_sentences: true,
        min_chunk_size: 100,
    };

    let chunker = Chunker::with_options(options.clone());
    let chunks = chunker.chunk(sample_text)?;

    println!("   Chunks created: {}", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        println!(
            "\n   Chunk {} ({} tokens, {} chars):\n   \"{}\"",
            i + 1,
            chunk.token_count,
            chunk.content.len(),
            chunk.content.chars().take(100).collect::<String>()
        );

        // Show overlap visualization
        if i < chunks.len() - 1 {
            let next_chunk = &chunks[i + 1];
            let overlap_text = if chunk.content.len() > 100 && next_chunk.content.len() > 100 {
                // Check if there's overlap by comparing last 50 chars
                let chunk_end: String = chunk.content.chars().skip(
                    chunk.content.chars().count().saturating_sub(50)
                ).collect();
                let next_start: String = next_chunk.content.chars().take(50).collect();

                if chunk_end == next_start {
                    format!("\n   ⚡ Overlap detected: \"{}\"", chunk_end);
                } else {
                    "\n   ⚠ No exact overlap (semantic boundary)".to_string()
                }
            } else {
                "".to_string()
            };

            println!("{}", overlap_text);
        }
    }
    println!();

    // Example 5: Performance comparison
    println!("5️⃣  Performance Comparison\n");

    let strategies = vec![
        ("Paragraph-aware", ChunkOptions {
            chunk_size: 512,
            chunk_overlap: 50,
            respect_paragraphs: true,
            respect_sentences: true,
            min_chunk_size: 100,
        }),
        ("Sentence-aware", ChunkOptions {
            chunk_size: 512,
            chunk_overlap: 50,
            respect_paragraphs: false,
            respect_sentences: true,
            min_chunk_size: 100,
        }),
        ("Size-based", ChunkOptions {
            chunk_size: 512,
            chunk_overlap: 0,
            respect_paragraphs: false,
            respect_sentences: false,
            min_chunk_size: 100,
        }),
    ];

    println!("   {:<20} | {:>8} | {:>12} | {:>10}", "Strategy", "Chunks", "Time (μs)", "Avg Size");
    println!("   {}", "-".repeat(60));

    for (name, options) in strategies {
        let chunker = Chunker::with_options(options);

        let start = Instant::now();
        let chunks = chunker.chunk(sample_text)?;
        let elapsed = start.elapsed().as_micros();

        let avg_size: u32 = chunks.iter().map(|c| c.token_count).sum::<u32>() / chunks.len() as u32;

        println!("   {:<20} | {:>8} | {:>12} | {:>10}", name, chunks.len(), elapsed, avg_size);
    }
    println!();

    // Example 6: Edge cases
    println!("6️⃣  Edge Cases\n");

    // Empty text
    let empty_chunks = Chunker::new().chunk("")?;
    println!("   Empty text: {} chunks", empty_chunks.len());

    // Very short text
    let short_chunks = Chunker::new().chunk("Hello world")?;
    println!("   Short text (11 chars): {} chunks", short_chunks.len());
    if !short_chunks.is_empty() {
        println!("      First chunk: {} tokens", short_chunks[0].token_count);
    }

    // Text with only whitespace
    let ws_chunks = Chunker::new().chunk("   \n\n   \n\n")?;
    println!("   Whitespace only: {} chunks", ws_chunks.len());

    println!("\n✅ Example complete!");
    println!("\n💡 Recommendations:");
    println!("   - Use paragraph-aware chunking for technical docs");
    println!("   - Use sentence-aware chunking for narrative text");
    println!("   - Use size-based chunking only when speed is critical");
    println!("   - Always use overlap (10-20% of chunk_size) for context");

    Ok(())
}

/// Estimate tokens using the same algorithm as the chunker
fn estimate_tokens(text: &str) -> u32 {
    // Rough estimate: ~4 characters per token
    // Plus adjustment for unicode
    let char_count = text.chars().count() as u32;
    let unicode_multiplier = if text.contains(|c: char| c > '\u{7F}') { 1.5 } else { 1.0 };
    ((char_count as f32 * unicode_multiplier) / 4.0).ceil() as u32
}
