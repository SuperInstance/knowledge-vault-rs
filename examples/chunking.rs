//! Text chunking: split documents into token-bounded, boundary-respecting pieces.
//!
//! Run with: `cargo run --example chunking`
//!
//! Chunking is the first stage of the indexing pipeline: each chunk later gets
//! embedded and stored as a searchable vector row.

use knowledge_vault::{ChunkOptions, Chunker};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Default options: 512-token chunks, 50-token overlap, respect sentence
    // and paragraph boundaries. Tune these per corpus (code wants smaller
    // chunks with syntactic boundaries; prose tolerates larger ones).
    let chunker = Chunker::with_options(ChunkOptions {
        chunk_size: 128,
        chunk_overlap: 20,
        min_chunk_size: 30,
        respect_sentences: true,
        respect_paragraphs: true,
    });

    let text = "\
Knowledge vaults work because retrieval quality tracks chunk quality.

A chunk that is too large dilutes the embedding signal: one vector ends up
averaging several unrelated ideas, and similarity search can no longer
distinguish them.

A chunk that is too small loses context: the vector is precise but the
matching text no longer makes sense outside the surrounding paragraph.

Overlap is the compromise. Each chunk repeats a few tokens from its
neighbor, so ideas that straddle a boundary are represented on both sides.

The chunker in this crate targets token budgets, then backs off to the
nearest sentence or paragraph boundary before finalizing a piece.";

    let chunks = chunker.chunk(text)?;
    println!("{} chars -> {} chunks\n", text.len(), chunks.len());

    for c in &chunks {
        println!(
            "chunk #{:<3} offsets {:>4}..{:<4} ~{} tokens",
            c.index, c.start_offset, c.end_offset, c.token_count
        );
        let preview: String = c.content.chars().take(72).collect();
        println!("      {preview}...");
    }

    // Chunk #0 starts at 0; with overlap, chunk #1 starts before chunk #0 ends.
    if chunks.len() >= 2 {
        let overlap = chunks[0].end_offset.saturating_sub(chunks[1].start_offset);
        println!("\nadjacent chunks overlap by {overlap} chars (configured: 20 tokens)");
    }

    Ok(())
}
