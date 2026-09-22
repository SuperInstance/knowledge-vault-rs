// Knowledge Vault Benchmarks
//
// Comprehensive benchmark suite for knowledge-vault-rs using Criterion.
// Measures performance of:
// - Document chunking
// - Embedding generation
// - Vector search (10, 100, 1000 documents)
// - Index insertion speed

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use knowledge_vault::{Chunker, ChunkOptions, KnowledgeVault, LocalEmbedder};
use std::time::Duration;

/// Generate sample text for benchmarking
fn generate_text(paragraphs: usize, sentences_per_paragraph: usize) -> String {
    let mut text = String::new();
    let sentence_templates = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Knowledge management systems require efficient data structures.",
        "Vector embeddings enable semantic search capabilities.",
        "Rust provides memory safety without garbage collection.",
        "Document chunking improves retrieval augmented generation.",
        "SQLite with VSS extension provides efficient vector similarity search.",
        "Async programming models enable concurrent I/O operations.",
        "File watching systems can automatically update knowledge bases.",
        "Embedding models transform text into high-dimensional vectors.",
        "Cosine similarity measures the angle between vector representations.",
    ];

    for p in 0..paragraphs {
        if p > 0 {
            text.push_str("\n\n");
        }
        for s in 0..sentences_per_paragraph {
            let template = sentence_templates[s % sentence_templates.len()];
            text.push_str(template);
            if s < sentences_per_paragraph - 1 {
                text.push(' ');
            }
        }
    }

    text
}

/// Benchmark document chunking with different sizes
fn bench_chunking(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunking");

    let sizes = vec![100, 500, 1000, 5000, 10000];

    for size in sizes {
        let text = generate_text(size / 20, 20); // ~size chars
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &text, |b, text| {
            let chunker = Chunker::new();
            b.iter(|| {
                black_box(chunker.chunk(black_box(text)).unwrap())
            });
        });
    }

    group.finish();
}

/// Benchmark chunking with different options
fn bench_chunking_options(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunking_options");

    let text = generate_text(100, 10); // Medium document

    let options = vec![
        ("default", ChunkOptions::default()),
        ("small_chunks", ChunkOptions {
            chunk_size: 128,
            chunk_overlap: 25,
            min_chunk_size: 50,
            ..Default::default()
        }),
        ("large_chunks", ChunkOptions {
            chunk_size: 1024,
            chunk_overlap: 100,
            min_chunk_size: 200,
            ..Default::default()
        }),
        ("no_overlap", ChunkOptions {
            chunk_overlap: 0,
            ..Default::default()
        }),
    ];

    for (name, opts) in options {
        group.bench_with_input(name, &opts, |b, opts| {
            let chunker = Chunker::with_options(opts.clone());
            b.iter(|| {
                black_box(chunker.chunk(black_box(&text)).unwrap())
            });
        });
    }

    group.finish();
}

/// Benchmark embedding generation (using placeholder embedder)
fn bench_embedding_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_generation");

    // Test different text sizes
    let texts = vec![
        ("short", "This is a short text."),
        ("medium", &generate_text(5, 10)),
        ("long", &generate_text(20, 10)),
    ];

    for (name, text) in texts {
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(name, text, |b, text| {
            // Note: Using placeholder embedder for benchmarking
            // Real embedder would require actual model files
            b.iter(|| {
                // Simulate embedding generation with placeholder
                let dims = 384;
                let _embedding = vec![0.0f32; dims];
                black_box(&_embedding)
            });
        });
    }

    group.finish();
}

/// Benchmark vector search with different document counts
fn bench_vector_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search");
    group.measurement_time(Duration::from_secs(30));

    let doc_counts = vec![10, 100, 1000];

    for count in doc_counts {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            // Setup: Create in-memory vault and insert documents
            let vault = setup_vault_with_docs(count).unwrap();

            // Create query embedding
            let query_embedding = vec![0.1f32; 384];

            b.iter(|| {
                black_box(vault.search(black_box(&query_embedding), black_box(5)).unwrap())
            });
        });
    }

    group.finish();
}

/// Benchmark vault insertion speed
fn bench_vault_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("vault_insertion");

    let batch_sizes = vec![1, 10, 50, 100];

    for size in batch_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let vault = KnowledgeVault::open_in_memory(384).unwrap();

                for i in 0..size {
                    let text = generate_text(10, 10);
                    let hash = format!("hash_{}", i);

                    // Insert document
                    vault.insert_document(
                        &hash,
                        &format!("doc_{}.md", i),
                        &text,
                        "markdown",
                    ).unwrap();

                    // Create and insert chunks
                    let chunker = Chunker::new();
                    let chunks = chunker.chunk(&text).unwrap();

                    for (idx, chunk) in chunks.iter().enumerate() {
                        let embedding = vec![0.0f32; 384]; // Placeholder
                        vault.insert_chunk(
                            &hash,
                            idx as u32,
                            &chunk.content,
                            &chunk.start_offset,
                            &chunk.end_offset,
                            &embedding,
                        ).unwrap();
                    }
                }

                black_box(vault)
            });
        });
    }

    group.finish();
}

/// Benchmark search with different top-k values
fn bench_search_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_top_k");

    let vault = setup_vault_with_docs(1000).unwrap();
    let query_embedding = vec![0.1f32; 384];

    let k_values = vec![1, 5, 10, 20, 50];

    for k in k_values {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| {
                black_box(vault.search(black_box(&query_embedding), black_box(k)).unwrap())
            });
        });
    }

    group.finish();
}

/// Helper function to set up a vault with a specific number of documents
fn setup_vault_with_docs(count: usize) -> Result<KnowledgeVault, Box<dyn std::error::Error>> {
    let vault = KnowledgeVault::open_in_memory(384)?;
    let chunker = Chunker::new();

    for i in 0..count {
        let text = generate_text(10, 10);
        let hash = format!("{:x}", md5::compute(&text));
        let filename = format!("doc_{:03}.md", i);

        // Insert document
        vault.insert_document(&hash, &filename, &text, "markdown")?;

        // Create and insert chunks
        let chunks = chunker.chunk(&text)?;

        for (idx, chunk) in chunks.iter().enumerate() {
            let embedding = vec![0.0f32; 384]; // Placeholder embedding
            vault.insert_chunk(
                &hash,
                idx as u32,
                &chunk.content,
                &chunk.start_offset,
                &chunk.end_offset,
                &embedding,
            )?;
        }
    }

    Ok(vault)
}

criterion_group!(
    benches,
    bench_chunking,
    bench_chunking_options,
    bench_embedding_generation,
    bench_vector_search,
    bench_vault_insertion,
    bench_search_top_k
);
criterion_main!(benches);

// MD5 implementation for content hashing
mod md5 {
    use std::fmt::Write;

    pub fn compute(data: &str) -> String {
        // Simple hash for benchmarking (not cryptographically secure)
        let mut hash: u64 = 5381;
        for byte in data.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }

        let mut result = String::new();
        write!(&mut result, "{:016x}", hash).unwrap();
        result
    }
}
