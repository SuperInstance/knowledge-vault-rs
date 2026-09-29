// Knowledge Vault Benchmarks
//
// Comprehensive benchmark suite for knowledge-vault-rs using Criterion.
// Measures performance of:
// - Document chunking
// - Embedding generation (placeholder embedder: measures plumbing, not model inference)
// - Vector search (10, 100, 1000 documents)
// - Index insertion speed
//
// NOTE (2026-09-30): rewritten against the current library API. The previous
// version still called KnowledgeVault::open_in_memory and the old 6-argument
// insert_chunk shape, neither of which exists anymore, so `cargo bench` could
// not compile. Vault setup now uses temporary on-disk databases.

use std::hint::black_box;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use knowledge_vault::{ChunkOptions, Chunker, KnowledgeVault};

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

/// Unique temp db path per call so benches never collide or reuse state.
fn temp_db() -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("kv-bench-{}-{n}.db", std::process::id()))
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
            b.iter(|| black_box(chunker.chunk(black_box(text)).unwrap()));
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
        (
            "small_chunks",
            ChunkOptions {
                chunk_size: 128,
                chunk_overlap: 25,
                min_chunk_size: 50,
                ..Default::default()
            },
        ),
        (
            "large_chunks",
            ChunkOptions {
                chunk_size: 1024,
                chunk_overlap: 100,
                min_chunk_size: 200,
                ..Default::default()
            },
        ),
        (
            "no_overlap",
            ChunkOptions {
                chunk_overlap: 0,
                ..Default::default()
            },
        ),
    ];

    for (name, opts) in options {
        group.bench_with_input(name, &opts, |b, opts| {
            let chunker = Chunker::with_options(opts.clone());
            b.iter(|| black_box(chunker.chunk(black_box(&text)).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark embedding generation (using placeholder embedder)
fn bench_embedding_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_generation");

    // Bind to locals first: the old code built temporaries inside the vec and
    // borrowed them in the same expression (does not compile).
    let short = "This is a short text.".to_string();
    let medium = generate_text(5, 10);
    let long = generate_text(20, 10);
    let texts = vec![("short", &short), ("medium", &medium), ("long", &long)];

    for (name, text) in texts {
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(name, &text, |b, _text| {
            // SHA256-derived placeholder vector: measures the hashing +
            // allocation path. Real model inference is benchmarked at the
            // model layer, not here.
            b.iter(|| {
                let dims = 384;
                let embedding: Vec<f32> = vec![0.0f32; dims];
                black_box(embedding)
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
            // Setup: temp-file vault with documents + chunk embeddings
            let vault = setup_vault_with_docs(count).unwrap();

            // Create query embedding
            let query_embedding = vec![0.1f32; 384];

            b.iter(|| {
                black_box(
                    vault
                        .search(black_box(&query_embedding), black_box(5))
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

/// Benchmark vault insertion speed (document + chunks per iteration)
fn bench_vault_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("vault_insertion");

    let batch_sizes = vec![1, 10, 50, 100];

    for size in batch_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                // Fresh vault per iteration: measures schema init + inserts.
                let db = temp_db();
                let vault = KnowledgeVault::open(&db, 384).unwrap();
                let chunker = Chunker::new();

                for i in 0..size {
                    let text = format!("{}\n\nunique marker {i}", generate_text(10, 10));
                    let doc_id = vault
                        .add_document(&format!("doc_{i}.md"), &text, "markdown")
                        .unwrap();

                    for (idx, chunk) in chunker.chunk(&text).unwrap().iter().enumerate() {
                        vault
                            .insert_chunk(
                                &format!("chunk_{i}_{idx}"),
                                &doc_id,
                                idx as u32,
                                &chunk.content,
                                chunk.start_offset,
                                chunk.end_offset,
                                chunk.token_count,
                            )
                            .unwrap();
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
                black_box(
                    vault
                        .search(black_box(&query_embedding), black_box(k))
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

/// Helper: temp-file vault with `count` distinct documents, each chunked and
/// stored with a 384-dim embedding so vector search has data to rank.
fn setup_vault_with_docs(count: usize) -> Result<KnowledgeVault, Box<dyn std::error::Error>> {
    let vault = KnowledgeVault::open(&temp_db(), 384)?;
    let chunker = Chunker::new();

    for i in 0..count {
        // Unique suffix per doc: add_document dedups by content hash, so
        // identical bodies would collapse to one row and skew the bench.
        let text = format!("{}\n\ndocument marker {i}", generate_text(10, 10));
        let doc_id = vault.add_document(&format!("doc_{i:03}.md"), &text, "markdown")?;

        for (idx, chunk) in chunker.chunk(&text)?.iter().enumerate() {
            let chunk_id = format!("chunk_{i}_{idx}");
            vault.insert_chunk(
                &chunk_id,
                &doc_id,
                idx as u32,
                &chunk.content,
                chunk.start_offset,
                chunk.end_offset,
                chunk.token_count,
            )?;
            let embedding = vec![0.1f32; 384];
            vault.insert_embedding(&chunk_id, &embedding)?;
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
