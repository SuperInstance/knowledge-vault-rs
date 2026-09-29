//! Embeddings: generate vectors for text and compare them by cosine similarity.
//!
//! Run with: `cargo run --example embeddings`
//!
//! Uses `PlaceholderEmbedder`, a deterministic SHA256-derived embedder that
//! needs NO model file — good for tests and API exploration. For real
//! semantic embeddings, load a GGUF model with `LocalEmbedder::load(path)`;
//! the `EmbeddingProvider` trait is identical for both.

use knowledge_vault::embeddings::{
    cosine_similarity, normalize_embedding, EmbeddingProvider, PlaceholderEmbedder,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let embedder = PlaceholderEmbedder::new(384);

    // Single text -> vector. Real embedders are async because they may run
    // inference; the placeholder just hashes.
    let rust_doc = "Rust is a systems language focused on safety and performance.";
    let v1 = embedder.embed(rust_doc).await?;
    println!("embedded {} chars -> {} dims", rust_doc.len(), v1.len());

    // Batch embedding processes inputs with bounded concurrency (8 tasks).
    let batch = [
        "Rust is a systems language focused on safety and performance.",
        "The Eiffel Tower is located in Paris, France.",
        "Ownership rules prevent data races at compile time.",
        "Croissants are baked with laminated dough.",
    ];
    let vecs = embedder.embed_batch(&batch).await?;
    println!("batch embedded {} texts\n", vecs.len());

    // Cosine similarity between every pair. With the placeholder embedder the
    // vectors reflect surface-level token hashing, NOT semantics — so treat
    // the numbers as a plumbing demo, not a quality measure. The same code
    // with LocalEmbedder produces meaningfully similar scores.
    println!("pairwise cosine similarity:");
    print!("{:>18}", "");
    for j in 0..batch.len() {
        print!("  doc{j:<10}");
    }
    println!();
    for i in 0..batch.len() {
        print!("{:>18}", format!("doc{i}"));
        for j in 0..batch.len() {
            if i == j {
                print!("  {:<13}", "1.000");
            } else {
                print!("  {:<13.3}", cosine_similarity(&vecs[i], &vecs[j]));
            }
        }
        println!();
    }

    // Before indexing, embeddings are L2-normalized in place so that cosine
    // similarity reduces to a dot product.
    let mut v2 = vecs[1].clone();
    normalize_embedding(&mut v2);
    let norm: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("\nnormalized vector L2 norm = {norm:.4}");

    Ok(())
}
