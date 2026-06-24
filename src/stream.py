import torch
from model.mamba2 import Mamba2SSDBlock
from tokenizer.video_tokenizer import SpatioTemporalVideoTokenizer
from cache.streaming_cache import MambaStreamingCache
from pipeline.async_ingestion import AsynchronousVideoIngestionPipeline

def run_streaming_inference(video_path: str):
    """
    Orchestrates the real-time video streaming processing loops.
    Integrates async CUDA pipeline, 3D tokenizer, Mamba-2 SSD layer, and state caches.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Targeting inference device: {device}")

    # Hyperparameters
    embed_dim = 128
    d_state = 32
    n_heads = 4
    batch_size = 1

    # 1. Initialize tokenizer and model blocks
    tokenizer = SpatioTemporalVideoTokenizer(in_channels=3, patch_size=(2, 8, 8), embed_dim=embed_dim).to(device)
    model = Mamba2SSDBlock(d_model=embed_dim, d_state=d_state, d_inner=embed_dim, n_heads=n_heads).to(device)
    tokenizer.eval()
    model.eval()

    # 2. Initialize constant-memory recurrent state cache
    head_dim = embed_dim // n_heads
    cache = MambaStreamingCache(batch_size=batch_size, n_heads=n_heads, head_dim=head_dim, d_state=d_state, device=device)

    # 3. Setup non-blocking asynchronous CUDA ingestion pipeline
    # We batch frames in sets of 8 (corresponds to patch depth for 3D convolution)
    ingestion = AsynchronousVideoIngestionPipeline(video_path, batch_frames=8)
    ingestion.start()

    print("Running asynchronous spatio-temporal inference pipeline...")
    frame_idx = 0
    with torch.no_grad():
        for gpu_batch in ingestion.stream_gpu_batches(device):
            # gpu_batch: [Channels=3, Frames=8, Height, Width]
            # Add batch dimension: [Batch=1, Channels=3, Frames=8, Height, Width]
            gpu_batch = gpu_batch.unsqueeze(0)

            # Map raw frames to token embeddings: [Batch=1, Sequence=Length, embed_dim=128]
            tokens = tokenizer(gpu_batch)

            # Retrieve latent state vector from streaming cache
            prev_h = cache.get_state()

            # Execute forward inference pass
            out, next_h = model(tokens, prev_h)

            # Update cache context with new recurrent state matrix
            cache.update_state(next_h)

            frame_idx += 8
            print(f"Processed frame batch indices: {frame_idx - 8} - {frame_idx} | Output dimensions: {out.shape}")

    ingestion.stop()
    print("Inference loop processing completed successfully.")

if __name__ == "__main__":
    # Fallback to simulated sandbox ingestion if file is missing
    run_streaming_inference("dummy_video.mp4")
