import pytest
import torch
from src.model.mamba2 import Mamba2SSDBlock
from src.tokenizer.video_tokenizer import SpatioTemporalVideoTokenizer
from src.cache.streaming_cache import MambaStreamingCache

def test_mamba2_block_dims():
    device = "cpu"
    d_model = 32
    d_state = 16
    d_inner = 64
    n_heads = 2
    
    model = Mamba2SSDBlock(d_model=d_model, d_state=d_state, d_inner=d_inner, n_heads=n_heads).to(device)
    model.eval()

    # Sequence shape: [Batch=2, Sequence=5, d_model=32]
    x = torch.randn(2, 5, d_model, device=device)
    
    with torch.no_grad():
        out, state = model(x)
        
    assert out.shape == x.shape
    assert state.shape == (2, n_heads, d_inner // n_heads, d_state)

def test_video_tokenizer_dims():
    device = "cpu"
    tokenizer = SpatioTemporalVideoTokenizer(in_channels=3, patch_size=(2, 4, 4), embed_dim=32).to(device)
    
    # Input video tensor: [Batch=1, Channels=3, Frames=4, Height=16, Width=16]
    x = torch.randn(1, 3, 4, 16, 16, device=device)
    out = tokenizer(x)
    
    # Output shape: [Batch=1, Sequence= (4/2) * (16/4) * (16/4) = 32, embed_dim=32]
    assert out.shape == (1, 32, 32)

def test_streaming_cache():
    cache = MambaStreamingCache(batch_size=1, n_heads=2, head_dim=16, d_state=8, device="cpu")
    s = cache.get_state()
    assert s.shape == (1, 2, 16, 8)
    
    new_s = torch.randn_like(s)
    cache.update_state(new_s)
    assert torch.equal(cache.get_state(), new_s)
