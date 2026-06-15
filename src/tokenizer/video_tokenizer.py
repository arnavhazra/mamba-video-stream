import torch
import torch.nn as nn

class SpatioTemporalVideoTokenizer(nn.Module):
    """
    Tokenizer that maps sequence blocks of raw video frames to token representations.
    Utilizes 3D spatial-temporal convolutions for unified patches representation.
    """
    def __init__(self, in_channels: int = 3, patch_size: tuple[int, int, int] = (2, 8, 8), embed_dim: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 3D Convolution layer representing: [Batch, Channels, Depth (Frames), Height, Width]
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: [Batch, Channels, Frames, Height, Width]
        Output shape: [Batch, Token_Sequence_Length, embed_dim]
        """
        # Execute 3D projection
        # Input tensor needs contiguous layout to facilitate GPU transfer optimizations
        x_proj = self.proj(x) # [Batch, embed_dim, F_patched, H_patched, W_patched]
        
        # Flatten spatial-temporal patches into sequence tokens
        # Output shape: [Batch, embed_dim, Sequence]
        x_flat = x_proj.flatten(2) 
        
        # Permute to map standard NLP sequence format: [Batch, Sequence, embed_dim]
        return x_flat.transpose(1, 2)
