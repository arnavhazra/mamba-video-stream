import torch

class MambaStreamingCache:
    """
    State cache tracking the recurrent latent state vectors (h_t) indefinitely.
    Ensures O(1) memory complexity during online sequence inference.
    """
    def __init__(self, batch_size: int, n_heads: int, head_dim: int, d_state: int, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_state = d_state
        self.device = device
        self.dtype = dtype

        # Initialize global persistent state tensor (filled with zeros)
        self.reset()

    def reset(self) -> None:
        """
        Clears/resets state matrices before new stream initialization.
        """
        self.state = torch.zeros(
            self.batch_size,
            self.n_heads,
            self.head_dim,
            self.d_state,
            device=self.device,
            dtype=self.dtype
        )

    def get_state(self) -> torch.Tensor:
        """
        Returns active cached hidden state matrix.
        """
        return self.state

    def update_state(self, new_state: torch.Tensor) -> None:
        """
        Overwrites active cached state with the updated vector step.
        """
        # Detach states to prevent gradient graph accumulations and potential memory leaks
        self.state = new_state.detach()
