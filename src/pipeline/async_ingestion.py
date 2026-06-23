import cv2
import torch
import queue
import threading
from typing import Generator, Tuple

class AsynchronousVideoIngestionPipeline:
    """
    Asynchronously streams video frames from disk, copying them to pinned memory,
    and pushing to GPU via non-blocking CUDA stream operations.
    """
    def __init__(self, video_path: str, batch_frames: int = 8, max_queue_size: int = 16):
        self.video_path = video_path
        self.batch_frames = batch_frames
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        
        # Define asynchronous CUDA stream if CUDA is active
        self.cuda_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def start(self) -> None:
        self.running = True
        self.worker = threading.Thread(target=self._reader_thread, daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.running = False
        if hasattr(self, 'worker'):
            self.worker.join(timeout=1.0)

    def _reader_thread(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            # Seed mock frames if path is empty
            self._fill_mock_data()
            cap.release()
            return

        frame_batch = []
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Normalize and structure frame: [Channels, Height, Width]
            frame_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).to(torch.float32) / 255.0
            
            # Place in pinned memory to optimize non-blocking PCIe transfers
            pinned_frame = frame_tensor.pin_memory()
            frame_batch.append(pinned_frame)

            if len(frame_batch) == self.batch_frames:
                # Stack batch sequence [C, Frames, H, W]
                stacked = torch.stack(frame_batch, dim=1)
                try:
                    self.frame_queue.put(stacked, timeout=2.0)
                except queue.Full:
                    pass
                frame_batch = []

        cap.release()
        self.running = False

    def _fill_mock_data(self) -> None:
        """
        Supplies mock tensor frames for sandbox test execution.
        """
        for _ in range(5):
            # Batch shape: [Channels=3, Frames=8, H=128, W=128]
            dummy = torch.randn(3, 8, 128, 128).pin_memory()
            self.frame_queue.put(dummy)
        self.running = False

    def stream_gpu_batches(self, device: str) -> Generator[torch.Tensor, None, None]:
        """
        Yields batches uploaded asynchronously to GPU device.
        """
        while self.running or not self.frame_queue.empty():
            try:
                cpu_batch = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self.cuda_stream and device.startswith("cuda"):
                # Asynchronous GPU transfer using the allocated CUDA stream
                with torch.cuda.stream(self.cuda_stream):
                    gpu_batch = cpu_batch.to(device, non_blocking=True)
                # Wait for transfer thread execution to sync safely
                self.cuda_stream.synchronize()
            else:
                gpu_batch = cpu_batch.to(device)

            yield gpu_batch
