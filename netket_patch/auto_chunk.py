import gc
from math import ceil, log2


def AutoChunk(Driver):
    class _Driver(Driver):
        def __init__(self, *args, **kwargs):
            init_chunk_size_multiplier = kwargs.pop("init_chunk_size_multiplier", 1)
            self.min_chunk_size = kwargs.pop("min_chunk_size", 1)

            super().__init__(*args, **kwargs)

            chunk_size = self.state.chunk_size
            # If `state.chunk_size` is already set, we use that as the initial value
            if chunk_size is None:
                # Default initial value
                chunk_size = self.state.n_samples_per_rank * init_chunk_size_multiplier
            # Round up to a power of 2
            chunk_size = 2 ** ceil(log2(chunk_size))
            self.state.chunk_size = chunk_size
            print(f"Initialize chunk size to {self.state.chunk_size}")

        def _forward_and_backward(self):
            while True:
                try:
                    super()._forward_and_backward()
                except RuntimeError as e:
                    gc.collect()
                    chunk_size = self.state.chunk_size // 2
                    if chunk_size < self.min_chunk_size:
                        print(f"Minimum chunk size {self.min_chunk_size} reached")
                        raise e
                    print(f"Reduce chunk size to {chunk_size}")
                    # This driver modifies `state.chunk_size` in place
                    self.state.chunk_size = chunk_size

    return _Driver
