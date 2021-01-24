try:
    import cupy as cp
    CUDA_AVA = True
except ImportError:
    CUDA_AVA = False