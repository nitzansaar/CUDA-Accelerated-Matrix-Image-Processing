import ctypes
import numpy as np
import time

# Load your shared library
lib = ctypes.CDLL("./libconvolve.so")

# Define the function signature
lib.cuda_convolve.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),  # input image
    ctypes.POINTER(ctypes.c_ubyte),  # output image
    ctypes.POINTER(ctypes.c_float),  # filter
    ctypes.c_int,                    # M
    ctypes.c_int                     # N
]

def make_filter(N, mode="edge"):
    filt = np.zeros((N,N), dtype=np.float32)
    if mode == "blur":
        filt[:] = 1.0 / (N*N)
    else:
        filt[:] = -1.0
        filt[N//2, N//2] = N*N - 1
    return filt

def cuda_convolve(image: np.ndarray, filt: np.ndarray):
    M = image.shape[0]
    N = filt.shape[0]

    img_in = image.astype(np.uint8).copy()
    img_out = np.zeros_like(img_in)
    start = time.time()
    lib.cuda_convolve(
        img_in.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        img_out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        filt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(M),
        ctypes.c_int(N)
    )
    end = time.time()
    return img_out, end - start
