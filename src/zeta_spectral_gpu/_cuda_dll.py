"""Make the pip-wheel CUDA libraries importable on Windows.

``cupy-cuda12x[ctk]`` ships the CUDA libraries as separate ``nvidia-*-cu12``
wheels, each dropping its DLLs under ``site-packages/nvidia/<lib>/bin``. CuPy's
NVRTC path and the CUDA runtime load fine (the ``kernels/`` RawModule statistics
and ``cupy.searchsorted`` work without help), but ``cupy.linalg.eigh`` fails with
``DLL load failed while importing cusolver``: ``cusolver64_11.dll`` pulls in
cuBLAS / cuSPARSE / nvJitLink, which live in *sibling* ``nvidia/*/bin`` dirs that
are not on the Windows DLL search path. Adding those directories with
``os.add_dll_directory`` before the first eigh call resolves the dependency
chain.

No-op off Windows and when the wheels are absent (a system-CUDA or Linux
install), so it is always safe to call from a GPU import shim.
"""

from __future__ import annotations

import functools
import glob
import os


@functools.lru_cache(maxsize=1)
def add_cuda_dll_directories() -> int:
    """Add the ``nvidia/*/bin`` wheel dirs to the Windows DLL search path.

    Returns the number of directories added (0 off Windows, or when the
    ``nvidia-*-cu12`` wheels are not installed). Cached so repeated calls from
    the lazy ``cupy`` import are cheap and don't re-register the same dirs.
    """
    if os.name != "nt":
        return 0
    try:
        import nvidia  # PEP 420 namespace package from the nvidia-*-cu12 wheels
    except ImportError:
        return 0

    count = 0
    for base in nvidia.__path__:
        for bindir in sorted(glob.glob(os.path.join(base, "*", "bin"))):
            if os.path.isdir(bindir):
                os.add_dll_directory(bindir)
                count += 1
    return count
