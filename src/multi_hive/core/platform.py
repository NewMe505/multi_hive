"""
platform.py — the Windows/Linux compatibility seam.

The hive was written against Linux and reached for the stdlib `resource`
module in two places: peak-RSS sampling (metrics) and RLIMIT ceilings on the
reviewer sandbox (reviewer_node). `resource` does not exist on Windows, so
importing either module there was an immediate ImportError.

Everything platform-specific now lives behind these two functions.

Known asymmetry, deliberately not hidden
----------------------------------------
sandbox_preexec() returns None on Windows. preexec_fn requires fork(), which
Windows does not have, so the address-space / file-size / process-count caps
on generated code are NOT enforced there. Matching them would mean a Job
Object via ctypes — worth doing if untrusted code ever runs on Windows, but
it is a real feature, not a shim, so it is not silently faked here.
Generated code is still bounded by the subprocess timeout on both platforms.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Callable

IS_WINDOWS = os.name == "nt"

try:  # POSIX only
    import resource  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised on Windows
    resource = None  # type: ignore[assignment]


def peak_rss_mb() -> float:
    """
    Peak resident set size of the current process, in MB.

    High-water mark since process start on every platform, so it reads as a
    comparable upper bound across sprints within one run — not "peak for this
    sprint alone". Returns 0.0 rather than raising if the platform will not
    say; a metrics field is never worth crashing a sprint over.
    """
    if resource is not None:
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is kilobytes on Linux but bytes on macOS.
        if sys.platform == "darwin":
            return raw / (1024 * 1024)
        return raw / 1024

    if IS_WINDOWS:
        return _windows_peak_rss_mb()

    return 0.0


def _windows_peak_rss_mb() -> float:
    """
    PeakWorkingSetSize via psapi — keeps psutil out of the dependency list.

    argtypes/restype are declared rather than left to ctypes' defaults, and
    that is load-bearing: GetCurrentProcess() returns the pseudo-handle
    (HANDLE)-1, and ctypes' default int restype truncates it to a 32-bit -1.
    The subsequent call then fails and the whole shim silently reports 0 MB.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE

        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)

        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return 0.0

        return counters.PeakWorkingSetSize / (1024 * 1024)
    except Exception:
        return 0.0


def sandbox_preexec() -> Callable[[], None] | None:
    """
    SEC-H3: hard resource ceilings for the reviewer sandbox subprocess.

    Returns a preexec_fn on POSIX, or None on Windows (see module docstring).
    Pass the result straight to subprocess.Popen(preexec_fn=...).
    """
    if resource is None or IS_WINDOWS:
        return None

    def _apply_limits() -> None:
        try:
            # 2 GB address space — enough for OpenBLAS/SciPy to initialise.
            resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
            resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024**2, 10 * 1024**2))
            if hasattr(resource, "RLIMIT_NPROC"):
                resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except (ValueError, OSError):
            pass

    return _apply_limits
