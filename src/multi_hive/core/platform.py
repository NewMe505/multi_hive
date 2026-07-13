"""
platform.py — the Windows/Linux compatibility seam.

The hive was written against Linux and reached for the stdlib `resource` module
in two places: peak-RSS sampling (metrics) and resource ceilings on the reviewer
sandbox. `resource` does not exist on Windows, so importing either module there
was an immediate ImportError.

Sandboxing generated code, on both platforms
--------------------------------------------
Generated code is untrusted, and it is *executed*. Each platform enforces the
ceilings through the only mechanism it has:

  POSIX     RLIMITs applied in a preexec_fn between fork() and exec().
  Windows   A Job Object the child is assigned to immediately after spawn.
            Windows has no fork(), so preexec_fn does not exist there.

The two are not identical, and the differences are stated rather than papered
over:

  address space / memory   both      2 GB   (RLIMIT_AS   / ProcessMemoryLimit)
  process count            both      64     (RLIMIT_NPROC / ActiveProcessLimit)
  file size                POSIX only 10 MB (RLIMIT_FSIZE; Job Objects have no
                                             equivalent — a runaway write on
                                             Windows is bounded only by disk)

On Windows there is also a sub-millisecond window between CreateProcess and
AssignProcessToJobObject during which the child is unconstrained. Closing it
properly needs CREATE_SUSPENDED plus a ResumeThread on a handle that
subprocess.Popen does not expose. In practice the child is a fresh python.exe
that spends its first ~50ms loading its own runtime, long before any generated
code executes. It is a real gap, and it is a small one.

The subprocess timeout bounds runtime on both platforms regardless.

What this does NOT do: confine the filesystem
---------------------------------------------
None of these ceilings restrict *where* the child writes. There is no chroot,
mount namespace, seccomp filter, or AppContainer anywhere here — only RLIMITs (on
POSIX) or a Job Object (on Windows), which bound memory, process count, and
per-file size, not filesystem paths. The reviewer runs the child with its cwd
inside the workspace, but cwd only changes relative-path resolution; generated
code that opens an absolute path outside the workspace is not stopped. This is a
resource sandbox, not a filesystem jail — treat the models it runs as
semi-trusted. See reviewer_node for the same caveat at the point of execution.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any

IS_WINDOWS = os.name == "nt"

try:  # POSIX only
    import resource  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised on Windows
    resource = None  # type: ignore[assignment]

# Ceilings applied to generated code. 2 GB is large enough for OpenBLAS/SciPy to
# initialise, which the DSP tasks need.
_MEMORY_LIMIT_BYTES = 2 * 1024**3
_FILE_SIZE_LIMIT_BYTES = 10 * 1024**2
_PROCESS_LIMIT = 64


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
    POSIX half of the sandbox: RLIMITs applied between fork() and exec().

    Returns a preexec_fn to hand straight to subprocess.Popen(preexec_fn=...),
    or None on Windows, where fork() does not exist. On Windows the ceilings are
    applied after the spawn instead — see confine().
    """
    if resource is None or IS_WINDOWS:
        return None

    def _apply_limits() -> None:
        try:
            resource.setrlimit(
                resource.RLIMIT_AS, (_MEMORY_LIMIT_BYTES, _MEMORY_LIMIT_BYTES)
            )
            resource.setrlimit(
                resource.RLIMIT_FSIZE, (_FILE_SIZE_LIMIT_BYTES, _FILE_SIZE_LIMIT_BYTES)
            )
            if hasattr(resource, "RLIMIT_NPROC"):
                resource.setrlimit(resource.RLIMIT_NPROC, (_PROCESS_LIMIT, _PROCESS_LIMIT))
        except (ValueError, OSError):
            pass

    return _apply_limits


def confine(pid: int) -> Any | None:
    """
    Windows half of the sandbox: assign a freshly-spawned child to a Job Object
    carrying the memory and process-count ceilings.

    Call immediately after subprocess.Popen. Returns an opaque handle that the
    caller MUST keep alive for the lifetime of the process and then pass to
    release() — the job is created with KILL_ON_JOB_CLOSE, so dropping the last
    reference to it terminates the child. That is the desired behaviour (no
    orphaned runaway code), but it means an early garbage collection would kill
    a healthy sandbox run.

    A no-op returning None on POSIX, where sandbox_preexec() already did the job
    before exec().
    """
    if not IS_WINDOWS:
        return None

    try:
        import ctypes
        from ctypes import wintypes

        JobObjectExtendedLimitInformation = 9
        JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
        JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

        PROCESS_SET_QUOTA = 0x0100
        PROCESS_TERMINATE = 0x0001

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.SetInformationJobObject.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]

        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        info.BasicLimitInformation.ActiveProcessLimit = _PROCESS_LIMIT
        info.ProcessMemoryLimit = _MEMORY_LIMIT_BYTES

        if not k32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
        ):
            k32.CloseHandle(job)
            return None

        handle = k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not handle:
            k32.CloseHandle(job)
            return None

        assigned = k32.AssignProcessToJobObject(job, handle)
        k32.CloseHandle(handle)

        if not assigned:
            k32.CloseHandle(job)
            return None

        return job
    except Exception:
        # A sandbox we could not build is worth reporting, never worth crashing
        # the sprint over. The subprocess timeout still bounds the child.
        return None


def release(job: Any | None) -> None:
    """Close a Job Object handle from confine(). Safe to call with None."""
    if job is None or not IS_WINDOWS:
        return
    try:
        import ctypes

        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(job)
    except Exception:
        pass
