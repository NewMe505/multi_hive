"""
The sandbox executes untrusted, model-authored code. These tests assert it is
actually confined, rather than trusting that the ctypes incantation compiled.

Designing this test honestly matters more than it looks. A first version ran a
12.8 GB memory bomb and asserted "the child died" — and it passed, while the
sandbox was doing nothing at all. The child had died because Ollama was holding
18 GB and the machine genuinely ran out of RAM. A green test, measuring nothing.

So the bomb here allocates 3 GB: comfortably over the 2 GB ceiling, and just as
comfortably under the RAM of any machine this runs on. It therefore SURVIVES
unconfined and DIES confined, which is the only shape of test that can tell the
sandbox apart from the weather. `test_the_bomb_survives_unconfined` is the
control that keeps it honest.
"""
import subprocess
import sys
import textwrap

import pytest

from multi_hive.core.platform import (
    _MEMORY_LIMIT_BYTES,
    IS_WINDOWS,
    confine,
    release,
    sandbox_preexec,
)

# Over the 2 GB ceiling, well under any real machine's RAM.
BOMB = textwrap.dedent(
    """
    chunks = []
    for _ in range(96):                 # 96 x 32 MB = 3 GB
        chunks.append(bytearray(32 * 1024 * 1024))
    print("SURVIVED")
    """
)


def _spawn(code: str, confined: bool) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=sandbox_preexec() if confined else None,
    )
    job = confine(proc.pid) if confined else None
    try:
        out, _ = proc.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    finally:
        release(job)

    return subprocess.CompletedProcess(
        proc.args, proc.returncode, out.decode("utf-8", "replace"), ""
    )


def test_the_bomb_survives_unconfined():
    """
    The control. If this fails, the machine is simply out of memory and the
    confinement test below proves nothing — exactly the trap the first version
    of this file fell into.
    """
    result = _spawn(BOMB, confined=False)
    assert result.returncode == 0 and "SURVIVED" in result.stdout, (
        "the 3 GB bomb could not even run unconfined — this machine is short on "
        "RAM, so test_memory_bomb_is_killed proves nothing right now"
    )


def test_memory_bomb_is_killed_when_confined():
    """The same bomb, over the same ceiling, must not survive the sandbox."""
    result = _spawn(BOMB, confined=True)

    assert "SURVIVED" not in result.stdout, (
        f"generated code allocated past the "
        f"{_MEMORY_LIMIT_BYTES / 1024**3:.0f} GB ceiling — the sandbox is NOT enforcing"
    )
    assert result.returncode != 0


def test_ordinary_code_still_runs_inside_the_sandbox():
    # The ceiling must not be so tight that legitimate work dies; numpy-scale
    # allocations have to fit.
    result = _spawn("d = bytearray(64 * 1024 * 1024); print('alive', len(d))", confined=True)
    assert result.returncode == 0, result.stdout
    assert "alive" in result.stdout


@pytest.mark.skipif(not IS_WINDOWS, reason="Job Objects are the Windows mechanism")
def test_windows_job_reports_the_limit_it_was_given():
    """Ask the OS what it thinks the limit is, rather than trusting our own call."""
    import ctypes
    from ctypes import wintypes

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    job = confine(proc.pid)
    try:
        assert job, "no Job Object was created — the sandbox is not enforcing"

        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in "abcdef"]

        class BASIC(ctypes.Structure):
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

        class EXT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC),
                ("IoInfo", IO),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        info, returned = EXT(), wintypes.DWORD()
        ok = k32.QueryInformationJobObject(
            ctypes.c_void_p(job), 9, ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(returned)
        )
        assert ok, "could not read the job back from the OS"

        JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
        assert info.BasicLimitInformation.LimitFlags & JOB_OBJECT_LIMIT_PROCESS_MEMORY
        assert info.ProcessMemoryLimit == _MEMORY_LIMIT_BYTES
    finally:
        release(job)
        proc.kill()
        proc.wait(timeout=30)
