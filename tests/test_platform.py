"""The Windows/Linux seam. These are the calls that used to be `import resource`."""
import os

from multi_hive.core.platform import IS_WINDOWS, peak_rss_mb, sandbox_preexec


def test_peak_rss_is_reported_on_this_platform():
    rss = peak_rss_mb()
    assert isinstance(rss, float)
    # A live CPython process is never using zero memory. If this is 0.0 the
    # platform shim silently failed rather than measured.
    assert rss > 0.0


def test_sandbox_preexec_matches_platform_capability():
    preexec = sandbox_preexec()

    if IS_WINDOWS:
        # No fork() on Windows, so preexec_fn is unsupported. subprocess.Popen
        # rejects a non-None preexec_fn there outright.
        assert preexec is None
    else:
        assert callable(preexec)


def test_sandbox_env_has_what_python_needs_to_boot():
    from multi_hive.config import sandbox_env

    env = sandbox_env()
    assert env["PATH"]
    assert env["OMP_NUM_THREADS"] == "1"

    if os.name == "nt":
        # python.exe cannot load its DLLs without SystemRoot.
        assert env["SystemRoot"]
    else:
        assert env["HOME"]
