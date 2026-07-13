"""
The regression detector has to be trustworthy, or nobody will believe it.

Two failure modes matter, and they pull in opposite directions: a detector that
cries wolf at thermal noise gets ignored, and a detector that shrugs at broken
code is worse than none.
"""
import pytest

from multi_hive.bench import history
from multi_hive.bench.history import Run, compare


def _run(subject: str, results: dict[str, bool], wall: float, ts: float, dirty: bool = False) -> Run:
    run = Run(
        suite="sprint",
        subject=subject,
        tasks=[
            {"task": name, "passed": passed, "wall_sec": wall / max(len(results), 1)}
            for name, passed in results.items()
        ],
    )
    run.timestamp = ts
    run.dirty = dirty
    run.commit = f"c{int(ts)}"
    return run


class TestRegressionDetection:
    def test_a_task_that_used_to_pass_and_now_fails_is_a_regression(self):
        base = _run("hive", {"semver": True, "lru": True}, wall=100, ts=1)
        now = _run("hive", {"semver": False, "lru": True}, wall=100, ts=2)

        cmp = compare(now, base)

        assert cmp.quality_regression
        assert cmp.is_regression
        assert cmp.regressed_tasks == ["semver"]
        assert cmp.quality_delta == -1

    def test_fixing_a_task_is_not_a_regression(self):
        base = _run("hive", {"semver": False}, wall=100, ts=1)
        now = _run("hive", {"semver": True}, wall=100, ts=2)

        cmp = compare(now, base)

        assert not cmp.is_regression
        assert cmp.fixed_tasks == ["semver"]
        assert cmp.quality_delta == 1

    def test_small_speed_wobble_is_weather_not_a_regression(self):
        # Local inference on a laptop is noisy: thermal throttling, another model
        # resident in VRAM, the OS indexing something. A detector that fires on
        # 10% will be ignored within a week.
        base = _run("hive", {"semver": True}, wall=100, ts=1)
        now = _run("hive", {"semver": True}, wall=110, ts=2)

        assert not compare(now, base).speed_regression

    def test_a_real_slowdown_is_caught(self):
        base = _run("hive", {"semver": True}, wall=100, ts=1)
        now = _run("hive", {"semver": True}, wall=140, ts=2)  # 1.4x

        cmp = compare(now, base)
        assert cmp.speed_regression and cmp.is_regression

    def test_quality_gain_does_not_excuse_a_slowdown(self):
        # Escalating everything to the strong model would look exactly like this.
        # It is a trade worth *knowing about*, not one to make silently.
        base = _run("hive", {"a": True, "b": False}, wall=100, ts=1)
        now = _run("hive", {"a": True, "b": True}, wall=300, ts=2)

        cmp = compare(now, base)
        assert cmp.quality_delta == 1
        assert cmp.speed_regression


class TestBaselineSelection:
    @pytest.fixture(autouse=True)
    def _isolate_history(self, tmp_path, monkeypatch):
        monkeypatch.setattr(history, "HISTORY_FILE", tmp_path / "bench_history.jsonl")

    def test_a_dirty_run_is_recorded_but_never_used_as_a_baseline(self):
        # A benchmark of uncommitted code cannot be reproduced, so it cannot be
        # the thing a future run is measured against.
        history.record(_run("hive", {"a": True}, wall=100, ts=1))
        history.record(_run("hive", {"a": True}, wall=100, ts=2, dirty=True))

        current = _run("hive", {"a": True}, wall=100, ts=3)
        baseline = history.baseline_for(current)

        assert baseline is not None
        assert baseline.commit == "c1", "a dirty run was used as a baseline"

    def test_baseline_is_the_most_recent_clean_run_of_the_same_subject(self):
        history.record(_run("hive", {"a": True}, wall=100, ts=1))
        history.record(_run("other", {"a": True}, wall=100, ts=2))
        history.record(_run("hive", {"a": True}, wall=100, ts=3))

        baseline = history.baseline_for(_run("hive", {"a": True}, wall=100, ts=4))
        assert baseline.commit == "c3"

    def test_no_baseline_on_the_very_first_run(self):
        assert history.baseline_for(_run("hive", {"a": True}, wall=100, ts=1)) is None

    def test_a_strict_aggregate_is_never_baselined_against_a_single_run(self):
        # The false alarm the audit's compare() note was about: a --repeat 3
        # aggregate scores a task passed only if it passed all three runs, while a
        # single run is a lucky sample. Comparing them turns a real 1/4 into a
        # "regression" from a fluke 3/4. Only like is compared with like.
        single = _run("hive", {"a": True}, wall=100, ts=1)
        single.repeat = 1
        history.record(single)

        aggregate = _run("hive", {"a": True}, wall=100, ts=2)
        aggregate.repeat = 3

        assert history.baseline_for(aggregate) is None, (
            "a strict x3 aggregate was baselined against a single-run sample"
        )

    def test_aggregates_baseline_against_prior_aggregates_of_the_same_repeat(self):
        for ts in (1, 2):
            run = _run("hive", {"a": True}, wall=100, ts=ts)
            run.repeat = 3
            history.record(run)

        current = _run("hive", {"a": True}, wall=100, ts=3)
        current.repeat = 3

        baseline = history.baseline_for(current)
        assert baseline is not None and baseline.commit == "c2"
