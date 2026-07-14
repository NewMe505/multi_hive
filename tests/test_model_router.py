"""
The escalation ladder. This is the routing decision, so it gets real coverage.

The load-bearing property is the ratchet: a task escalates to the strong model
and stays there. The two models do not fit in 8 GB of VRAM together, so a tier
that could fall back mid-task would evict and reload the model on every attempt.
"""
import pytest

from multi_hive.core.model_router import FAST, STRONG, classify_complexity, select_tier


class TestClassifyComplexity:
    def test_boilerplate_is_trivial(self):
        assert classify_complexity("Write a function that adds two numbers") == "trivial"

    @pytest.mark.parametrize(
        "task",
        [
            "Implement a thread-safe LRU cache",
            "Refactor the parser into a state machine",
            "Optimise this to O(1) lookup",
            "Handle the race condition in the async writer",
        ],
    )
    def test_signals_of_real_difficulty_are_hard(self, task):
        assert classify_complexity(task) == "hard"

    def test_many_clauses_reads_as_moderate(self):
        task = (
            "Build a module that loads config, validates it, merges defaults, "
            "writes a log line, and exposes a getter"
        )
        assert classify_complexity(task) == "moderate"

    def test_missing_task_does_not_crash(self):
        # current_task is Optional throughout the graph.
        assert classify_complexity(None) == "moderate"
        assert classify_complexity("") == "moderate"


class TestSelectTier:
    def test_easy_work_starts_fast(self):
        assert select_tier("trivial", editor_retries=0) == FAST

    def test_hard_work_starts_strong(self):
        # Paying a doomed fast attempt AND the reload to escalate costs more
        # than starting on the model that can do the job.
        assert select_tier("hard", editor_retries=0) == STRONG

    def test_one_failure_escalates(self):
        # Retrying with the model that just failed re-buys the same failure.
        assert select_tier("trivial", editor_retries=0) == FAST
        assert select_tier("trivial", editor_retries=1) == STRONG

    def test_repeat_error_escalates_immediately(self):
        # The same error fingerprint twice means the model is fixing symptoms.
        # More attempts from it change nothing; a different model might.
        assert select_tier("trivial", editor_retries=0, repeat_error=True) == STRONG

    def test_force_tier_overrides_everything(self, monkeypatch):
        import multi_hive.core.model_router as router

        monkeypatch.setattr(router, "FORCE_TIER", FAST)
        assert router.select_tier("hard", editor_retries=9, repeat_error=True) == FAST

        monkeypatch.setattr(router, "FORCE_TIER", STRONG)
        assert router.select_tier("trivial", editor_retries=0) == STRONG


class TestLlmFactoryTiers:
    def test_purpose_and_tier_are_independent_cache_keys(self):
        from multi_hive.core import llm_factory

        # The reviewer prompt on the fast model and on the strong model are two
        # different clients; neither may evict the other.
        fast = llm_factory._resolve("reviewer", "fast")
        strong = llm_factory._resolve("reviewer", "strong")

        assert fast["model"] != strong["model"]
        assert fast["temperature"] == strong["temperature"] == 0.0
        assert fast["num_predict"] == strong["num_predict"] == 128

    def test_unknown_tier_is_rejected(self):
        from multi_hive.core import llm_factory

        with pytest.raises(ValueError, match="Unknown model tier"):
            llm_factory._resolve("editor", "gigantic")

    def test_unknown_purpose_is_rejected(self):
        from multi_hive.core import llm_factory

        with pytest.raises(ValueError, match="Unknown LLM purpose"):
            llm_factory._resolve("astrologer", "fast")


# ── HIVE_PLAN_TIER — who decides what the task IS ────────────────────────────


def test_plan_tier_defaults_to_the_router():
    """Unset means no behaviour change: the router classifies the objective."""
    from multi_hive.core import model_router
    from multi_hive.core.model_router import FAST, STRONG, select_plan_tier

    assert model_router.PLAN_TIER == ""  # not set in the test env
    assert select_plan_tier("write a function that adds two numbers") == FAST
    assert select_plan_tier("refactor the parser to be thread-safe") == STRONG


def test_plan_tier_pins_the_planner_and_ticket_writer(monkeypatch):
    """
    The plan and the tickets decide what the task IS. Everything downstream is
    executing that paraphrase, and a bad ticket cannot be rescued by escalating
    the editor — the editor is faithfully building the wrong thing.

    There is no retry ladder here to climb out of it either: by the time the editor
    fails, the ticket it is failing against has already been written.
    """
    from multi_hive.core import model_router
    from multi_hive.core.model_router import STRONG, select_plan_tier

    monkeypatch.setattr(model_router, "PLAN_TIER", STRONG)
    # A trivial objective the router would otherwise send to the fast model.
    assert select_plan_tier("write a function that adds two numbers") == STRONG


def test_force_tier_still_outranks_plan_tier(monkeypatch):
    """
    HIVE_FORCE_TIER exists so a benchmark can pin every tier. A benchmark silently
    un-pinned by a routing rule is not a benchmark, so PLAN_TIER does not get to
    outrank it — the same rule tier_floor already lives under.
    """
    from multi_hive.core import model_router
    from multi_hive.core.model_router import FAST, STRONG, select_plan_tier

    monkeypatch.setattr(model_router, "FORCE_TIER", FAST)
    monkeypatch.setattr(model_router, "PLAN_TIER", STRONG)
    assert select_plan_tier("anything at all") == FAST
