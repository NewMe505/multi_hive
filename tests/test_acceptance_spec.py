"""
The model must not grade its own homework.

The editor used to write both the implementation and the asserts that judged it,
and that fails in both directions:

  wrong assert  → rejects correct code. Observed live: the model asserted
                  `wrap_text("hello world", 10) == ["hello world"]`. That string
                  is eleven characters and the width is ten, so it cannot fit —
                  the assertion was simply false. Its own test rejected its own
                  correct implementation, burnt the retry budget, escalated the
                  tier, and woke a human.

  lazy assert   → waves a bug through, because the check was written by the thing
                  that produced the bug.

spec_writer_node now derives the criteria from the task, before any code exists.
reviewer_node runs the code against that. And because a *spec* can also be wrong,
reviewer_node may drop an assertion the adjudicator rules invalid — which opens
an obvious hole (delete the spec until nothing fails), so the guards below are
the load-bearing part of this file.
"""
from unittest.mock import Mock, patch

import pytest

from multi_hive.config import SPEC_REPAIR_LIMIT
from multi_hive.nodes.execution.reviewer_node import _assertion_is_wrong
from multi_hive.nodes.execution.spec_writer_node import parse_spec, spec_writer_node
from multi_hive.prompts import get_editor_prompt


class TestSpecParsing:
    def test_a_stateful_script_survives_intact(self):
        # The contract for an LRU cache is inherently stateful: put, then get.
        # State persists down the script, which is the entire point of the format.
        raw = "c = LRUCache(2)\nc.put(1, 1)\nassert c.get(1) == 1\nassert c.get(9) == -1"
        assert parse_spec(raw) == [
            "c = LRUCache(2)",
            "c.put(1, 1)",
            "assert c.get(1) == 1",
            "assert c.get(9) == -1",
        ]

    def test_a_bare_mutation_assert_is_refused(self):
        # The spec writer really produced this: `assert LRUCache(2).put(2, 6)`.
        # put() mutates and returns None, so this line fails against a perfectly
        # correct implementation. An assert must compare a value to an expectation.
        raw = "c = LRUCache(2)\nassert c.put(2, 6)\nassert c.get(2) == 6\nassert c.get(1) == -1"
        spec = parse_spec(raw)
        assert "assert c.put(2, 6)" not in spec
        assert "assert c.get(2) == 6" in spec

    def test_prose_and_fences_are_stripped(self):
        raw = (
            "Here are the acceptance tests:\n"
            "```python\n"
            "assert add(2, 3) == 5\n"
            "assert add(0, 0) == 0\n"
            "```\n"
            "These cover the basic case."
        )
        assert parse_spec(raw) == ["assert add(2, 3) == 5", "assert add(0, 0) == 0"]

    @pytest.mark.parametrize(
        "hostile",
        [
            "assert __import__('os').system('x') == 0",
            "assert open('/etc/passwd').read() == 'x'",
            "assert eval('1+1') == 2",
            "import os",
        ],
    )
    def test_statements_that_reach_outside_the_sandbox_are_refused(self, hostile):
        # These lines execute inside the sandbox, against generated code. They are
        # model output — untrusted input — and dressing a call up as an `assert`
        # does not make it safe.
        raw = f"{hostile}\nassert f(1) == 1\nassert f(2) == 2"
        assert hostile not in parse_spec(raw)

    def test_unparseable_lines_are_dropped(self):
        # A syntax error inside the harness surfaces as "the implementation
        # exploded", which is a lie about whose fault it is.
        raw = "assert add(2, == 5\nassert add(1, 1) == 2\nassert add(0, 0) == 0"
        assert parse_spec(raw) == ["assert add(1, 1) == 2", "assert add(0, 0) == 0"]

    def test_a_spec_with_no_assertions_is_no_spec(self):
        # Setup lines alone check nothing. Refuse it rather than pretend the code
        # was verified.
        assert parse_spec("c = LRUCache(2)\nc.put(1, 1)") == []

    def test_an_assertion_that_never_touches_the_code_is_refused(self):
        # The spec writer really produced this for the word-wrap task:
        #     t = "This is a test string that needs to be wrapped."
        #     assert len(t.split()) == 7          <- ten words, actually
        # It asserts something about the TEST DATA. No implementation can make it
        # pass or fail, so its only possible effect is rejecting correct code.
        raw = (
            't = "This is a test string that needs to be wrapped."\n'
            "w = 10\n"
            "assert len(t.split()) == 7\n"
            "assert wrap_text(t, w) == ['This is', 'a test']\n"
            "assert wrap_text('abc', 10) == ['abc']\n"
        )
        spec = parse_spec(raw)

        assert "assert len(t.split()) == 7" not in spec
        assert "assert wrap_text(t, w) == ['This is', 'a test']" in spec

    def test_assertions_on_tainted_setup_variables_are_kept(self):
        # `c` came from LRUCache(...), so `assert c.get(1) == 1` genuinely exercises
        # the implementation even though `c` itself is a local. The taint has to
        # flow through the assignment, or this whole format collapses.
        raw = "c = LRUCache(2)\nc.put(1, 1)\nassert c.get(1) == 1\nassert c.get(9) == -1"
        assert parse_spec(raw) == [
            "c = LRUCache(2)",
            "c.put(1, 1)",
            "assert c.get(1) == 1",
            "assert c.get(9) == -1",
        ]

    def test_a_spec_left_too_thin_after_filtering_is_refused_entirely(self):
        # One surviving assertion cannot be adjudicated away if it turns out to be
        # wrong — the last check is never dropped — so a lone bad assert would
        # guarantee a false failure. Better no spec than a spec that can only lie.
        raw = 't = "abc"\nassert len(t) == 3\nassert wrap_text(t, 10) == ["abc"]\n'
        assert parse_spec(raw) == []

    def test_no_spec_is_not_fatal(self):
        # A spec writer that returns prose must not kill the sprint; reviewer_node
        # degrades to "does the module import", and the failure is in the ledger.
        state = {"current_task": "do a thing", "messages": [], "model_tier": "fast"}

        with patch("multi_hive.nodes.execution.spec_writer_node.get_llm") as llm:
            llm.return_value.invoke = Mock(
                return_value=Mock(content="I think you should write some tests.")
            )
            delta = spec_writer_node(state)

        assert delta == {"acceptance": [], "spec_repairs": 0}


class TestTheCodeIsNeverLeftUnchecked:
    """
    The rule this file exists to enforce, learned the expensive way.

    Removing the editor's self-asserts was right in principle — a model must not
    grade its own homework — but the independent spec is not always producible.
    When it was missing AND the self-asserts were gone, nothing verified the code
    at all, and the hive shipped a word-wrap with no hard-split. The sprint
    benchmark went 3/4 -> 1/4.

    So: with a spec, the spec checks the code. Without one, the model's own
    asserts do, flawed as they are. A bad check beats no check. There is no third
    state where nothing checks anything.
    """

    def test_with_a_spec_the_editor_is_told_not_to_write_tests(self):
        prompt = get_editor_prompt(
            "objective", "", acceptance=["c = LRUCache(2)", "assert c.get(1) == -1"]
        )

        assert "Do NOT write assert statements" in prompt
        assert "assert c.get(1) == -1" in prompt  # it is shown the contract
        assert "ASSERT RULES" not in prompt

    def test_without_a_spec_the_editor_must_test_itself(self):
        prompt = get_editor_prompt("objective", "", acceptance=[])

        assert "ASSERT RULES" in prompt
        assert "__main__" in prompt
        assert "Do NOT write assert statements" not in prompt

    def test_the_fallback_still_warns_about_the_mistakes_it_actually_makes(self):
        # The self-asserts are only tolerable if the model is warned off the two
        # errors it demonstrably makes: wrong arithmetic, and asserting the
        # truthiness of a mutation that returns None.
        prompt = get_editor_prompt("objective", "", acceptance=[])

        assert "CHECK YOUR ARITHMETIC" in prompt
        assert "returns None" in prompt


class TestSpecIsWrittenOncePerTask:
    def test_a_retry_does_not_get_fresh_goalposts(self):
        # If the spec were regenerated on every attempt, the editor would be aiming
        # at a moving target and a failure would mean nothing.
        state = {
            "current_task": "do a thing",
            "acceptance": ["assert f(1) == 1"],
            "messages": [],
        }

        with patch("multi_hive.nodes.execution.spec_writer_node.get_llm") as llm:
            assert spec_writer_node(state) == {}
            llm.assert_not_called()


class TestAdjudicationGuards:
    """
    reviewer_node can drop an acceptance assertion the adjudicator calls invalid.
    That is necessary — a wrong spec rejects correct code — and it is also the
    most dangerous code in the system, because "delete the test until it passes"
    is exactly how a model would cheat if we let it.
    """

    def _adjudicate(self, verdict: str, **kwargs):
        defaults = {
            "current_task": "wrap text at a width",
            "assertion": "assert wrap_text('hello world', 10) == ['hello world']",
            "code": "def wrap_text(t, w): ...",
            "tier": "fast",
            "can_repair": True,
        }
        defaults.update(kwargs)

        with patch("multi_hive.nodes.execution.reviewer_node.get_llm") as llm:
            llm.return_value.invoke = Mock(return_value=Mock(content=verdict))
            return _assertion_is_wrong(**defaults)

    def test_a_genuinely_wrong_assertion_is_dropped(self):
        assert self._adjudicate("ASSERT_WRONG: 'hello world' is 11 chars, cannot fit width 10")

    def test_the_code_is_blamed_by_default(self):
        assert not self._adjudicate("CODE_WRONG: the implementation never splits long words")

    def test_an_ambiguous_verdict_blames_the_code(self):
        # Ruling a valid assertion "wrong" deletes a real check and lets a bug
        # through. Ruling a wrong assertion "the code's fault" merely reproduces
        # the behaviour we already had. Ambiguity must resolve to the safe one.
        assert not self._adjudicate("I'm not really sure, it could be either honestly")
        assert not self._adjudicate("")

    def test_a_dead_adjudicator_cannot_delete_the_spec(self):
        with patch("multi_hive.nodes.execution.reviewer_node.get_llm") as llm:
            llm.side_effect = ConnectionError("ollama is down")
            assert not _assertion_is_wrong(
                current_task="t",
                assertion="assert f(1) == 1",
                code="...",
                tier="fast",
                can_repair=True,
            )

    def test_the_repair_budget_is_a_hard_stop(self):
        # can_repair=False is how reviewer_node enforces SPEC_REPAIR_LIMIT and the
        # never-drop-the-last-assertion rule. When it is off, we must not even ask
        # the model — an unbounded "is this assert wrong?" loop is how the spec
        # gets deleted one line at a time.
        with patch("multi_hive.nodes.execution.reviewer_node.get_llm") as llm:
            assert not _assertion_is_wrong(
                current_task="t",
                assertion="assert f(1) == 1",
                code="...",
                tier="fast",
                can_repair=False,
            )
            llm.assert_not_called()

    def test_the_limit_is_small(self):
        # Not a real assertion about behaviour so much as a tripwire: if someone
        # raises this to 10, the spec can be dismantled almost entirely.
        assert SPEC_REPAIR_LIMIT <= 3
