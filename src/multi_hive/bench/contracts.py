"""
contracts.py — human-written acceptance contracts for the bench tasks.

These exist to measure the one thing the contract feature claims: that when a
human states what "correct" means, the hive delivers it, instead of arguing with
itself about asserts it made up. Run them with:

    python scripts/bench.py sprint --contract

Two rules were followed writing these, and both are load-bearing.

1. DERIVED FROM THE PROMPT, NOTHING ELSE
   Every assert below restates a requirement the task prompt already spells out
   in prose. Nothing here is knowledge the model was denied. Where a hidden test
   probes something the prompt does NOT state — that overwriting an existing LRU
   key refreshes its recency, that a fully contained interval must not extend its
   enclosing one — the contract stays silent, because a human writing a contract
   from that prompt would have stayed silent too. The contract is a strict subset
   of the hidden suite, on purpose.

2. DIFFERENT LITERALS THAN THE HIDDEN TESTS
   This is what keeps the benchmark honest. The editor SEES the contract, so a
   model that memorises `wrap_text("abcdefghij", 4) == [...]` instead of
   implementing a hard split would sail through it. So no value in this file
   appears in suite.py: the contract splits "supercalifragilistic" at width 6,
   the hidden test splits "abcdefghij" at width 4. Same requirement, different
   numbers.

   Which makes the hidden suite a working gaming detector. Code that hardcodes
   the contract passes the contract and FAILS the bench, and the gap between the
   two scores is exactly the amount of cheating. If contract mode ever reports a
   satisfied contract alongside a failed hidden suite, that is the alarm, and it
   is a real finding rather than a suspicion.

Never copy an assert from suite.py into this file. It would turn the benchmark
into an open-book exam and the number it prints into a lie.
"""
from __future__ import annotations

# Keyed by Task.name in suite.py.
CONTRACTS: dict[str, str] = {
    # Prompt states: capacity, get() returns -1 when absent, put() evicts the
    # least recently used entry when over capacity, and a get() counts as a use.
    # It does NOT state that overwriting a key counts as a use — so neither does
    # this. The hidden suite still checks it.
    "lru_cache": """\
c = LRUCache(3)
c.put(10, 100)
c.put(20, 200)
c.put(30, 300)

assert c.get(10) == 100

# 10 was just read, so 20 is now the least recently used and is what gets evicted.
c.put(40, 400)
assert c.get(20) == -1
assert c.get(10) == 100
assert c.get(30) == 300
assert c.get(40) == 400

assert LRUCache(2).get(7) == -1
""",
    # Prompt states: merges overlapping intervals, returns them sorted by start,
    # the input may be unsorted, and touching intervals count as overlapping.
    #
    # The empty-input case is deliberately absent. `merge_intervals([]) == []` is
    # the one assert whose literals CANNOT be varied — there is only one empty
    # list — so it would be shared with the hidden suite, and a shared assert is
    # a hole in the gaming detector. It also happens to be the one case the prompt
    # never mentions. Both reasons point the same way: leave it out.
    "merge_intervals": """\
assert merge_intervals([(5, 7)]) == [(5, 7)]

# Unsorted input, and the result comes back sorted by start.
assert merge_intervals([(20, 30), (0, 10), (5, 15)]) == [(0, 15), (20, 30)]

# Touching counts as overlapping.
assert merge_intervals([(1, 4), (4, 9)]) == [(1, 9)]

# Disjoint intervals are left alone.
assert merge_intervals([(0, 1), (5, 6)]) == [(0, 1), (5, 6)]
""",
    # Prompt states: -1/0/1, Semantic Versioning 2.0.0 precedence, optional
    # -prerelease and +build. Naming the spec imports its precedence rules, and a
    # human who knows semver writes exactly these.
    #
    # Every version string here is one the hidden suite does not use. The hidden
    # suite lives on 1.0.0 and 2.0.0; this contract lives on 4.x and 5.x.
    "semver": """\
assert compare_semver("4.1.0", "4.1.1") == -1
assert compare_semver("2.1.0", "2.0.9") == 1
assert compare_semver("3.4.5", "3.4.5") == 0

# Build metadata is ignored entirely.
assert compare_semver("1.2.3+sha.aaa", "1.2.3+sha.bbb") == 0

# A prerelease sorts below its own release.
assert compare_semver("5.0.0-rc.1", "5.0.0") == -1

# Numeric identifiers compare numerically: 2 < 11, not "11" < "2".
assert compare_semver("5.0.0-rc.2", "5.0.0-rc.11") == -1

# More identifiers wins when the common prefix is equal.
assert compare_semver("5.0.0-rc", "5.0.0-rc.1") == -1

# Numeric identifiers sort below alphanumeric ones.
assert compare_semver("5.0.0-1", "5.0.0-alpha") == -1
""",
    # Prompt states: greedy, at most `width`, break only on spaces, a word longer
    # than `width` is hard-split, runs of spaces collapse, no empty lines, no
    # leading or trailing spaces.
    #
    # This is the task the whole feature was built for. The model's own assert
    # here was `wrap_text("hello world", 10) == ["hello world"]` — 11 characters
    # into a width of 10 — so it rejected its own correct code, burned the retry
    # budget, escalated the tier, and woke a human. Nothing was wrong with the
    # program. A human cannot write that assert, because a human can count.
    "word_wrap": """\
assert wrap_text("", 8) == []
assert wrap_text("one two", 10) == ["one two"]
assert wrap_text("one two three", 7) == ["one two", "three"]

# A word longer than the width is hard-split across lines.
assert wrap_text("supercalifragilistic", 6) == ["superc", "alifra", "gilist", "ic"]

# Runs of multiple spaces collapse.
assert wrap_text("a  b", 4) == ["a b"]

# No line is empty, none has leading or trailing space, none exceeds the width.
for line in wrap_text("the rain in spain falls mainly on the plain", 11):
    assert line, "empty line"
    assert line == line.strip(), f"padded line {line!r}"
    assert len(line) <= 11, f"line over width: {line!r}"
""",
    # Prompt states subtractive notation, and that is the whole task. Every value
    # here differs from the hidden suite's: the contract checks 19 / 48 / 2024, the
    # suite checks 4 / 9 / 40 / 90 / 400 / 900 / 1994 / 3999. Same requirement,
    # different numbers — so a model that memorises this table still fails the bench.
    "roman": """\
assert to_roman(19) == "XIX"
assert to_roman(48) == "XLVIII"
assert to_roman(2024) == "MMXXIV"

assert from_roman("XIX") == 19
assert from_roman("MMXXIV") == 2024

# from_roman is the inverse of to_roman.
assert from_roman(to_roman(777)) == 777
""",
    # Prompt states: matching type, correct order, every other character ignored.
    # The interleaved case is what separates a stack from a counter, and a human
    # reading "in the correct order" would think to write it.
    "brackets": """\
assert is_balanced("{}") is True
assert is_balanced("{[()]}") is True
assert is_balanced("x{y}z") is True

assert is_balanced("]") is False
assert is_balanced("(}") is False
assert is_balanced("[(])") is False
""",
    # Prompt states: every run carries an explicit count, including runs of length
    # 1, and decode is the exact inverse of encode. Both are checked. Nothing else.
    "rle": """\
assert encode("zzzy") == "z3y1"
assert encode("q") == "q1"
assert encode("m" * 10) == "m10"

assert decode("z3y1") == "zzzy"

# decode is the exact inverse of encode.
assert decode(encode("pppqrrr")) == "pppqrrr"
""",
    # Prompt states: only dicts are recursed into, a list is left exactly as it is,
    # and the separator is configurable. It does NOT say what an empty nested dict
    # should do — so this contract stays silent about it, and the hidden suite still
    # checks it. The silence is the design, not an omission.
    "flatten": """\
assert flatten({"x": {"y": 9}}) == {"x.y": 9}
assert flatten({"p": {"q": {"r": 7}}}) == {"p.q.r": 7}

# Only dicts are recursed into. A list is left exactly as it is.
assert flatten({"k": [3, 4]}) == {"k": [3, 4]}

assert flatten({"m": {"n": 5}}, sep="_") == {"m_n": 5}
""",
    # The multi-file task. This contract names only what stats.py exports, because
    # stats.py is the module the contract harness imports — it cannot reach into
    # tokens.py, and a contract that tried would be testing the harness instead of
    # the code.
    #
    # That is a real limitation, and worth stating rather than hiding: this contract
    # can check that the SYSTEM behaves. It cannot check that stats.py actually
    # delegates to tokens.py instead of quietly reimplementing it. The hidden suite
    # checks exactly that, with `"tokens" in sys.modules`.
    #
    # So on this one task the hidden suite is strictly stronger than the contract —
    # the only place in the benchmark where that is true, and something to remember
    # when reading a contract-mode score for it.
    #
    # Literals disjoint from the hidden suite, as always: the contract counts x/y/z
    # and "Fog", the suite counts the/cat/dog and "Dog".
    "word_stats": """\
assert top_words("x y x z x y", 2) == [("x", 3), ("y", 2)]

# Ties are broken alphabetically.
assert top_words("q p r", 3) == [("p", 1), ("q", 1), ("r", 1)]

# Case and punctuation are normalised away before counting.
assert top_words("Fog! fog, FOG. mist", 1) == [("fog", 3)]
""",
}


def contract_for_task(name: str) -> str:
    """The contract for a bench task, or "" if it has none."""
    return CONTRACTS.get(name, "")
