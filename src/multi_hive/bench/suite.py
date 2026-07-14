"""
suite.py — the tasks, and the hidden tests that grade them.

The tests are hidden from the model on purpose. An earlier version of this
harness graded on "does it compile, does it run, does it contain the right
function names" — and every model scored full marks, which measured nothing.
Compiling and running is a floor, not a ranking.

What actually distinguishes a weak model from a strong one is code that is
clean, confident, and *subtly wrong*: semver comparison that silently ignores
build metadata, an LRU cache whose get() does not refresh recency, a word-wrap
that never hard-splits an oversized word. Each task below therefore ships a test
suite probing the edge cases the task implies but does not spell out.

Adding a task
-------------
Pick something with a trap. If a 7B model passes it on the first try, it is not
telling you anything you did not already know.
"""
from __future__ import annotations

import secrets
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    name: str
    complexity: str  # trivial | moderate | hard
    filename: str  # the GRADED artefact — the hidden tests import this one
    prompt: str  # what the model is told
    tests: str  # what it is graded on, and never sees

    # Other files the task must also produce, alongside `filename`, in outputs/.
    #
    # This is the workload multi_hive actually exists for, and until now the
    # benchmark contained not one example of it. Every other task in this suite is
    # a single self-contained function in a single file — which is precisely what a
    # one-shot call to a strong model is best at, and precisely where planning,
    # ticketing, cross-file context and iterative retry have no room to add
    # anything. On that suite the pipeline is 2x slower than one call to the 30B
    # and two tasks worse, and it could not have been otherwise: there was nothing
    # to plan and nothing to decompose.
    #
    # A task with a second file changes the question. The graded module IMPORTS the
    # other one, so the two have to agree on an interface that neither file states
    # on its own. That is a coordination problem, and coordination is the only thing
    # a pipeline can do that a single prompt cannot.
    extra_files: tuple[str, ...] = ()

    @property
    def files(self) -> tuple[str, ...]:
        """Every file the task must produce. The first one is the graded artefact."""
        return (self.filename, *self.extra_files)

    @property
    def objective(self) -> str:
        """The prompt as a user objective, for the end-to-end sprint suite."""
        if not self.extra_files:
            return f"{self.prompt}\nSave it to outputs/{self.filename}"

        listing = "\n".join(f"  outputs/{name}" for name in self.files)
        return f"{self.prompt}\n\nSave the files to:\n{listing}"


@dataclass
class Grade:
    extracted: bool = False
    compiles: bool = False
    imports: bool = False
    passed: bool = False
    failure: str = ""

    @property
    def score(self) -> int:
        return int(self.passed)


TASKS: list[Task] = [
    Task(
        name="lru_cache",
        complexity="moderate",
        filename="lru.py",
        prompt=(
            "Implement an LRUCache class with O(1) get and put, using a dict plus a "
            "doubly linked list. Do NOT use collections.OrderedDict or functools.lru_cache.\n"
            "  __init__(self, capacity: int)\n"
            "  get(self, key) -> int, returning -1 if the key is absent\n"
            "  put(self, key, value) -> None, evicting the least recently used entry "
            "when over capacity\n"
            "Reading a key with get() counts as using it."
        ),
        # Traps: does get() refresh recency, and does overwriting an existing key
        # count as a use? Both are easy to miss and invisible to a smoke test.
        tests="""
c = M.LRUCache(2)
c.put(1, 1); c.put(2, 2)
assert c.get(1) == 1
c.put(3, 3)
assert c.get(2) == -1, "get() did not refresh recency"
assert c.get(1) == 1 and c.get(3) == 3

c = M.LRUCache(2)
c.put(1, 1); c.put(2, 2); c.put(1, 10)
c.put(3, 3)
assert c.get(2) == -1, "overwriting an existing key did not refresh recency"
assert c.get(1) == 10, "overwrite did not update the value"

assert M.LRUCache(1).get(99) == -1
""",
    ),
    Task(
        name="merge_intervals",
        complexity="moderate",
        filename="intervals.py",
        prompt=(
            "Implement merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]] "
            "which merges all overlapping intervals and returns them sorted by start.\n"
            "The input may be unsorted. Intervals that merely touch (e.g. (1,3) and (3,5)) "
            "count as overlapping and must be merged."
        ),
        # Traps: unsorted input, touching-but-not-overlapping, and a fully
        # contained interval that must not extend the enclosing one.
        tests="""
assert M.merge_intervals([]) == []
assert M.merge_intervals([(1, 3)]) == [(1, 3)]

got = M.merge_intervals([(1, 3), (2, 6), (8, 10), (15, 18)])
assert got == [(1, 6), (8, 10), (15, 18)], got

got = M.merge_intervals([(8, 10), (1, 3), (2, 6)])
assert got == [(1, 6), (8, 10)], f"unsorted input mishandled: {got}"

got = M.merge_intervals([(1, 3), (3, 5)])
assert got == [(1, 5)], f"touching intervals not merged: {got}"

got = M.merge_intervals([(1, 10), (2, 4)])
assert got == [(1, 10)], f"contained interval broke the range: {got}"
""",
    ),
    Task(
        name="semver",
        complexity="hard",
        filename="semver.py",
        prompt=(
            "Implement compare_semver(a: str, b: str) -> int returning -1 if a < b, "
            "0 if equal, 1 if a > b, following the Semantic Versioning 2.0.0 precedence "
            "rules.\n"
            "Handle versions of the form MAJOR.MINOR.PATCH with an optional "
            "-prerelease and an optional +build metadata suffix."
        ),
        # Every trap here is stated plainly in the semver spec, and every one is
        # commonly missed: build metadata is ignored entirely; a prerelease sorts
        # BELOW its release; numeric identifiers compare numerically (2 < 10, not
        # "10" < "2"); more identifiers wins on an equal prefix.
        tests="""
assert M.compare_semver("1.0.0", "2.0.0") == -1
assert M.compare_semver("2.0.0", "1.0.0") == 1
assert M.compare_semver("1.2.3", "1.2.3") == 0

assert M.compare_semver("1.0.0+build1", "1.0.0+build2") == 0, "build metadata must be ignored"
assert M.compare_semver("1.0.0-alpha", "1.0.0") == -1, "prerelease must sort below its release"
assert M.compare_semver("1.0.0-alpha", "1.0.0-beta") == -1

assert M.compare_semver("1.0.0-alpha.2", "1.0.0-alpha.10") == -1, \\
    "numeric prerelease identifiers must compare numerically, not as strings"
assert M.compare_semver("1.0.0-alpha", "1.0.0-alpha.1") == -1, \\
    "more prerelease identifiers wins when the prefix is equal"
assert M.compare_semver("1.0.0-alpha.1", "1.0.0-alpha.beta") == -1, \\
    "numeric identifiers sort below alphanumeric ones"
""",
    ),
    Task(
        name="word_wrap",
        complexity="hard",
        filename="wrap.py",
        prompt=(
            "Implement wrap_text(text: str, width: int) -> list[str] which greedily wraps "
            "text into lines of at most `width` characters, breaking only on spaces.\n"
            "A single word longer than `width` must be hard-split across lines. "
            "Runs of multiple spaces collapse. No line may be empty, and no line may "
            "have leading or trailing spaces."
        ),
        # Traps: the oversized word (almost every model forgets the hard split),
        # the exact-width boundary, and collapsing whitespace. As of the last run
        # NEITHER tier passes this — it is the honest ceiling of the current setup.
        tests="""
assert M.wrap_text("", 10) == []
assert M.wrap_text("hello world", 20) == ["hello world"]
assert M.wrap_text("hello world", 5) == ["hello", "world"]

got = M.wrap_text("aaa bbb ccc", 7)
assert got == ["aaa bbb", "ccc"], got

got = M.wrap_text("abcdefghij", 4)
assert got == ["abcd", "efgh", "ij"], f"oversized word not hard-split: {got}"

got = M.wrap_text("hi     there", 12)
assert got == ["hi there"], f"whitespace not collapsed: {got}"

for line in M.wrap_text("the quick brown fox jumps over the lazy dog", 9):
    assert line == line.strip() and line, f"bad line {line!r}"
    assert len(line) <= 9, f"line over width: {line!r}"
""",
    ),
    # ── Added to give the suite enough resolution to measure a change ─────────
    #
    # Four tasks was not a benchmark, it was a rumour. One task is 25% of the
    # score, and two of the four (semver, word_wrap) are coin flips — so the
    # instrument could not resolve a one-task improvement, which is roughly what
    # every worthwhile change to this system is worth. We were measuring with a
    # ruler whose error bars were wider than the thing being measured.
    #
    # These four are deliberately *achievable* — a task no tier can pass tells you
    # nothing on any run, and a task every tier passes tells you nothing either.
    # Each one has a trap the prompt implies but does not spell out, which is the
    # only kind of task worth adding (see the module docstring).
    Task(
        name="roman",
        complexity="moderate",
        filename="roman.py",
        prompt=(
            "Implement two functions for Roman numerals in the range 1..3999:\n"
            "  to_roman(n: int) -> str\n"
            "  from_roman(s: str) -> int\n"
            "Use standard subtractive notation."
        ),
        # Trap: "standard subtractive notation" is the whole task and it is stated
        # in four words. A model that builds a naive greedy table emits IIII for 4
        # and LXXXX for 90 — clean, confident, and wrong. Round-tripping catches the
        # from_roman half, which is usually the sloppier one.
        tests="""
assert M.to_roman(4) == "IV", M.to_roman(4)
assert M.to_roman(9) == "IX", M.to_roman(9)
assert M.to_roman(40) == "XL", M.to_roman(40)
assert M.to_roman(90) == "XC", M.to_roman(90)
assert M.to_roman(400) == "CD", M.to_roman(400)
assert M.to_roman(900) == "CM", M.to_roman(900)
assert M.to_roman(1994) == "MCMXCIV", M.to_roman(1994)
assert M.to_roman(3999) == "MMMCMXCIX", M.to_roman(3999)

assert M.from_roman("IV") == 4
assert M.from_roman("MCMXCIV") == 1994

for n in (1, 3, 8, 14, 44, 399, 1666, 3999):
    assert M.from_roman(M.to_roman(n)) == n, f"round trip failed at {n}"
""",
    ),
    Task(
        name="brackets",
        complexity="trivial",
        filename="brackets.py",
        prompt=(
            "Implement is_balanced(s: str) -> bool which returns True when every "
            "bracket in s is closed by the matching type in the correct order.\n"
            "The brackets are (), [] and {}. Any other character is ignored."
        ),
        # Trap: a model that counts brackets instead of stacking them passes "()[]"
        # and then says "([)]" is balanced. The interleaved case is the whole point,
        # and a counter cannot see it. Empty string is vacuously True.
        tests="""
assert M.is_balanced("") is True
assert M.is_balanced("()") is True
assert M.is_balanced("([{}])") is True
assert M.is_balanced("a(b)c[d]{e}") is True, "non-bracket characters not ignored"

assert M.is_balanced("(") is False
assert M.is_balanced(")(") is False, "order not checked"
assert M.is_balanced("(]") is False, "closing bracket type not checked"
assert M.is_balanced("([)]") is False, "interleaved brackets accepted — counting, not stacking"
assert M.is_balanced("(((") is False
""",
    ),
    Task(
        name="rle",
        complexity="moderate",
        filename="rle.py",
        prompt=(
            "Implement run-length encoding:\n"
            "  encode(s: str) -> str   e.g. 'aaabb' -> 'a3b2'\n"
            "  decode(s: str) -> str   the exact inverse of encode\n"
            "Every run carries an explicit count, including runs of length 1."
        ),
        # Traps: the run of length 1 (models love to emit "abc" instead of
        # "a1b1c1", which silently breaks the inverse), and the multi-digit count —
        # a decoder that reads one character per count turns "a12" into 1 'a' and a
        # stray '2'. Both are invisible unless you round-trip.
        tests="""
assert M.encode("") == ""
assert M.encode("aaabb") == "a3b2", M.encode("aaabb")
assert M.encode("abc") == "a1b1c1", f"runs of length 1 need a count: {M.encode('abc')}"
assert M.encode("a" * 12) == "a12", f"multi-digit count: {M.encode('a' * 12)}"

assert M.decode("a3b2") == "aaabb"
assert M.decode("a12") == "a" * 12, "decoder read one digit, not the whole count"

for s in ("", "a", "abc", "wwwwaaadexxxxxx", "z" * 30 + "y"):
    assert M.decode(M.encode(s)) == s, f"round trip failed on {s!r}"
""",
    ),
    Task(
        name="flatten",
        complexity="moderate",
        filename="flatten.py",
        prompt=(
            "Implement flatten(d: dict, sep: str = '.') -> dict which flattens a "
            "nested dictionary into a single level, joining the nested keys with "
            "`sep`.\n"
            "Only dictionaries are recursed into. Any other value — including a "
            "list — is left exactly as it is."
        ),
        # Trap: "only dictionaries are recursed into" is stated, and models still
        # walk into lists, or worse, flatten a dict that happens to sit inside one.
        # The empty nested dict is the second trap: it contributes no leaf, so it
        # should contribute no key, and a naive recursion silently drops it or
        # emits a key with no value.
        tests="""
assert M.flatten({}) == {}
assert M.flatten({"a": 1}) == {"a": 1}
assert M.flatten({"a": {"b": 1}}) == {"a.b": 1}

got = M.flatten({"a": {"b": {"c": 1}}, "d": 2})
assert got == {"a.b.c": 1, "d": 2}, got

got = M.flatten({"a": [1, {"b": 2}]})
assert got == {"a": [1, {"b": 2}]}, f"recursed into a list: {got}"

got = M.flatten({"a": {"b": 1}}, sep="/")
assert got == {"a/b": 1}, f"sep ignored: {got}"

got = M.flatten({"a": {}, "b": 1})
assert got == {"b": 1}, f"empty nested dict produced a key: {got}"
""",
    ),
    # ── The task this project actually exists for ─────────────────────────────
    #
    # Every other task in this suite is one self-contained function in one file.
    # That is exactly what a one-shot call to a strong model is best at, and exactly
    # where a pipeline has nothing to contribute: nothing to plan, nothing to
    # decompose, no cross-file context to carry, no reason to iterate. Measured on
    # the 8-task suite, the full hive is 2x slower than a single call to the 30B and
    # two tasks worse — and it could not have been otherwise.
    #
    # This one has two files, and the graded module IMPORTS the other. Neither file
    # states the interface on its own; the model has to hold both in its head at
    # once and make them agree. That is a coordination problem, and coordination is
    # the only thing a pipeline can do that a single prompt cannot — it is what the
    # planner, the ticket queue, and core/ast_utils (which feeds one file's
    # signatures into the next file's prompt) were all built for.
    #
    # If the hive cannot beat a one-shot 30B here, that is a finding about the
    # architecture, and a much more useful one than "it loses at leetcode".
    Task(
        name="word_stats",
        complexity="hard",
        filename="stats.py",
        extra_files=("tokens.py",),
        prompt=(
            "Implement a two-module word-frequency tool.\n\n"
            "outputs/tokens.py defines:\n"
            "  tokenize(text: str) -> list[str]\n"
            "  Lowercases the text, splits on whitespace, and strips leading and "
            "trailing punctuation (.,!?;:'\") from each word. Words that become "
            "empty are dropped.\n\n"
            "outputs/stats.py defines:\n"
            "  top_words(text: str, n: int) -> list[tuple[str, int]]\n"
            "  The n most frequent words as (word, count), most frequent first. "
            "Ties are broken alphabetically.\n\n"
            "stats.py MUST import tokenize from tokens.py and use it. "
            "Do NOT reimplement tokenization inside stats.py."
        ),
        # The `sys.modules` check is the whole point: it proves stats.py actually
        # DELEGATES to tokens.py rather than quietly reimplementing it. A model that
        # inlines the tokenizer produces two files that both look right and are not
        # a system — and that is the failure mode a single-file benchmark can never
        # see.
        #
        # The other traps are ordinary: n larger than the vocabulary, and empty
        # input. Neither is stated in the prompt.
        tests="""
import sys
assert "tokens" in sys.modules, "stats.py did not import tokens.py — it reimplemented it"

T = sys.modules["tokens"]
assert T.tokenize("Hello, world!") == ["hello", "world"], T.tokenize("Hello, world!")
assert T.tokenize("  a   b  ") == ["a", "b"]
assert T.tokenize("...") == [], "punctuation-only word was not dropped"

got = M.top_words("the cat the dog the bird cat", 2)
assert got == [("the", 3), ("cat", 2)], got

got = M.top_words("b a c", 3)
assert got == [("a", 1), ("b", 1), ("c", 1)], f"ties not broken alphabetically: {got}"

got = M.top_words("Dog! dog, DOG. cat", 1)
assert got == [("dog", 3)], f"case/punctuation not normalised through tokenize: {got}"

assert M.top_words("", 3) == []

got = M.top_words("a b", 5)
assert got == [("a", 1), ("b", 1)], f"n larger than the vocabulary: {got}"
""",
    ),
]

BY_NAME = {task.name: task for task in TASKS}


# exec_module is guarded with BaseException, not Exception, so a candidate that
# calls sys.exit()/exit() at import becomes an IMPORT_FAIL instead of escaping
# with returncode 0. The PASS line carries a per-run nonce ({token}) the
# candidate cannot know, and grade() confirms the pass by that line — never by
# the return code alone, which a top-level exit(0) or os._exit(0) can forge.
_HARNESS = """
import importlib.util, sys, traceback
spec = importlib.util.spec_from_file_location("candidate", r"{module}")
M = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(M)
except BaseException:
    print("IMPORT_FAIL"); traceback.print_exc(); sys.exit(2)
try:
{body}
except AssertionError as e:
    print("TEST_FAIL"); print(str(e)[:200]); sys.exit(3)
except Exception as e:
    print("TEST_ERROR"); print(f"{{type(e).__name__}}: {{str(e)[:160]}}"); sys.exit(4)
print("PASS {token}")
"""


def grade(code: str | dict[str, str], task: Task, timeout: int = 60) -> Grade:
    """
    Run `task`'s hidden suite against the candidate, in a separate process.

    Separate process because the code under test is model-authored: it may
    hang, exit(), or blow the stack, and none of that should take the harness
    with it.

    `code` is either a single source string (the common case — one file), or a
    {filename: source} map for a multi-file task. Every file is written into the
    temp directory side by side, so the graded module's own
    `from tokens import tokenize` resolves against its sibling exactly as it does
    in the workspace: the harness script lives in that directory, so it is
    sys.path[0].

    A missing sibling is a FAILURE, not a crash. It has to be — "the model wrote
    one of the two files it was asked for" is precisely the outcome a multi-file
    task exists to catch, and reporting it as an import error would hide what
    actually went wrong.
    """
    files = {task.filename: code} if isinstance(code, str) else dict(code)

    if not (files.get(task.filename) or "").strip():
        result = Grade()
        result.failure = "no code"
        return result

    missing = [name for name in task.files if not (files.get(name) or "").strip()]
    if missing:
        result = Grade(extracted=True)
        result.failure = f"missing {', '.join('outputs/' + m for m in missing)}"
        return result

    result = Grade(extracted=True)

    with tempfile.TemporaryDirectory() as tmp:
        # Every file, side by side — the graded module imports its siblings by
        # plain name, which only works because they share a directory.
        for name, source in files.items():
            (Path(tmp) / name).write_text(source, encoding="utf-8")

        module = Path(tmp) / task.filename

        for name in task.files:
            compiled = subprocess.run(
                [sys.executable, "-m", "py_compile", str(Path(tmp) / name)],
                capture_output=True,
                text=True,
            )
            if compiled.returncode != 0:
                result.failure = (
                    "syntax error" if name == task.filename else f"syntax error in {name}"
                )
                return result
        result.compiles = True

        body = "\n".join("    " + line for line in task.tests.strip().splitlines())
        token = secrets.token_hex(8)
        harness = Path(tmp) / "harness.py"
        harness.write_text(
            _HARNESS.format(module=str(module), body=body, token=token), encoding="utf-8"
        )

        try:
            run = subprocess.run(
                [sys.executable, str(harness)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            result.failure = "timeout"
            return result

        out = (run.stdout or "") + (run.stderr or "")

        # Exit 0 alone is not a pass. A candidate that calls exit(0)/os._exit(0)
        # at import terminates the harness with returncode 0 before a single
        # assert runs; the nonce-tagged PASS line is printed only after every
        # assert survived, so it — not the return code — is the ground truth.
        if run.returncode == 0 and f"PASS {token}" in out:
            result.imports = result.passed = True
            return result

        if "IMPORT_FAIL" in out:
            lines = [x for x in out.strip().splitlines() if x.strip()]
            result.failure = lines[-1][:70] if lines else "import failed"
            return result

        if run.returncode == 0:
            # Exited 0 without the sentinel → a top-level exit() ran before the
            # tests could. Not a pass.
            result.imports = True
            result.failure = "process exited before tests ran (top-level exit()?)"
            return result

        result.imports = True
        stdout_lines = [x for x in (run.stdout or "").strip().splitlines() if x.strip()]
        result.failure = (
            stdout_lines[1][:70] if len(stdout_lines) > 1 else "assertion failed"
        )
        return result
