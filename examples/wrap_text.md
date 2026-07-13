Implement wrap_text(text: str, width: int) -> list[str] which greedily wraps text
into lines of at most `width` characters, breaking only on spaces.

A single word longer than `width` must be hard-split across lines. Runs of
multiple spaces collapse. No line may be empty, and no line may have leading or
trailing spaces.

Save it to outputs/wrap.py

ACCEPTANCE outputs/wrap.py
assert wrap_text("", 8) == []
assert wrap_text("one two", 10) == ["one two"]
assert wrap_text("one two three", 7) == ["one two", "three"]

# A word longer than the width is hard-split across lines.
assert wrap_text("supercalifragilistic", 6) == ["superc", "alifra", "gilist", "ic"]

# Runs of multiple spaces collapse.
assert wrap_text("a  b", 4) == ["a b"]

# No line is empty, none is padded, none exceeds the width.
for line in wrap_text("the rain in spain falls mainly on the plain", 11):
    assert line, "empty line"
    assert line == line.strip(), f"padded line {line!r}"
    assert len(line) <= 11, f"line over width: {line!r}"
