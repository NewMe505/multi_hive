"""ast_utils was imported everywhere and existed nowhere. Pin its contract."""
from multi_hive.core.ast_utils import get_code_outline

SOURCE = '''
import numpy as np
from scipy import signal

SAMPLE_RATE = 44100

def generate_sine(freq: float, duration: float) -> np.ndarray:
    """A docstring that should not appear in the outline."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    return np.sin(2 * np.pi * freq * t)

class DelayLine:
    def __init__(self, taps: int):
        self.taps = taps

    async def process(self, x):
        return x
'''


def test_outline_keeps_signatures_and_drops_bodies():
    outline = get_code_outline(SOURCE)

    assert "import numpy as np" in outline
    assert "from scipy import signal" in outline
    assert "SAMPLE_RATE = ..." in outline
    assert "def generate_sine(freq: float, duration: float) -> np.ndarray" in outline
    assert "class DelayLine:" in outline
    assert "async def process(self, x)" in outline

    # Bodies and docstrings are the whole point of an outline: they must go.
    assert "np.linspace" not in outline
    assert "should not appear" not in outline


def test_unparseable_source_degrades_instead_of_raising():
    # Outlines are prompt context. A syntax error in a half-generated file must
    # not raise into the middle of a sprint.
    outline = get_code_outline("def broken(:\n    pass")
    assert "def broken(:" in outline


def test_empty_source_is_empty():
    assert get_code_outline("") == ""
    assert get_code_outline("   \n  ") == ""
