"""The labelling script must not lose what he has already labelled."""

import importlib.util
import pathlib

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "label_speakers.py"


def _module():
    spec = importlib.util.spec_from_file_location("label_speakers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_clip_with_no_similarity_does_not_crash_the_session():
    """Sep 26 2026: float('') on the fourteenth clip ended the run unsaved."""
    row = {"created_at": "", "similarity": "", "duration_s": "3.10"}
    assert "sim=n/a" in _module().describe(row, 14, 14)


def test_a_scored_clip_shows_its_similarity():
    row = {"created_at": "2026-09-26T00:30:36", "similarity": "0.7760", "duration_s": "3.75"}
    assert _module().describe(row, 3, 14) == "[3/14] 00:30:36  sim=0.776  3.75s"


def test_each_label_is_saved_when_it_is_given():
    source = SCRIPT.read_text()
    block = source.split('row["label"] = CHOICES[answer]')[1].split("break")[0]
    assert "save(rows)" in block
