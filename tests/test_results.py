"""Tests for the results module."""
from laser_beam_analysis.results import BeamAnalysisResult


def test_default_construction():
    r = BeamAnalysisResult()
    assert r.raw_image is None
    assert r.aligned_image is None
    assert r.alignment == {}
    assert r.profiles_basex == []
    assert r.audit == {}


def test_no_shared_mutable_defaults():
    a = BeamAnalysisResult()
    b = BeamAnalysisResult()
    a.alignment["x"] = 1
    assert b.alignment == {}
    a.profiles_basex.append({"k": 1})
    assert b.profiles_basex == []


def test_set_and_read_back():
    r = BeamAnalysisResult()
    r.audit = {"status": "ok"}
    r.metrology = {"pixel_size_um": 2.0}
    assert r.audit == {"status": "ok"}
    assert r.metrology == {"pixel_size_um": 2.0}