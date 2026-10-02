import pytest

from ber.metrics import f05, macro_f05


def test_empty_empty():
    assert f05(set(), set()) == 1.0


def test_singleton_false_merge():
    assert f05({"S2-1"}, set()) == 0.0


def test_missed_all():
    assert f05(set(), {"S2-1"}) == 0.0


def test_extra_ids_example_from_spec():
    assert f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"}) == pytest.approx(0.714, abs=1e-3)


def test_missing_ids():
    # p=1, r=0.5 -> 1.25*0.5/(0.25+0.5)=0.8333
    assert f05({"a"}, {"a", "b"}) == pytest.approx(0.8333, abs=1e-4)


def test_macro_includes_singletons():
    truth = {"x": set(), "y": {"a", "b"}}
    assert macro_f05({"y": {"a", "b"}}, truth) == 1.0
    assert macro_f05({"x": {"a"}, "y": {"a", "b"}}, truth) == 0.5
