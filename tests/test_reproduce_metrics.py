from pathlib import Path

import pytest

from scripts.reproduce_paper_metrics import _metric_check, parse_archive_args


def test_parse_archive_args():
    assert parse_archive_args(["WildFake=/tmp/wild.pt", "aigibench=relative.pt"]) == {
        "wildfake": Path("/tmp/wild.pt"),
        "aigibench": Path("relative.pt"),
    }


def test_parse_archive_args_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicated"):
        parse_archive_args(["wildfake=a.pt", "WildFake=b.pt"])


def test_metric_check_uses_absolute_tolerance():
    assert _metric_check(0.9, 0.900004, 5e-6)["passed"]
    assert not _metric_check(0.9, 0.900006, 5e-6)["passed"]
