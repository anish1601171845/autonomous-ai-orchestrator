import pytest

from student_dashboard import group_counts, group_mean, kpis, top_n

DATA = [{"group": "a", "value": 10}, {"group": "b", "value": 20}, {"group": "a", "value": 30}, {"group": "c", "value": None}]


def test_kpis():
    k = kpis(DATA, "value")
    assert k["count"] == 4 and k["valid"] == 3
    assert k["total"] == 60 and k["mean"] == 20 and k["median"] == 20


def test_kpis_empty():
    assert kpis([], "value")["valid"] == 0


def test_grouping():
    assert group_counts(DATA, "group") == {"a": 2, "b": 1, "c": 1}
    assert group_mean(DATA, "group", "value") == {"a": 20.0, "b": 20.0}


def test_top_n():
    assert [r["value"] for r in top_n(DATA, "value", 2)] == [30, 20]
    with pytest.raises(ValueError):
        top_n(DATA, "value", -1)
