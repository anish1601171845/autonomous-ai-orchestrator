import pytest

from database import Database


@pytest.fixture
def db():
    d = Database()
    d.create_table("records", {"name": "TEXT NOT NULL", "score": "REAL"})
    yield d
    d.close()


def test_crud_roundtrip(db):
    rid = db.insert("records", {"name": "alpha", "score": 9.5})
    assert db.get("records", rid)["name"] == "alpha"
    assert db.update("records", rid, {"score": 7.0})
    assert db.get("records", rid)["score"] == 7.0
    assert db.delete("records", rid)
    assert db.get("records", rid) is None


def test_list_with_filters(db):
    db.insert("records", {"name": "a", "score": 1})
    db.insert("records", {"name": "b", "score": 2})
    db.insert("records", {"name": "a", "score": 3})
    assert [r["score"] for r in db.list("records", name="a")] == [1, 3]


def test_rejects_unsafe_identifiers(db):
    with pytest.raises(ValueError):
        db.insert("records; DROP TABLE records", {"name": "x"})
    with pytest.raises(ValueError):
        db.create_table("t", {"c": "TEXT); DROP TABLE records; --"})


def test_not_null_enforced(db):
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        db.insert("records", {"score": 1})
