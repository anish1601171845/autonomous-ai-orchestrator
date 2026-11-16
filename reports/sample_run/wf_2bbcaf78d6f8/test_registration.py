import pytest

from registration import AuthError, UserService


@pytest.fixture
def svc():
    return UserService(iterations=1000)


def test_register_and_login(svc):
    user = svc.register("alice", "alice@example.com", "s3cretpass")
    assert user.password_hash != "s3cretpass"
    token = svc.authenticate("alice", "s3cretpass")
    assert svc.user_for_token(token).email == "alice@example.com"


def test_duplicate_registration_rejected(svc):
    svc.register("bob", "bob@example.com", "passw0rd!")
    with pytest.raises(AuthError):
        svc.register("BOB", "other@example.com", "passw0rd!")
    with pytest.raises(AuthError):
        svc.register("bobby", "bob@example.com", "passw0rd!")


@pytest.mark.parametrize("email", ["no-at-sign", "a@b", ""])
def test_invalid_email(svc, email):
    with pytest.raises(AuthError):
        svc.register("carol", email, "passw0rd1")


@pytest.mark.parametrize("pw", ["short1", "lettersonly", "12345678"])
def test_weak_password(svc, pw):
    with pytest.raises(AuthError):
        svc.register("dave", "dave@example.com", pw)


def test_wrong_password_and_logout(svc):
    svc.register("erin", "erin@example.com", "goodpass9")
    with pytest.raises(AuthError):
        svc.authenticate("erin", "badpass9")
    token = svc.authenticate("erin", "goodpass9")
    svc.logout(token)
    with pytest.raises(AuthError):
        svc.user_for_token(token)
