import pytest

from project_config import AppConfig, load_config


def test_defaults():
    c = load_config({})
    assert c == AppConfig()


def test_env_overrides_and_coercion():
    c = load_config({"APP_PORT": "9000", "APP_DEBUG": "true", "APP_APP_NAME": "demo"})
    assert c.port == 9000 and c.debug is True and c.app_name == "demo"


@pytest.mark.parametrize("env", [{"APP_PORT": "0"}, {"APP_ENVIRONMENT": "qa"}, {"APP_DEBUG": "maybe"},
                                 {"APP_ENVIRONMENT": "production", "APP_DEBUG": "1"}])
def test_invalid(env):
    with pytest.raises(ValueError):
        load_config(env)
