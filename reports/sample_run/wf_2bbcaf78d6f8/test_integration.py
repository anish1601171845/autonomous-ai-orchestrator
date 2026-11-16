import importlib

import pytest

MODULES = ['database', 'login', 'project_config', 'registration', 'student_dashboard']


@pytest.mark.parametrize('name', MODULES)
def test_module_imports_and_exposes_api(name):
    mod = importlib.import_module(name)
    public = [n for n in dir(mod) if not n.startswith('_')]
    assert public, f'{name} exposes no public API'
