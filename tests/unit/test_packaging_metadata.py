"""
Packaging metadata compliance tests for the WhiteBox XAI SDK.

Guards against regressions in pyproject.toml specification:
- SPDX license string format (PEP 621 / PEP 639)
- setuptools>=77.0 build requirement
- Absence of deprecated 'License ::' classifiers
- Presence of license-files declaration and referenced file
"""

from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

# Repository root: tests/unit/ -> tests/ -> <root>, where pyproject.toml lives.
SDK_DIR = Path(__file__).resolve().parent.parent.parent
PYPROJECT_PATH = SDK_DIR / "pyproject.toml"


def _load_pyproject() -> dict:
    with open(PYPROJECT_PATH, "rb") as f:
        return tomllib.load(f)


def test_build_system_requires_modern_setuptools():
    data = _load_pyproject()
    build_requires = data.get("build-system", {}).get("requires", [])
    assert any(
        req.startswith("setuptools>=") and int(req.split(">=")[1].split(".")[0]) >= 77
        for req in build_requires
    ), f"Expected setuptools>=77.0 in build-system.requires, got: {build_requires}"


def test_project_license_is_spdx_string():
    data = _load_pyproject()
    project = data.get("project", {})
    license_field = project.get("license")
    assert isinstance(
        license_field, str
    ), f"project.license should be an SPDX string (e.g. 'MIT'), got {type(license_field)}: {license_field}"
    assert license_field == "MIT"


def test_project_license_files_declared_and_exist():
    data = _load_pyproject()
    project = data.get("project", {})
    license_files = project.get("license-files")
    assert isinstance(
        license_files, list
    ), f"project.license-files must be a list, got {type(license_files)}: {license_files}"
    assert "LICENSE" in license_files
    for filename in license_files:
        assert (
            SDK_DIR / filename
        ).is_file(), f"Declared license file '{filename}' not found in {SDK_DIR}"


def test_no_deprecated_license_classifiers():
    data = _load_pyproject()
    classifiers = data.get("project", {}).get("classifiers", [])
    license_classifiers = [c for c in classifiers if c.startswith("License ::")]
    assert (
        not license_classifiers
    ), f"License classifiers are deprecated alongside SPDX license expression; found: {license_classifiers}"
