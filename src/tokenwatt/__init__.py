from importlib.metadata import PackageNotFoundError, version as _pkg_version
import pathlib

_version_file = pathlib.Path(__file__).resolve().parents[2] / "VERSION"
if _version_file.is_file():
    __version__ = _version_file.read_text().strip()
else:  # installed as a wheel without the repo VERSION file
    try:
        __version__ = _pkg_version("tokenwatt")
    except PackageNotFoundError:
        __version__ = "0+unknown"
