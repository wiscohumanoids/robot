"""Discover versioned arm calibration files alongside session and validation reports."""

from pathlib import Path
import re


_VERSION = re.compile(r'\d{4}-\d{2}-\d{2}_v(\d+)\.yaml')


def calibration_versions(directory):
    versions = []
    for path in Path(directory).glob('*_v*.yaml'):
        match = _VERSION.fullmatch(path.name)
        if match and path.is_file():
            versions.append((int(match[1]), str(path)))
    return [path for _, path in sorted(versions)]
