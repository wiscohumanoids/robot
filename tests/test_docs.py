"""Documentation hygiene: cheap insurance against the docs rotting.
  - every relative link in the repo's own markdown resolves (files and #anchors)
  - every ROS package has a README
  - generated doc blocks are up to date (tools/gen_docs.py)

    python -m pytest tests -q
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'ros2_ws' / 'src'
LINK = re.compile(r'\[[^\]]*\]\(([^)\s]+)\)')
SKIP_DIRS = ('.venv', 'node_modules', 'ros2_ws/build', 'ros2_ws/install', 'ros2_ws/log')


def _markdown():
    out = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '*.md'],
                         cwd=ROOT, capture_output=True, text=True).stdout.split('\n')
    return [ROOT / f for f in out if f and not f.startswith(SKIP_DIRS)]


def _slug(heading):
    """GitHub's anchor rule: lowercase, drop punctuation except - and _, spaces -> -."""
    h = re.sub(r'[`*_]', '', heading.strip().lower())
    return re.sub(r'[^\w\- ]', '', h).replace(' ', '-')


def _anchors(path):
    text = re.sub(r'```.*?```', '', path.read_text(errors='ignore'), flags=re.S)
    return {_slug(m.group(1)) for m in re.finditer(r'^#+\s+(.*)$', text, flags=re.M)}


def test_relative_links_and_anchors_resolve():
    broken = []
    for md in _markdown():
        text = re.sub(r'```.*?```', '', md.read_text(errors='ignore'), flags=re.S)
        for m in LINK.finditer(text):
            url = m.group(1)
            if re.match(r'^[a-z][a-z0-9+.-]*:', url):      # http:, https:, mailto: ...
                continue
            path, _, anchor = url.partition('#')
            target = (md.parent / path) if path else md
            if not target.exists():
                broken.append(f'{md.relative_to(ROOT)}: {url}')
            elif anchor and target.suffix == '.md' and not anchor.startswith('L') and anchor not in _anchors(target):
                broken.append(f'{md.relative_to(ROOT)}: {url} (no such heading)')
    assert not broken, 'broken links:\n  ' + '\n  '.join(broken)


def test_every_ros_package_has_a_readme():
    missing = [p.parent.name for p in SRC.glob('*/package.xml') if not (p.parent / 'README.md').exists()]
    assert not missing, f'ROS packages without a README.md: {missing}'


def test_generated_docs_are_current():
    r = subprocess.run([sys.executable, str(ROOT / 'tools' / 'gen_docs.py'), '--check'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
