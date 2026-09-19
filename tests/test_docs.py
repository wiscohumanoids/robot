"""Documentation hygiene: every relative link in the repo's own markdown resolves
(files and #heading anchors), every package has a README, and no doc still
mentions removed interfaces. Cheap insurance against the docs rotting."""
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
VENDORED = {'mujoco_ros2_control'}   # third-party; carries its own docs/licence

LINK = re.compile(r'(?<!\!)\[[^\]]*\]\(([^)\s]+)\)')


def _own_markdown():
    files = glob.glob(os.path.join(ROOT, '*.md'))
    for pkg in os.listdir(SRC):
        if pkg not in VENDORED:
            files += glob.glob(os.path.join(SRC, pkg, 'README.md'))
    return sorted(files)


def _slug(heading):
    text = re.sub(r'[`*_]{1}', lambda m: '_' if m.group(0) == '_' else '', heading.strip().lower())
    text = re.sub(r'[^\w\- ]', '', text)          # GitHub keeps word chars, hyphens, spaces
    return text.replace(' ', '-')


def _anchors(path):
    out = set()
    in_fence = False
    for line in open(path):
        if line.startswith('```'):
            in_fence = not in_fence
        m = None if in_fence else re.match(r'#{1,6}\s+(.*)', line)
        if m:
            out.add(_slug(m.group(1)))
    return out


def test_relative_links_and_anchors_resolve():
    problems = []
    for md in _own_markdown():
        text = open(md).read()
        text = re.sub(r'```.*?```', '', text, flags=re.S)   # ignore code blocks
        for target in LINK.findall(text):
            if re.match(r'[a-z]+://|mailto:', target):
                continue
            path, _, anchor = target.partition('#')
            resolved = md if not path else os.path.normpath(os.path.join(os.path.dirname(md), path))
            if not os.path.exists(resolved):
                problems.append(f'{os.path.relpath(md, ROOT)}: broken link {target}')
            elif anchor and resolved.endswith('.md') and anchor not in _anchors(resolved):
                problems.append(f'{os.path.relpath(md, ROOT)}: no heading for anchor {target}')
    assert not problems, '\n'.join(problems)


def test_every_package_has_a_readme():
    for pkg in sorted(os.listdir(SRC)):
        if pkg in VENDORED or not os.path.isdir(os.path.join(SRC, pkg)):
            continue
        assert os.path.isfile(os.path.join(SRC, pkg, 'README.md')), f'{pkg} has no README.md'


def test_no_doc_still_documents_the_removed_velocity_command():
    for md in _own_markdown():
        for n, line in enumerate(open(md), 1):
            if 'VelocityCommand' in line:
                assert re.search(r'earlier|replaced|removed|custom', line, re.I), (
                    f'{os.path.relpath(md, ROOT)}:{n} still documents VelocityCommand as live: {line.strip()}')
