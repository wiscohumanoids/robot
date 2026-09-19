#!/usr/bin/env python3
"""Regenerates the tables in INTERFACE_CONTRACT.md and STATUS.md from
src/humanoid_interfaces/config/interface_contract.yaml.

    python3 tools/gen_docs.py            # rewrite the generated blocks in place
    python3 tools/gen_docs.py --check    # exit 1 if any block is out of date (CI)

Only text between `<!-- BEGIN GENERATED:name -->` and `<!-- END GENERATED:name -->`
is touched; everything else in those files is hand-written. Never edit a
generated block by hand -- edit the YAML and re-run this script.
Needs only PyYAML.
"""
import argparse
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = os.path.join(ROOT, 'src', 'humanoid_interfaces', 'config', 'interface_contract.yaml')
STATUS_LABEL = {'real': 'REAL', 'stub': 'STUB', 'scaffold': 'SCAFFOLD',
                'external': 'EXTERNAL', 'missing': 'MISSING'}


def load():
    with open(CONTRACT) as f:
        return yaml.safe_load(f)


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    out += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(out)


def switch(node):
    arg = node.get('launch_arg')
    if not arg:
        return '-'
    if node['status'] in ('stub', 'scaffold'):
        return f'`{arg}:=stub|external`'
    return f'`{arg}:=true|false`'


def block_summary(c):
    counts = {}
    for n in c['nodes']:
        counts[n['status']] = counts.get(n['status'], 0) + 1
    return md_table(['Status', 'Nodes', 'Meaning'], [
        ('**REAL**', counts.get('real', 0), 'Complete for its purpose today.'),
        ('**STUB**', counts.get('stub', 0), 'Correct interface, fake internals -- the internals are what a team builds.'),
        ('**SCAFFOLD**', counts.get('scaffold', 0), 'Correct structure/interface, hardware I/O stubbed.'),
        ('**EXTERNAL**', counts.get('external', 0), 'Third-party ROS node used as-is (configure only).'),
        ('**MISSING**', counts.get('missing', 0), 'Nothing exists yet; the contract reserves the interface.'),
    ])


def block_nodes(c):
    parts = []
    for layer in c['layers']:
        nodes = [n for n in c['nodes'] if n['layer'] == layer['id']]
        if not nodes:
            continue
        rows = []
        for n in nodes:
            rows.append((
                f"`{n['id']}`", f"**{STATUS_LABEL[n['status']]}**",
                f"`{n['package']}`" if n.get('package') else '-',
                n['summary'], n.get('replaces') or '-', n['area'], switch(n)))
        parts.append(f"#### {layer['title']}\n\n" + md_table(
            ['Node', 'Status', 'Package', 'What it does today', 'Real thing that replaces it', 'Area', 'Launch switch'],
            rows))
    return '\n\n'.join(parts)


def block_areas(c):
    areas = {}
    for n in c['nodes']:
        areas.setdefault(n['area'], []).append(n)
    rows = []
    for area in sorted(areas):
        ns = areas[area]
        rows.append((f'**{area}**',
                     ', '.join(f"`{n['id']}` ({STATUS_LABEL[n['status']].lower()})" for n in ns)))
    return md_table(['Area', 'Nodes it owns (status)'], rows)


def rate_text(t):
    if t['check'] == 'rate':
        text = f"{t['rate_hz']} Hz"
    elif t['check'] == 'latched':
        text = 'latched'
    else:
        text = 'event'
    return text + (' (sim only)' if t.get('sim_only') else '')


def block_topics(c):
    rows = [(f"`{t['name']}`", f"`{t['type']}`", f"`{t['owner']}`",
             ', '.join(f'`{x}`' for x in t['consumers']) or '-', rate_text(t))
            for t in c['topics']]
    return md_table(['Topic', 'Type', 'Owner (sole publisher)', 'Consumers', 'Rate'], rows)


def block_actions(c):
    rows = [(f"`{a['name']}`", f"`{a['type']}`", f"`{a['server']}`",
             ', '.join(f'`{x}`' for x in a['clients'])) for a in c['actions']]
    return md_table(['Action', 'Type', 'Server', 'Clients'], rows)


def block_tf(c):
    rows = [(f"`{e['parent']}` -> `{e['child']}`", f"`{e['owner']}`",
             'static' if e.get('static') else f"{e['rate_hz']} Hz", e.get('note', '')) for e in c['tf']]
    return md_table(['TF edge', 'Owner (sole publisher)', 'Rate', 'Note'], rows)


def block_frames(c):
    return md_table(['Frame', 'Meaning'], [(f'`{k}`', v) for k, v in c['frames'].items()])


BLOCKS = {
    'summary': block_summary, 'nodes': block_nodes, 'areas': block_areas,
    'topics': block_topics, 'actions': block_actions, 'tf': block_tf, 'frames': block_frames,
}
TARGETS = {
    'INTERFACE_CONTRACT.md': ['frames', 'topics', 'actions', 'tf'],
    'STATUS.md': ['summary', 'nodes', 'areas'],
}


def render(text, names, contract, path):
    for name in names:
        pattern = re.compile(
            rf'(<!-- BEGIN GENERATED:{name} -->\n).*?(<!-- END GENERATED:{name} -->)', re.S)
        if not pattern.search(text):
            raise SystemExit(f'{path}: missing GENERATED:{name} markers')
        body = BLOCKS[name](contract)
        text = pattern.sub(lambda m: m.group(1) + body + '\n' + m.group(2), text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    contract = load()
    stale = []
    for filename, names in TARGETS.items():
        path = os.path.join(ROOT, filename)
        with open(path) as f:
            old = f.read()
        new = render(old, names, contract, filename)
        if new != old:
            stale.append(filename)
            if not args.check:
                with open(path, 'w') as f:
                    f.write(new)
    if args.check and stale:
        print('out of date (run `python3 tools/gen_docs.py`): ' + ', '.join(stale))
        return 1
    if stale:
        print('updated: ' + ', '.join(stale))
    return 0


if __name__ == '__main__':
    sys.exit(main())
