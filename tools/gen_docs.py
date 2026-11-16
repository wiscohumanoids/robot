#!/usr/bin/env python3
"""Regenerates the tables in docs/interface_contract.md from the machine-readable contract,
ros2_ws/src/hsim_bringup/config/interface_contract.yaml (the file `contract_check` enforces live).

    python3 tools/gen_docs.py            # rewrite the generated blocks in place
    python3 tools/gen_docs.py --check    # exit 1 if any block is out of date (CI, tests/test_docs.py)

Only text between `<!-- BEGIN GENERATED:<name> -->` and `<!-- END GENERATED:<name> -->` is touched;
everything else in the file is hand-written. Never edit a generated block by hand: edit the YAML and
re-run this script. Needs only PyYAML.
"""
import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'ros2_ws' / 'src' / 'hsim_bringup' / 'config' / 'interface_contract.yaml'
TARGETS = [ROOT / 'docs' / 'interface_contract.md']


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    out += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(out)


def _owners(o):
    return ', '.join(f'`{x}`' for x in (o if isinstance(o, list) else [o]))


def block_topics(c):
    rows = []
    for name, s in sorted(c['topics'].items(), key=lambda kv: (kv[1]['phase'], kv[0])):
        rate = '-' if s.get('rate_hz') is None else f"{s['rate_hz']:g} Hz"
        rows.append((f'`{name}`', f"`{s['type']}`", _owners(s['owner']), rate, s['phase'],
                     'no' if s.get('required') is False else 'yes'))
    return md_table(['Topic', 'Type', 'Owner (only publisher)', 'Rate', 'Phase', 'Required'], rows)


def block_tf(c):
    rows = [(f"`{e['parent']}` → `{e['child']}`", f"`{e['owner']}`",
             {True: 'static', False: 'dynamic', None: 'static or live'}[e['static']], e['phase'])
            for e in c['tf_edges']]
    return md_table(['Edge', 'Owner', 'Kind', 'Phase'], rows)


def _calls(c, key):
    rows = [(f'`{n}`', f"`{s['type']}`", f"`{s['server']}`", ', '.join(f'`{x}`' for x in s['clients']), s['phase'])
            for n, s in sorted(c.get(key, {}).items(), key=lambda kv: (kv[1]['phase'], kv[0]))]
    return md_table(['Name', 'Type', 'Server', 'Clients', 'Phase'], rows)


def block_shared(c):
    return ', '.join(f'`{t}`' for t in c.get('shared_topics', []))


BLOCKS = {'topics': block_topics, 'tf': block_tf, 'actions': lambda c: _calls(c, 'actions'),
          'services': lambda c: _calls(c, 'services'), 'shared': block_shared}


def render(text, contract):
    def repl(m):
        name = m.group(1)
        if name not in BLOCKS:
            raise SystemExit(f'unknown generated block: {name}')
        return f'<!-- BEGIN GENERATED:{name} -->\n{BLOCKS[name](contract)}\n<!-- END GENERATED:{name} -->'
    return re.sub(r'<!-- BEGIN GENERATED:(\w+) -->.*?<!-- END GENERATED:\1 -->', repl, text, flags=re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='exit 1 if a generated block is stale')
    args = ap.parse_args()
    contract = yaml.safe_load(CONTRACT.read_text())
    stale = []
    for path in TARGETS:
        old = path.read_text()
        new = render(old, contract)
        if new != old:
            stale.append(path.relative_to(ROOT))
            if not args.check:
                path.write_text(new)
    if args.check and stale:
        print('out of date (run python3 tools/gen_docs.py):', *stale)
        return 1
    print('updated:' if stale else 'up to date', *stale)
    return 0


if __name__ == '__main__':
    sys.exit(main())
