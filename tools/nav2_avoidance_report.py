#!/usr/bin/env python3
"""Build a standalone HTML replay from the run_<map>.json files avoidance_test.py saved.

    python3 tools/nav2_avoidance_report.py reports/nav2_avoidance

Reads run_*.json in that directory, writes report.html next to them. Plain
Python, no ROS: open the HTML in any browser.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">' \
       '<meta name="viewport" content="width=device-width, initial-scale=1"></head><body>\n{}\n</body></html>\n'


def main(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, 'run_*.json')))
    if not files:
        print(f'no run_*.json in {out_dir}', file=sys.stderr)
        return 1
    runs = [json.load(open(f)) for f in files]
    order = ['obstacle_1', 'obstacle_2', 'obstacle_3', 'zigzag', 'unreachable']
    runs.sort(key=lambda r: (r.get('goal_yaw') is not None, order.index(r['map']) if r['map'] in order else 99,
                             r.get('goal_yaw') or 0.0))
    template = open(os.path.join(HERE, 'nav2_avoidance_report_template.html')).read()
    path = os.path.join(out_dir, 'report.html')
    with open(path, 'w') as f:
        f.write(PAGE.format(template.replace('__RUNS__', json.dumps(runs, separators=(',', ':')))))
    print(path)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'reports/nav2_avoidance'))
