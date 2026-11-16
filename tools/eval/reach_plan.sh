#!/usr/bin/env bash
# GRADER: run a hand-written plan, then check the robot ended within arm's reach of OBJECT.
#   tools/eval/reach_plan.sh ENV OBJECT_NAME <plan steps as for run_plan.py...>
E=$(dirname "$0"); env=$1 obj=$2; shift 2
result=$(python3 "$E/run_plan.py" "$@" | tail -1)
if [[ "$result" != *SUCCESS* ]]; then echo "FAIL: plan did not succeed: $result"; exit 1; fi
python3 "$E/reach_check.py" "$env" "$obj"
