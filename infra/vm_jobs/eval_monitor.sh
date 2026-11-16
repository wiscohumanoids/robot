#!/usr/bin/env bash
# Every 15 min: behaviour-test the newest J1c (G1) and J3c (Wisco) walking checkpoints, arms still and random.
# One summary line per test appended to ~/hsim_runs/_diag/eval_history.log. Run in tmux hsim-mon-eval.
cd ~/humanoid-sim-stack/training/locomotion || exit 1
LOG=~/hsim_runs/_diag/eval_history.log; mkdir -p ~/hsim_runs/_diag
E=../../infra/vm_jobs/eval_velocity.py
declare -A done_ck
while true; do
  for spec in "J1j_g1_vel_arms_free|Hsim-G1-Velocity-Flat-ArmsFree|.|hsim_tasks" \
              ; do
    IFS='|' read -r job task ipath imod <<<"$spec"
    ck=$(find ~/hsim_runs/$job -name "model_*.pt" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d" " -f2-)   # newest by time (resume dirs also hold the copied start checkpoint)
    [ -z "$ck" ] && continue
    [ "${done_ck[$job]}" == "$ck" ] && continue
    for arms in default random; do
      # Wisco play mode holds curriculum level 1 (1 kg payload, full arm range) unless WISCO_PLAY_LEVEL is set:
      # arms still -> level 0 (pure walking skill), arms random -> level 1 (full difficulty)
      lvl=1; [ "$arms" == "default" ] && lvl=0
      out=$(WISCO_PLAY_LEVEL=$lvl WISCO_PLAY_LEVEL2=0 env -u LD_LIBRARY_PATH timeout 900 .venv/bin/python $E "$ck" --arms $arms --num-envs 32 --task $task \
            --import-path $ipath --import-module $imod --json ~/hsim_runs/_diag/eval_${job}_$(basename $ck .pt)_$arms.json 2>&1 \
            | grep -E "^SUMMARY|forward 0.5|turn 0.8|sidestep")
      echo "$(date -u +%H:%M) $job $(basename $ck) arms=$arms | $(echo "$out" | tr '\n' '|' | tr -s ' ')" >> $LOG
    done
    # in-place turns from standstill (arms moving, full difficulty)
    out=$(WISCO_PLAY_LEVEL=1 WISCO_PLAY_LEVEL2=0 env -u LD_LIBRARY_PATH timeout 900 .venv/bin/python $E "$ck" --script turns --arms random --num-envs 32 --task $task \
          --import-path $ipath --import-module $imod --json ~/hsim_runs/_diag/eval_${job}_$(basename $ck .pt)_turns.json 2>&1 \
          | grep -E "^(left|right) " | awk '{printf "%s %s->%s, ", $1, $2, $8}')
    echo "$(date -u +%H:%M) $job $(basename $ck) TURNS | $out" >> $LOG
    done_ck[$job]=$ck
  done
  sleep 900
done
