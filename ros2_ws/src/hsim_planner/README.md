# hsim_planner — LLM task planner (Claude)

Node `planner`: `/planner/command` (your sentence) + `/world/summary` (object memory) → Claude (`claude-opus-5-5`,
low effort, structured JSON) → plan check → `/behavior/execute_plan`; replans up to 2× with the failure reason.
Replies on `/planner/reply`, progress on `/planner/status`. "stop" cancels without calling Claude.

Needs `ANTHROPIC_API_KEY` in the repo's `.env` (copy `.env.example`), then recreate the container
(`tools/ros.sh down && tools/ros.sh up`).

Talk to it: `tools/ask.sh "pick up the red block"` (Mac) or `ros2 run hsim_planner ask "..."` (container).
Tests without the API: `pytest src/hsim_planner/test`, `tools/eval/fake_claude_flow.py` (scripted Claude).
