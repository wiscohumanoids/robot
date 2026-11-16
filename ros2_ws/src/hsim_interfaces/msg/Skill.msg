# One step of a task plan. The planner (LLM) fills these with names only, never coordinates.
string name          # navigate_to | find_object | search_for | look_around | pick | place | say
string target        # object description, e.g. "red block" (empty if the skill has none)
string[] hints       # optional, e.g. places to search in order: ["table", "counter"]
string text          # for say: what to tell the user
