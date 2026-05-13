import json
import spot

def check_syntax_jsonl(filename):
    errors = 0
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            ltl = data["output"]
            try:
                spot.formula(ltl)
            except Exception:
                errors += 1
    print(f"Errors found: {errors}")

check_syntax_jsonl("rover_ltl_train.jsonl")