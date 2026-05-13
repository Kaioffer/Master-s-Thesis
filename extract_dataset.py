import json
import os
import random
import re

from datasets import load_dataset

# 1. Configuration
DATASET_NAME = "cRick/NL-to-LTL-Synthetic-Dataset"
OUTPUT_FILE = "rover_ltl_train.jsonl"
NUM_ROWS = 50000
RANDOM_SEED = 42
INJECT_ROVER_VARS = False

# Keep this aligned with the variable mapping in Llama_SFT_prompting.py.
# Turn this on only if you want extra practice with quoted arithmetic atoms.
INCLUDE_BOOLEAN_ARITHMETIC_ATOMS = False

# Variables currently exposed in the prompting script.
ROVER_VARS = [
    "battery",
    "chargePosition",
    "recharge",
    "goal",
    "pre_battery",
    "n",
    "plan",
    "length_plan",
    "chargeNeeded_var",
    "batteryFull",
    "currentPosition",
    "initialPosition",
    "currentPhysicalPosition",
    "start",
    "s0",
    "x",
    "y",
    "obstacle",
    "obstaclePhysicallyDetected",
    "isHeatpoint",
    "heatPhysicallyMeasured",
    "isReachable",
    "visited",
    "hottestUnvisited",
    "Obstacle_currentPosition",
    "speed",
    "removeGoalFromSet",
    "atGoal",
    "heatpointLocation",
    "isObstacle",
    "isShortestPath",
    "planContainsObstacle",
    "planHasStart",
]

# Optional quoted atoms for arithmetic-style examples.
BOOLEAN_ARITHMETIC_ATOMS = [
    '"battery > 0"',
    '"battery < 20"',
    '"battery == 100"',
    '"chargePosition == x"',
    '"currentPosition == chargePosition"',
    '"pre_battery - 1/n"',
    '"speed > 5"',
    '"battery <= chargeNeeded_var"',
]

GROUNDING_VOCAB = ROVER_VARS + (
    BOOLEAN_ARITHMETIC_ATOMS if INCLUDE_BOOLEAN_ARITHMETIC_ATOMS else []
)

SYSTEM_INSTRUCTION = (
    "Translate the natural-language requirement into exactly one Spot-compatible LTL formula. "
    "Use only the provided rover variables."
)


def ground_text_and_ltl(nl_text, ltl_formula, var_map):
    """Replace synthetic placeholders with rover variables consistently."""
    for placeholder, rover_val in var_map.items():
        ltl_formula = re.sub(rf"\b{placeholder}\b", rover_val, ltl_formula)
        clean_nl_val = rover_val.replace('"', '').replace('_', ' ')
        nl_text = nl_text.replace(placeholder, clean_nl_val)

    return nl_text, ltl_formula


def write_jsonl_row(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    random.seed(RANDOM_SEED)

    print(f"Loading {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    synthetic_rows = []
    for entry in ds:
        if len(synthetic_rows) >= NUM_ROWS:
            break

        if INJECT_ROVER_VARS:
            found_vars = list(set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{3,}", entry["ltl"])))
            if not found_vars:
                continue

            selected_vars = random.sample(GROUNDING_VOCAB, min(len(found_vars), len(GROUNDING_VOCAB)))
            var_map = dict(zip(found_vars, selected_vars))
            final_nl, final_ltl = ground_text_and_ltl(entry["en"], entry["ltl"], var_map)
        else:
            final_nl, final_ltl = entry["en"], entry["ltl"]

        synthetic_rows.append(
            {
                "instruction": SYSTEM_INSTRUCTION,
                "input": final_nl,
                "output": final_ltl,
            }
        )

        if len(synthetic_rows) % 5000 == 0:
            print(f"Collected synthetic rows: {len(synthetic_rows)}")

    random.shuffle(synthetic_rows)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in synthetic_rows[:NUM_ROWS]:
            write_jsonl_row(f, row)

    print(
        f"Success! Saved {min(NUM_ROWS, len(synthetic_rows))} rows to {OUTPUT_FILE} "
        f"(synthetic-only)."
    )


if __name__ == "__main__":
    main()