import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


# Variable mapping copied from the "gemini_prompting" mapping table.
MAPPING_VARIABLES = {
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
}


# Tokens that can appear in formulas but are not variables.
LTL_RESERVED = {
	"G",
	"F",
	"X",
	"U",
	"W",
	"R",
	"M",
	"true",
	"false",
}


def _normalize_name(name: str) -> str:
	return "".join(ch for ch in name.lower() if ch.isalnum())


def _name_similarity(a: str, b: str) -> float:
	na = _normalize_name(a)
	nb = _normalize_name(b)
	if not na or not nb:
		return 0.0
	if na == nb:
		return 1.0
	# Avoid noisy substring matches from very short names (for example "n").
	if min(len(na), len(nb)) >= 4 and (na in nb or nb in na):
		return 0.9
	return SequenceMatcher(None, na, nb).ratio()


def load_requirements(path: Path) -> list:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError("Expected requirements JSON to be a list of requirement objects")
	return data


def extract_variables_from_formula(formula: str) -> set:
	# Capture identifiers both inside and outside quoted comparisons.
	tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula or "")
	return {t for t in tokens if t not in LTL_RESERVED}


def collect_used_variables(requirements: list) -> tuple[set, dict]:
	used_variables = set()
	usage_by_requirement = {}

	for item in requirements:
		req_id = item.get("id", "UNKNOWN")
		formula = item.get("benchmark_ltl", "")
		vars_in_formula = extract_variables_from_formula(formula)
		usage_by_requirement[req_id] = sorted(vars_in_formula)
		used_variables.update(vars_in_formula)

	return used_variables, usage_by_requirement


def find_similar_names(source_names: set, target_names: set, threshold: float = 0.74) -> dict:
	suggestions = {}
	for name in sorted(source_names):
		candidates = []
		for target in target_names:
			score = _name_similarity(name, target)
			if score >= threshold:
				candidates.append((target, score))
		candidates.sort(key=lambda x: x[1], reverse=True)
		if candidates:
			suggestions[name] = candidates[:3]
	return suggestions


def print_report(
	used_variables: set,
	used_not_mapped: set,
	mapped_not_used: set,
	similar_from_used: dict,
	similar_from_mapped: dict,
):
	print("=" * 72)
	print("Variable Consistency Report")
	print("=" * 72)
	print(f"Total variables used in benchmark_ltl formulas: {len(used_variables)}")
	print(f"Total variables in mapping list: {len(MAPPING_VARIABLES)}")

	print("\n1) Variables used in specifications but NOT in mapping list")
	if used_not_mapped:
		for var in sorted(used_not_mapped):
			print(f"  - {var}")
	else:
		print("  - None")

	print("\n2) Variables in mapping list but NOT used in specifications")
	if mapped_not_used:
		for var in sorted(mapped_not_used):
			print(f"  - {var}")
	else:
		print("  - None")

	print("\n3) Similar-name candidates (possible naming mismatch)")
	if not similar_from_used and not similar_from_mapped:
		print("  - None")
		return

	if similar_from_used:
		print("  A) Unmapped used variables that look similar to mapped names:")
		for src, matches in similar_from_used.items():
			formatted = ", ".join(f"{m} ({s:.2f})" for m, s in matches)
			print(f"    - {src} -> {formatted}")

	if similar_from_mapped:
		print("  B) Unused mapped variables that look similar to used names:")
		for src, matches in similar_from_mapped.items():
			formatted = ", ".join(f"{m} ({s:.2f})" for m, s in matches)
			print(f"    - {src} -> {formatted}")


def main():
	parser = argparse.ArgumentParser(
		description=(
			"Compare variables used in benchmark_ltl formulas with a predefined variable mapping list."
		)
	)
	parser.add_argument(
		"--requirements",
		default=str(Path(__file__).with_name("requirements2.json")),
		help="Path to requirements.json",
	)
	args = parser.parse_args()

	requirements_path = Path(args.requirements)
	if not requirements_path.exists():
		raise FileNotFoundError(f"requirements file not found: {requirements_path}")

	requirements = load_requirements(requirements_path)
	used_variables, _usage_by_requirement = collect_used_variables(requirements)

	used_not_mapped = used_variables - MAPPING_VARIABLES
	mapped_not_used = MAPPING_VARIABLES - used_variables

	similar_from_used = find_similar_names(used_not_mapped, MAPPING_VARIABLES)
	similar_from_mapped = find_similar_names(mapped_not_used, used_variables)

	print_report(
		used_variables,
		used_not_mapped,
		mapped_not_used,
		similar_from_used,
		similar_from_mapped,
	)


if __name__ == "__main__":
	main()
