import json
from pathlib import Path
import re

import pandas as pd


def _normalize_ftltl_to_spot(formula: str) -> str:
	text = (formula or "").strip()
	if not text:
		return ""

	# Normalize common function-like predicates to atomic variable names.
	text = text.replace("chargeNeeded(plan)", "chargeNeeded_var")
	text = text.replace("length(plan)", "length_plan")
	text = text.replace("Obstacle(currentPosition)", "Obstacle_currentPosition")

	# Spot textual atomic propositions should use == for equality.
	text = re.sub(r"(?<![<>=!])=(?!=)", "==", text)

	def _top_level_contains(s: str, token: str) -> bool:
		depth = 0
		i = 0
		while i < len(s):
			ch = s[i]
			if ch == '"':
				i += 1
				while i < len(s) and s[i] != '"':
					i += 1
			elif ch == '(':
				depth += 1
			elif ch == ')':
				depth = max(0, depth - 1)
			elif depth == 0 and s.startswith(token, i):
				return True
			i += 1
		return False

	def _paren_spans(s: str) -> list[tuple[int, int]]:
		stack = []
		spans = []
		for i, ch in enumerate(s):
			if ch == '(':
				stack.append(i)
			elif ch == ')' and stack:
				start = stack.pop()
				spans.append((start, i))
		return spans

	def _is_comparison_predicate(content: str) -> bool:
		inner = content.strip()
		if not inner or '"' in inner:
			return False

		if re.match(r"^[GFX!]\b", inner):
			return False

		has_comparison = any(_top_level_contains(inner, op) for op in ["==", "!=", "<=", ">=", "<", ">"])
		if not has_comparison:
			return False

		has_boolean = any(_top_level_contains(inner, op) for op in ["->", "<->", "&", "|"])
		has_temporal_binary = any(_top_level_contains(inner, op) for op in [" U ", " W ", " R ", " M "])
		return not has_boolean and not has_temporal_binary

	for _ in range(6):
		spans = _paren_spans(text)
		candidates = []
		for start, end in spans:
			content = text[start + 1:end]
			if _is_comparison_predicate(content):
				candidates.append((start, end))

		if not candidates:
			break

		# Keep only maximal candidates (do not quote nested fragments when outer predicate is quoted).
		candidates.sort(key=lambda x: (x[0], -(x[1] - x[0])))
		maximal = []
		for s, e in candidates:
			if not any(ms <= s and e <= me for ms, me in maximal):
				maximal.append((s, e))

		new_text = text
		for start, end in sorted(maximal, key=lambda x: x[0], reverse=True):
			inner = re.sub(r"\s+", " ", new_text[start + 1:end].strip())
			replacement = f'("{inner}")'
			new_text = new_text[:start] + replacement + new_text[end + 1:]

		if new_text == text:
			break
		text = new_text

	# Small cleanup for redundant wrappers around quoted APs.
	text = re.sub(r"\(\s*\((\s*\"[^\"]+\"\s*)\)\s*\)", r"(\1)", text)

	# Minimal canonicalization for Spot's preferred style.
	text = re.sub(r"\s+", " ", text).strip()
	return text


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	input_file = base_dir / "Rover system requirements.xlsx"
	output_file = base_dir / "requirements.json"

	df = pd.read_excel(input_file)

	required_columns = ["ID", "NL description", "Translation of FRET into ftLTL"]
	missing = [col for col in required_columns if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")

	rows = []
	for _, row in df.iterrows():
		item_id = str(row["ID"]).strip() if pd.notna(row["ID"]) else ""
		description = str(row["NL description"]).strip() if pd.notna(row["NL description"]) else ""
		raw_benchmark_ltl = (
			str(row["Translation of FRET into ftLTL"]).strip()
			if pd.notna(row["Translation of FRET into ftLTL"])
			else ""
		)
		benchmark_ltl = _normalize_ftltl_to_spot(raw_benchmark_ltl)

		# Skip fully empty lines.
		if not item_id and not description and not benchmark_ltl:
			continue

		rows.append(
			{
				"id": item_id,
				"description": description,
				"benchmark_ltl": benchmark_ltl,
			}
		)

	with output_file.open("w", encoding="utf-8") as f:
		json.dump(rows, f, indent=2, ensure_ascii=False)

	print(f"Wrote {len(rows)} items to {output_file}")


if __name__ == "__main__":
	main()
