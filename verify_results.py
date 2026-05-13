import json
import spot
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_SPOT_READY = False


def _ensure_spot_setup():
    global _SPOT_READY
    if not _SPOT_READY:
        spot.setup()
        _SPOT_READY = True


def verify_results(llm_output, benchmark):
    """Return (is_correct, error_type) for one LTL prediction.

    error_type values:
    - None when logically equivalent
    - "wrong_logic" when both formulas parse but are not equivalent
    - "syntax_error" when either formula fails to parse
    """
    _ensure_spot_setup()

    try:
        f_benchmark = spot.formula(benchmark)
        f_llm = spot.formula(llm_output)
    except Exception:
        return False, "syntax_error"

    if spot.are_equivalent(f_benchmark, f_llm):
        return True, None

    return False, "wrong_logic"


def logical_closeness(gen_str, gold_str):
    """Return (is_subsumed, is_generalized) for generated vs benchmark formulas.

    - is_subsumed: generated implies benchmark (generated is stricter/more specific)
    - is_generalized: benchmark implies generated (generated is broader/more general)

    Returns (None, None) if parsing fails.
    """
    _ensure_spot_setup()

    try:
        # Validate both formulas first so parse failures map cleanly to None values.
        spot.formula(gen_str)
        spot.formula(gold_str)
    except Exception:
        return None, None

    # Some Spot builds do not expose spot.implies(...), so use:
    # A => B  iff  (A -> B) is equivalent to true (1).
    is_subsumed = _implies(gen_str, gold_str)
    is_generalized = _implies(gold_str, gen_str)
    return is_subsumed, is_generalized


def _implies(lhs_str, rhs_str):
    implication = f"({lhs_str}) -> ({rhs_str})"
    return bool(spot.are_equivalent(spot.formula(implication), spot.formula("1")))


def logical_closeness_score(is_subsumed, is_generalized):
    """Map logical relation to a normalized closeness score.

    - 1.0 when formulas are equivalent (both implications true)
    - 0.5 when only one implication holds
    - 0.0 when neither implication holds
    - None when relation is not available (e.g., parse error)
    """
    if is_subsumed is None or is_generalized is None:
        return None

    if is_subsumed and is_generalized:
        return 1.0
    if is_subsumed or is_generalized:
        return 0.5
    return 0.0


def verify_results_file(input_file):
    _ensure_spot_setup()

    with open(os.path.join(BASE_DIR, input_file), 'r') as f:
        results = json.load(f)

    total = len(results)
    correct = 0
    syntax_errors = 0

    print(f"{'ID':<10} | {'Status':<15} | {'Detaljer'}")
    print("-" * 60)

    for item in results:
        res_id = item.get('id', item.get('use_case_id', 'unknown'))
        benchmark = item.get('benchmark', item.get('benchmark_ltl', item.get('benchmark', '')))
        llm_out = item.get('llm_output', item.get('output', ''))

        is_correct, error_type = verify_results(llm_out, benchmark)

        if is_correct:
            status = "Correct"
            correct += 1
        elif error_type == "syntax_error":
            status = "SYNTAX ERROR"
            syntax_errors += 1
        else:
            status = "Wrong Logic"

        print(f"{res_id:<10} | {status:<15} | {llm_out}\n")

    # Calculate and print overall statistics
    accuracy = (correct / total) * 100
    print("-" * 60)
    print(f"TOTAL EVALUATION:")
    print(f"Number of tests: {total}")
    print(f"Logically correct: {correct} ({accuracy:.1f}%)")
    print(f"Syntax errors: {syntax_errors}")

if __name__ == "__main__":
    # verify_results_file('gemini_results.json')
    verify_results_file('chatgpt_results.json')