import json
import os
import spot
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def verify_ltl_json(file_path):
    print(f"Loading benchmarks from: {file_path}")
    print("-" * 60)
    
    with open(file_path, 'r') as f:
        data = json.load(f)

    all_passed = True
    pass_count = 0

    for item in data:
        req_id = item.get('id', 'UNKNOWN')
        ltl_str = item.get('benchmark_ltl', '')

        try:
            # spot.formula() parses the string and throws a SyntaxError if invalid
            parsed_formula = spot.formula(ltl_str)
            print(parsed_formula)
            print(f"✅ [PASS] {req_id}: {ltl_str}")
            pass_count += 1
            
        except SyntaxError as e:
            print(f"❌ [FAIL] {req_id}: {ltl_str}")
            print(f"   Spot Parse Error:\n{e}\n")
            all_passed = False
            
        except Exception as e:
            print(f"❌ [FAIL] {req_id}: {ltl_str}")
            print(f"   Unexpected Error: {e}\n")
            all_passed = False

    print("-" * 60)
    print(f"Summary: {pass_count}/{len(data)} formulas passed Spot syntax validation.")
    
    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python verify_spot.py <path_to_benchmarks.json>")
    #     sys.exit(1)
        
    # verify_ltl_json(sys.argv[1])

    verify_ltl_json(os.path.join(BASE_DIR, 'requirements2.json'))
    # verify_ltl_json(os.path.join(BASE_DIR, 'gemini_results.json'))
    # verify_ltl_json(os.path.join(BASE_DIR, 'chatgpt_results.json'))