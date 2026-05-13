import json
import os
import random
import time
from google import genai
from google.genai import types
from verify_results import verify_results, logical_closeness, logical_closeness_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NORMAL_OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')
TEMP_OUTPUT_DIR = os.path.join(BASE_DIR, 'Output_temp_comparison')


def _load_api_key(service_name):
    key_file_candidates = [
        os.path.join(BASE_DIR, 'api keys'),
        os.path.join(BASE_DIR, 'api key'),
    ]

    key_file_path = next((path for path in key_file_candidates if os.path.exists(path)), None)
    if key_file_path is None:
        raise FileNotFoundError(
            "Could not find an API key file. Expected 'api keys' or 'api key' in the script folder."
        )

    with open(key_file_path, 'r') as file:
        current_service = None
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            if line.endswith(':'):
                current_service = line[:-1].strip().lower()
                continue

            if current_service == service_name.lower():
                return line.strip().strip('"').strip("'")

    raise ValueError(f"Could not find an API key for '{service_name}' in {key_file_path}.")

# 1. Configuration
API_KEY = _load_api_key('gemini')
gemini_client = genai.Client(api_key=API_KEY)

# Retry and pacing config to reduce temporary provider-side errors (503/429).
MAX_API_RETRIES = 6
INITIAL_RETRY_DELAY_SEC = 2.0
MAX_RETRY_DELAY_SEC = 45.0
INTER_REQUEST_DELAY_SEC = 1.0
HIGH_DEMAND_COOLDOWN_SEC = 20.0
DEFAULT_MODEL_NAME = "gemini-3-flash-preview"
MAX_RESPONSE_RETRIES = 3
ENABLE_TEMPERATURE_SWEEP = False
TEMPERATURE_SWEEP_ITERATIONS = 2
TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 1.0
TEMPERATURE_POINTS = 20

# System instruction
# Keep system instruction neutral.
SYSTEM_INSTRUCTION = """
You are an automaton who will strictly return a single LTL formula based on the natural language requirement provided.

### SYNTAX RULES:
1. ATOMIC PROPOSITIONS: Any variable comparison containing mathematical operators (==, !=, >, <, <=, >=, +, -, *, /) MUST be wrapped in double quotes. 
   - Example: G("battery > 0")
   - Boolean-only variables do not need quotes. Example: G(recharge -> F(atGoal))
2. OPERATORS: Use G (Globally), F (Eventually), X (Next), U (Until), W (Weak Until), R (Release), M (Strong Release).
3. BOOLEAN: Use ! (NOT), & (AND), | (OR), -> (IMPLIES), <-> (EQUIVALENT). Note: Use a single '&' and '|'.
4. COMPARISONS: Use '==' for equality and '!=' for inequality within quotes.

### STRICT RULES:
1. Use ONLY the variables provided in the mapping below.

### Variable Mapping Table:
- battery (Internal, integer): Current energy level (0-100).
- chargePosition (Internal, integer): Encodes the coordinate or ID of the fixed charging station location.
- recharge (Internal, boolean): Flag indicating that the rover needs to recharge its battery.
- goal (Internal, integer): Identifier or coordinate of the rovers current navigation target.
- pre_battery (Internal, integer): Previous timesteps battery value, used for energy consumption calculations.
- n (Internal, integer): Normalization factor or total number of plan steps used in energy calculation.
- plan (Internal, integer): Represents the active route or sequence of waypoints being executed by the rover.
- length_plan (Internal, integer): The total number of steps or waypoints in the current navigation plan.
- chargeNeeded_var (Internal, integer): Estimated amount of charge required to reach the charging station.
- batteryFull (Internal, boolean): Indicates that the rovers battery has reached full charge capacity.
- currentPosition (Input, integer): The rovers current logical position according to navigation data.
- initialPosition (Input, integer): The initial or starting position of the rover when the mission begins.
- currentPhysicalPosition (Input, integer): The physical location reported by sensors (used to check accuracy).
- start (Internal, integer): Initial map or system state in the vision/map validation process.
- s0 (Constant, integer): Static reference position used to validate the starting state.
- x (Internal, integer): Static Charging station location
- y (Internal, integer): Static Initial rover position
- obstacle (Input, integer): Position identifier of an obstacle detected by the vision subsystem.
- obstaclePhysicallyDetected (Internal, boolean): Sensor status: Obstacle has been verified by vision.
- isHeatpoint (Internal, boolean): True if the current tile being evaluated is a heatpoint.
- heatPhysicallyMeasured (Internal, boolean): Sensor status: Heatpoint verified by sensors.
- isReachable (Internal, boolean): True if a valid path exists to the heatpoint.
- visited (Internal, boolean): True if the current goal has already been inspected.
- hottestUnvisited (Internal, integer): The ID of the highest-value heatpoint not yet visited.
- Obstacle_currentPosition (Internal, boolean): True if there is an obstacle at the rover's current position.
- speed (Input, integer): The rovers velocity, typically in km/h.
- removeGoalFromSet (Output, boolean): Command to remove a completed goal from the navigation goal list.
- atGoal (Input, boolean): Status flag that becomes true when the rover reaches its goal position.
- heatpointLocation (Internal, integer): The specific coordinate or location ID associated with a detected heatpoint.
- isObstacle (Input, boolean): A temporary flag indicating that the currently evaluated coordinate or object is classified as an obstacle by the vision subsystem.
- isShortestPath (Internal, boolean): A status flag confirming that the current navigation plan has the lowest possible length_plan among all valid alternatives.
- planContainsObstacle (Internal, boolean): A safety verification result indicating whether any segment of the active navigation plan intersects with a known obstacle location.
- planHasStart (Internal, boolean): A validation check ensuring the current navigation plan begins exactly at the rover's currentPosition.

### OUTPUT INSTRUCTIONS:
Very important: Generate only the temporal logic specification without any explanation or additional commentary.
"""


def _is_retryable_api_error(error_text):
    if not error_text:
        return False

    normalized = error_text.upper()
    retryable_markers = [
        "503",
        "UNAVAILABLE",
        "429",
        "RESOURCE_EXHAUSTED",
        "INTERNAL",
        "DEADLINE_EXCEEDED"
    ]
    return any(marker in normalized for marker in retryable_markers)


def _is_clean_ltl_response(response_text):
    if not response_text:
        return False

    cleaned = response_text.strip()
    if not cleaned:
        return False

    if "\n" in cleaned or "`" in cleaned:
        return False

    lowered = cleaned.lower()
    banned_phrases = [
        "thinking",
        "wait",
        "i'll",
        "i will",
        "let's",
        "first",
        "therefore",
        "so ",
        "it sounds",
        "this means",
        "i think",
    ]
    if any(phrase in lowered for phrase in banned_phrases):
        return False

    if not cleaned[0] in ("G", "F", "X", "!", "(", "\""):
        return False

    return True


def prompt_gemini(requirement_text, model_name, temperature=0.0):

    last_error = "ERROR: Unknown API error"

    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=temperature,
                    max_output_tokens=2000
                ),
                contents=requirement_text
            )

            if response and response.text and response.text.strip():
                cleaned_response = response.text.strip()
                if _is_clean_ltl_response(cleaned_response):
                    print(f"Gemini response: {cleaned_response}")
                    return cleaned_response

                last_error = "ERROR: Non-LTL response received"
                print(
                    f"Rejected non-LTL response on attempt {attempt}/{MAX_API_RETRIES}: "
                    f"{cleaned_response[:200]}"
                )
                if attempt < MAX_RESPONSE_RETRIES:
                    time.sleep(0.5)
                    continue

            last_error = "ERROR: Empty response from model"
        except Exception as e:
            last_error = f"ERROR: {str(e)}"

        if _is_retryable_api_error(last_error) and attempt < MAX_API_RETRIES:
            backoff = min(MAX_RETRY_DELAY_SEC, INITIAL_RETRY_DELAY_SEC * (2 ** (attempt - 1)))
            jitter = random.uniform(0.0, 1.5)
            sleep_seconds = backoff + jitter
            print(
                f"Temporary API issue on attempt {attempt}/{MAX_API_RETRIES}: {last_error}. "
                f"Retrying in {sleep_seconds:.1f}s..."
            )
            time.sleep(sleep_seconds)
            continue

        return last_error

    return last_error

# 2. Single Evaluation Run
def run_evaluation_once(
    model_name,
    data,
    model_metadata,
    iteration,
    temperature=0.0,
    save_iteration_output=True,
    file_prefix="gemini_results",
    output_dir=NORMAL_OUTPUT_DIR,
):
    results = []
    
    for item in data:
        # print(f"Processing {item['id']}...")
        # llm_output = prompt_gemini(item['description'])

        # results.append({
        #     "id": item['id'],
        #     "nl": item['description'],
        #     "benchmark": item['benchmark_ltl'],
        #     "llm_output": llm_output
        # })
        #break # Remove this break to process all items
        # Models may have rate limits, so we add a small delay to avoid hitting them.
        #time.sleep(12)


        start_time = time.perf_counter()
        
        # API call
        llm_output = prompt_gemini(item['description'], model_name, temperature=temperature)

        # If provider is still overloaded after retries, cool down before next call.
        if _is_retryable_api_error(llm_output):
            print(
                f"Model still under high demand after retries. Cooling down for "
                f"{HIGH_DEMAND_COOLDOWN_SEC:.0f}s before continuing..."
            )
            time.sleep(HIGH_DEMAND_COOLDOWN_SEC)
        
        end_time = time.perf_counter()
        
        # Calculate performance metrics
        comp_time = end_time - start_time
        
        # Spot Check
        is_correct, error_type = verify_results(llm_output, item['benchmark_ltl'])
        is_subsumed, is_generalized = logical_closeness(llm_output, item['benchmark_ltl'])
        closeness_score = logical_closeness_score(is_subsumed, is_generalized)

        results.append({
            "use_case_id": item['id'],
            "model": model_name,
            "benchmark": item['benchmark_ltl'],
            "output": llm_output,
            "is_correct": is_correct,          # Accuracy metric
            "error_type": error_type,          # Error categorization
            "is_subsumed": is_subsumed,
            "is_generalized": is_generalized,
            "logical_closeness_score": closeness_score,
            "comp_time_sec": round(comp_time, 3), 
            "model_tier": model_metadata['tier'],
            "iteration": iteration,
            "temperature": temperature,
        })

        # Light pacing helps avoid burst traffic limits.
        time.sleep(INTER_REQUEST_DELAY_SEC)

    # Save single iteration results to JSON file
    if save_iteration_output:
        output_file = os.path.join(output_dir, f'{file_prefix}_iteration_{iteration}.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Iteration {iteration} completed! Results saved to {output_file}")
    
    return results


def _build_temperature_values(min_temp, max_temp, count=None, step=None):
    if step is not None and step > 0:
        values = []
        current = min_temp
        while current <= max_temp + 1e-9:
            values.append(round(current, 3))
            current += step
        return values

    if not count or count < 2:
        return [round(min_temp, 3)]

    step_size = (max_temp - min_temp) / (count - 1)
    return [round(min_temp + i * step_size, 3) for i in range(count)]


def _aggregate_temperature_runs(per_temperature_aggregates, model_name, model_metadata):
    total_correct = 0
    total_tests = 0
    total_comp_time = 0.0
    total_closeness_score_weighted = 0.0
    total_closeness_count = 0

    per_temperature_overall = []
    best_entry = None

    for entry in per_temperature_aggregates:
        stats = entry.get("overall_statistics", {})
        total_correct += stats.get("total_correct", 0)
        total_tests += stats.get("total_tests", 0)
        total_comp_time += stats.get("total_computation_time_sec", 0.0)
        closeness_count = stats.get("logical_closeness_evaluated_count", 0)
        closeness_avg = stats.get("average_logical_closeness_score")
        if closeness_avg is not None and closeness_count:
            total_closeness_score_weighted += closeness_avg * closeness_count
            total_closeness_count += closeness_count

        row = {
            "temperature": entry["temperature"],
            "overall_success_rate_percent": stats.get("overall_success_rate_percent"),
            "average_computation_time_sec": stats.get("average_computation_time_sec"),
            "total_tests": stats.get("total_tests", 0),
        }
        per_temperature_overall.append(row)

        if best_entry is None or (row["overall_success_rate_percent"] or 0) > (
            best_entry["overall_success_rate_percent"] or 0
        ):
            best_entry = row

    final_overall_success = (total_correct / total_tests * 100) if total_tests > 0 else 0
    final_average_comp_time = (total_comp_time / total_tests) if total_tests > 0 else 0

    return {
        "model": model_name,
        "model_metadata": model_metadata,
        "temperature_values": [entry["temperature"] for entry in per_temperature_aggregates],
        "temperatures_tested": len(per_temperature_aggregates),
        "iterations_per_temperature": TEMPERATURE_SWEEP_ITERATIONS,
        "per_temperature_overall": per_temperature_overall,
        "best_temperature_by_success": best_entry,
        "overall_statistics": {
            "total_correct": total_correct,
            "total_tests": total_tests,
            "overall_success_rate_percent": round(final_overall_success, 2),
            "average_computation_time_sec": round(final_average_comp_time, 3),
            "total_computation_time_sec": round(total_comp_time, 3),
            "average_logical_closeness_score": round(total_closeness_score_weighted / total_closeness_count, 3)
            if total_closeness_count > 0
            else None,
            "logical_closeness_evaluated_count": total_closeness_count,
        },
    }

# 3. Main Loop - Run 5 iterations and aggregate results
def run_evaluation(model_name, iterations=5):
    output_dir = NORMAL_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(BASE_DIR, 'requirements.json'), 'r') as f:
        data = json.load(f)

    model_metadata = {
        "model_id": model_name,
        "tier": "Flash" if "flash" in model_name else "Pro",
        "provider": "Google"
    }
    
    # Run the specified number of iterations
    all_iterations_results = []
    for iteration in range(1, iterations + 1):
        print(f"\n{'='*50}")
        print(f"Starting Iteration {iteration} of {iterations}")
        print(f"{'='*50}")
        iteration_results = run_evaluation_once(
            model_name, data, model_metadata, iteration, temperature=0.0, output_dir=output_dir
        )
        all_iterations_results.append(iteration_results)
    
    # Calculate success rates per item and overall
    aggregated_data = calculate_success_rates(all_iterations_results, data, model_name, model_metadata, iterations)
    
    # Save aggregated results
    aggregated_file = os.path.join(output_dir, 'gemini_results_aggregated.json')
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated_data, f, indent=4)
    print(f"\n{'='*50}")
    print(f"All iterations completed! Aggregated results saved to {aggregated_file}")
    print(f"{'='*50}")


def run_temperature_sweep(model_name=DEFAULT_MODEL_NAME):
    output_dir = TEMP_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(BASE_DIR, 'requirements.json'), 'r') as f:
        data = json.load(f)

    model_metadata = {
        "model_id": model_name,
        "tier": "Flash" if "flash" in model_name else "Pro",
        "provider": "Google"
    }

    temperature_values = _build_temperature_values(
        TEMPERATURE_MIN,
        TEMPERATURE_MAX,
        count=TEMPERATURE_POINTS,
    )
    per_temperature_aggregates = []

    for temperature in temperature_values:
        all_iteration_results = []
        print(f"\n{'='*50}")
        print(f"Temperature sweep at {temperature:.3f}")
        print(f"{'='*50}")

        for iteration in range(1, TEMPERATURE_SWEEP_ITERATIONS + 1):
            print(f"Running iteration {iteration}/{TEMPERATURE_SWEEP_ITERATIONS} at temperature {temperature:.3f}")
            iteration_results = run_evaluation_once(
                model_name,
                data,
                model_metadata,
                iteration,
                temperature=temperature,
                save_iteration_output=False,
                output_dir=output_dir,
            )
            all_iteration_results.append(iteration_results)

        temp_aggregate = calculate_success_rates(
            all_iteration_results,
            data,
            model_name,
            model_metadata,
            TEMPERATURE_SWEEP_ITERATIONS,
        )
        temp_aggregate["temperature"] = temperature
        per_temperature_aggregates.append(temp_aggregate)

        temp_file = os.path.join(
            output_dir,
            f"gemini_results_temperature_{temperature:.3f}_aggregated.json",
        )
        with open(temp_file, 'w') as f:
            json.dump(temp_aggregate, f, indent=4)
        print(f"Saved temperature aggregate to {temp_file}")

    aggregates_file = os.path.join(output_dir, 'gemini_results_temperature_aggregates.json')
    with open(aggregates_file, 'w') as f:
        json.dump(per_temperature_aggregates, f, indent=4)

    final_aggregate = _aggregate_temperature_runs(per_temperature_aggregates, model_name, model_metadata)
    final_file = os.path.join(output_dir, 'gemini_results_temperature_sweep_aggregated.json')
    with open(final_file, 'w') as f:
        json.dump(final_aggregate, f, indent=4)

    print(f"\n{'='*50}")
    print(f"Temperature sweep complete! Final aggregate saved to {final_file}")
    print(f"{'='*50}")

# 4. Calculate success rates from all iterations
def calculate_success_rates(all_iterations_results, data, model_name, model_metadata, iterations):
    item_success_rates = {}
    total_correct = 0
    total_tests = 0
    total_computation_time = 0.0
    total_closeness_score = 0.0
    total_closeness_count = 0
    total_subsumed_count = 0
    total_generalized_count = 0
    total_bidirectional_count = 0
    
    # Calculate per-item success rate
    for item in data:
        item_id = item['id']
        correct_count = 0
        total_count = 0
        item_total_comp_time = 0.0
        error_types = {}
        item_closeness_score = 0.0
        item_closeness_count = 0
        item_subsumed_count = 0
        item_generalized_count = 0
        item_bidirectional_count = 0
        
        for iteration_results in all_iterations_results:
            for result in iteration_results:
                if result['use_case_id'] == item_id:
                    total_count += 1
                    if result['is_correct']:
                        correct_count += 1

                    # Aggregate timing already measured per requirement per iteration.
                    item_total_comp_time += result.get('comp_time_sec', 0.0)

                    is_subsumed = result.get('is_subsumed')
                    is_generalized = result.get('is_generalized')
                    if is_subsumed is True:
                        item_subsumed_count += 1
                    if is_generalized is True:
                        item_generalized_count += 1
                    if is_subsumed is True and is_generalized is True:
                        item_bidirectional_count += 1

                    closeness = result.get('logical_closeness_score')
                    if closeness is not None:
                        item_closeness_score += closeness
                        item_closeness_count += 1
                    
                    # Track error types
                    error_type = result['error_type']
                    error_types[error_type] = error_types.get(error_type, 0) + 1
        
        if total_count > 0:
            success_rate = (correct_count / total_count) * 100
            avg_comp_time = item_total_comp_time / total_count
            item_success_rates[item_id] = {
                "correct": correct_count,
                "total": total_count,
                "success_rate_percent": round(success_rate, 2),
                "average_computation_time_sec": round(avg_comp_time, 3),
                "error_breakdown": error_types,
                "logical_closeness": {
                    "average_score": round(item_closeness_score / item_closeness_count, 3) if item_closeness_count > 0 else None,
                    "evaluated_count": item_closeness_count,
                    "subsumed_count": item_subsumed_count,
                    "generalized_count": item_generalized_count,
                    "bidirectional_count": item_bidirectional_count
                }
            }
            total_correct += correct_count
            total_tests += total_count
            total_computation_time += item_total_comp_time
            total_closeness_score += item_closeness_score
            total_closeness_count += item_closeness_count
            total_subsumed_count += item_subsumed_count
            total_generalized_count += item_generalized_count
            total_bidirectional_count += item_bidirectional_count
    
    # Calculate overall success rate
    overall_success_rate = (total_correct / total_tests * 100) if total_tests > 0 else 0
    average_computation_time = (total_computation_time / total_tests) if total_tests > 0 else 0
    
    aggregated_data = {
        "model": model_name,
        "model_metadata": model_metadata,
        "total_iterations": iterations,
        "per_item_success_rates": item_success_rates,
        "overall_statistics": {
            "total_correct": total_correct,
            "total_tests": total_tests,
            "overall_success_rate_percent": round(overall_success_rate, 2),
            "average_computation_time_sec": round(average_computation_time, 3),
            "total_computation_time_sec": round(total_computation_time, 3),
            "average_logical_closeness_score": round(total_closeness_score / total_closeness_count, 3) if total_closeness_count > 0 else None,
            "logical_closeness_evaluated_count": total_closeness_count,
            "subsumed_count": total_subsumed_count,
            "generalized_count": total_generalized_count,
            "bidirectional_count": total_bidirectional_count
        }
    }
    
    return aggregated_data

if __name__ == "__main__":
    if ENABLE_TEMPERATURE_SWEEP:
        run_temperature_sweep(DEFAULT_MODEL_NAME)
    else:
        run_evaluation(DEFAULT_MODEL_NAME, 5)
