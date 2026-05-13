import json
import os
import re
import time
from pathlib import Path
import torch
from unsloth import FastLanguageModel
from verify_results import verify_results, logical_closeness, logical_closeness_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = Path(BASE_DIR).parent
NORMAL_OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')
TEMP_OUTPUT_DIR = os.path.join(BASE_DIR, 'Output_temp_comparison')
DOCUMENTS_DIR = Path.home() / "Documents"
TRAINED_MODEL_DIR = DOCUMENTS_DIR / "llama_training" / "llama3_ltl_specialist"

MODEL_DIR_CANDIDATES = [
    TRAINED_MODEL_DIR,
    WORKSPACE_DIR / "llama3_ltl_specialist",
    Path.home() / ".cache" / "llama3_ltl_specialist",
]
MODEL_DIR = next((path for path in MODEL_DIR_CANDIDATES if path.exists()), MODEL_DIR_CANDIDATES[0])

# 1. Configuration
max_seq_length = 2048
dtype = None  # Auto-detect
load_in_4bit = True
ENABLE_TEMPERATURE_SWEEP = False
TEMPERATURE_SWEEP_ITERATIONS = 2
TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 1.0
TEMPERATURE_POINTS = 20

# System instruction
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

def load_model():
    """Load the fine-tuned Llama 3.1 model"""
    print("Loading fine-tuned Llama 3.1 model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MODEL_DIR),
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    # Set model to inference mode
    FastLanguageModel.for_inference(model)
    print("Model loaded successfully!")
    return model, tokenizer


def prompt_llama(requirement_text, model, tokenizer, temperature=0.0):
    """Prompt the local Llama model"""
    # Format input with Llama 3.1 instruct template
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_INSTRUCTION}<|eot_id|><|start_header_id|>user<|end_header_id|>

{requirement_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    max_new_tokens = 500
    do_sample = temperature > 0.0
    generation_kwargs = {
        "max_length": input_length + max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )
    
    # Decode only the newly generated tokens (skip the input prompt)
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    response = response.strip()
    print(f"Llama response: {response}")
    return response


def _postprocess_ltl_for_spot(response_text):
    """Normalize generated LTL so Spot can parse atomic comparisons reliably.

    Rule applied: unquoted atomic propositions with comparison/arithmetic syntax are
    wrapped in double quotes, while already quoted fragments are left untouched.
    """
    if not response_text:
        return response_text

    cleaned = response_text.strip()

    # Remove markdown/code-fence wrappers if the model emits them.
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    # Pattern targets comparison expressions that should be atomic quoted strings.
    comparison_expr = re.compile(
        r"(" 
        r"(?:\([^()]*\)|[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?)"
        r"(?:\s*[-+*/]\s*(?:\([^()]*\)|[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?))*"
        r"\s*(?:==|!=|<=|>=|<|>)\s*"
        r"(?:\([^()]*\)|[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?)"
        r"(?:\s*[-+*/]\s*(?:\([^()]*\)|[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?))*"
        r")"
    )

    # Split by already-quoted segments so we never rewrite inside quotes.
    quoted_or_not = re.compile(r'("[^"\\]*(?:\\.[^"\\]*)*")')
    parts = quoted_or_not.split(cleaned)

    for i, part in enumerate(parts):
        if not part:
            continue
        if part.startswith('"') and part.endswith('"'):
            continue

        parts[i] = comparison_expr.sub(lambda m: f'"{m.group(1).strip()}"', part)

    return "".join(parts)


# 2. Single Evaluation Run
def run_evaluation_once(
    model,
    tokenizer,
    data,
    model_metadata,
    iteration,
    temperature=0.0,
    save_iteration_output=True,
    file_prefix="llama_results",
    output_dir=NORMAL_OUTPUT_DIR,
):
    """Run a single iteration of evaluation"""
    results = []
    
    for item in data:
        start_time = time.perf_counter()
        
        # API call
        llm_output_raw = prompt_llama(item['description'], model, tokenizer, temperature=temperature)
        llm_output = _postprocess_ltl_for_spot(llm_output_raw)
        
        end_time = time.perf_counter()
        
        # Calculate performance metrics
        comp_time = end_time - start_time
        
        # Spot Check
        is_correct, error_type = verify_results(llm_output, item['benchmark_ltl'])
        is_subsumed, is_generalized = logical_closeness(llm_output, item['benchmark_ltl'])
        closeness_score = logical_closeness_score(is_subsumed, is_generalized)

        results.append({
            "use_case_id": item['id'],
            "model": model_metadata['model_id'],
            "benchmark": item['benchmark_ltl'],
            "output": llm_output,
            "raw_output": llm_output_raw,
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


def _aggregate_temperature_runs(per_temperature_aggregates, model_metadata):
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
        "model": model_metadata["model_id"],
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


# 3. Main Loop - Run multiple iterations and aggregate results
def run_evaluation(iterations=2):
    """Run multiple iterations and aggregate results"""
    output_dir = NORMAL_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    model, tokenizer = load_model()

    with open(os.path.join(BASE_DIR, 'requirements.json'), 'r') as f:
        data = json.load(f)

    model_metadata = {
        "model_id": "llama3.1-8b-sft",
        "tier": "Fine-tuned",
        "provider": "Local"
    }
    
    # Run the specified number of iterations
    all_iterations_results = []
    for iteration in range(1, iterations + 1):
        print(f"\n{'='*50}")
        print(f"Starting Iteration {iteration} of {iterations}")
        print(f"{'='*50}")
        iteration_results = run_evaluation_once(
            model, tokenizer, data, model_metadata, iteration, temperature=0.0, output_dir=output_dir
        )
        all_iterations_results.append(iteration_results)
    
    # Calculate success rates per item and overall
    aggregated_data = calculate_success_rates(all_iterations_results, data, model_metadata, iterations)
    
    # Save aggregated results
    aggregated_file = os.path.join(output_dir, 'llama_results_aggregated.json')
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated_data, f, indent=4)
    print(f"\n{'='*50}")
    print(f"All iterations completed! Aggregated results saved to {aggregated_file}")
    print(f"{'='*50}")


def run_temperature_sweep():
    output_dir = TEMP_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = load_model()

    with open(os.path.join(BASE_DIR, 'requirements.json'), 'r') as f:
        data = json.load(f)

    model_metadata = {
        "model_id": "llama3.1-8b-sft",
        "tier": "Fine-tuned",
        "provider": "Local"
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
                model,
                tokenizer,
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
            model_metadata,
            TEMPERATURE_SWEEP_ITERATIONS,
        )
        temp_aggregate["temperature"] = temperature
        per_temperature_aggregates.append(temp_aggregate)

        temp_file = os.path.join(
            output_dir,
            f"llama_results_temperature_{temperature:.3f}_aggregated.json",
        )
        with open(temp_file, 'w') as f:
            json.dump(temp_aggregate, f, indent=4)
        print(f"Saved temperature aggregate to {temp_file}")

    aggregates_file = os.path.join(output_dir, 'llama_results_temperature_aggregates.json')
    with open(aggregates_file, 'w') as f:
        json.dump(per_temperature_aggregates, f, indent=4)

    final_aggregate = _aggregate_temperature_runs(per_temperature_aggregates, model_metadata)
    final_file = os.path.join(output_dir, 'llama_results_temperature_sweep_aggregated.json')
    with open(final_file, 'w') as f:
        json.dump(final_aggregate, f, indent=4)

    print(f"\n{'='*50}")
    print(f"Temperature sweep complete! Final aggregate saved to {final_file}")
    print(f"{'='*50}")


# 4. Calculate success rates from all iterations
def calculate_success_rates(all_iterations_results, data, model_metadata, iterations):
    """Calculate per-item and overall success rates"""
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
        "model": model_metadata['model_id'],
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
        run_temperature_sweep()
    else:
        run_evaluation(iterations=5)
