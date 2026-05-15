import os
import json
import glob
import statistics

def calculate_times(file_pattern):
    files = sorted(glob.glob(file_pattern))
    avg_times = []
    total_times = []
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            total = len(data)
            if total > 0:
                total_time = sum(item.get("comp_time_sec", 0) for item in data)
                total_times.append(total_time)
                avg_times.append(total_time / total)
    return avg_times, total_times

def main():
    # Automatically resolve the Output directory relative to the script location
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output")
    if not os.path.exists(output_dir):
        print(f"Error: Directory '{output_dir}' not found.")
        return

    models = {
        "GPT-5.4": "chatgpt_results_iteration_*.json",
        "Gemini 3 Flash": "gemini_results_iteration_*.json",
        "Claude Sonnet 4.6": "claude_results_iteration_*.json",
        "Claude Opus 4.6": "claude_results_opus_iteration_*.json",
        "Llama 3.1 Base": "llama_base_results_iteration_*.json",
        "Llama 3.1 (FT)": "llama_results_iteration_*.json"
    }

    print(f"{'Model':<20} | {'Mean Avg Time (s)':<17} | {'Std Dev (s)':<11} | {'Mean Total Time (s)':<19}")
    print("-" * 80)

    for model_name, pattern in models.items():
        full_pattern = os.path.join(output_dir, pattern)
        avg_times, total_times = calculate_times(full_pattern)
        
        if avg_times:
            if len(avg_times) > 1:
                mean_avg_time = statistics.mean(avg_times)
                std_dev_avg = statistics.stdev(avg_times)
                mean_total_time = statistics.mean(total_times)
                print(f"{model_name:<20} | {mean_avg_time:>17.3f} | {std_dev_avg:>11.3f} | {mean_total_time:>19.3f}")
            else:
                print(f"{model_name:<20} | {avg_times[0]:>17.3f} | {'N/A':>11} | {total_times[0]:>19.3f}")
        else:
            print(f"{model_name:<20} | {'No data':>17} | {'N/A':>11} | {'N/A':>19}")

if __name__ == "__main__":
    main()
