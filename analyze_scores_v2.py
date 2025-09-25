#!/usr/bin/env python3
"""
Analyzes a JSONL file and outputs statistical results using a memory-efficient
streaming approach, making it suitable for very large files.
"""
import json
import argparse
import math

# Attempt to import the textstat library for readability analysis
try:
    import textstat
    textstat.set_lang("en_US")
except ImportError:
    print("⚠️ Warning: The 'textstat' library is not installed. Text difficulty analysis will be unavailable.")
    print("   To install it, run: pip install textstat")
    textstat = None

class StatTracker:
    """
    A helper class for calculating statistics online using Welford's algorithm.
    This allows for stable, single-pass computation of mean and standard deviation.
    """
    def __init__(self):
        self.count = 0
        self.min = float('inf')
        self.max = float('-inf')
        self.sum = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squares of differences from the current mean

    def update(self, value: float):
        """Updates all statistics with a new value."""
        if value is None or not isinstance(value, (int, float)):
            return
            
        self.count += 1
        self.sum += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        
        # Welford's online algorithm for numerically stable variance calculation
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
    
    @property
    def std(self) -> float:
        """Calculates the standard deviation."""
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / self.count)

def analyze_file_streamed(jsonl_file: str):
    """
    Analyzes a JSONL file via streaming and prints the statistical results.
    """
    stats_trackers = {}
    total_records = 0
    valid_records = 0
    error_records = 0

    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_records += 1
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"⚠️ Warning: Skipping invalid JSON on line {line_num}: {line.strip()}")
                    error_records += 1
                    continue
                
                valid_records += 1
                
                # --- Dynamically track all numerical metrics ---
                # 1. Top-level metrics, e.g., 'perplexity'
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        if key not in stats_trackers:
                            stats_trackers[key] = StatTracker()
                        stats_trackers[key].update(value)
                
                # 2. Nested scores inside an 'scores' or 'answer' dictionary
                scores_dict = data.get('scores', data.get('answer'))
                if isinstance(scores_dict, dict):
                    for key, value in scores_dict.items():
                        if isinstance(value, (int, float)):
                            # Prefix score metrics to distinguish them
                            score_key = f"score_{key}"
                            if score_key not in stats_trackers:
                                stats_trackers[score_key] = StatTracker()
                            stats_trackers[score_key].update(value)

                # 3. Text readability metrics if textstat is available
                if textstat and 'generated_text' in data and isinstance(data['generated_text'], str):
                    text = data['generated_text']
                    
                    # Flesch Reading Ease
                    if 'difficulty_flesch_ease' not in stats_trackers:
                        stats_trackers['difficulty_flesch_ease'] = StatTracker()
                    stats_trackers['difficulty_flesch_ease'].update(textstat.flesch_reading_ease(text))
                    
                    # Flesch-Kincaid Grade Level
                    if 'difficulty_flesch_grade' not in stats_trackers:
                        stats_trackers['difficulty_flesch_grade'] = StatTracker()
                    stats_trackers['difficulty_flesch_grade'].update(textstat.flesch_kincaid_grade(text))
    
    except FileNotFoundError:
        print(f"❌ Error: File '{jsonl_file}' not found.")
        return

    print(f"\n✅ Analysis complete for '{jsonl_file}'.")
    print(f"   Processed {total_records} total lines.")
    print(f"   (Found {valid_records} valid records and {error_records} invalid/skipped records)\n")
    
    if valid_records == 0:
        print("❗️ No valid records were found to analyze!")
        return
        
    # --- Print results by category ---
    general_metrics = {k: v for k, v in stats_trackers.items() if not k.startswith(('score_', 'difficulty_'))}
    score_metrics = {k: v for k, v in stats_trackers.items() if k.startswith('score_')}
    difficulty_metrics = {k: v for k, v in stats_trackers.items() if k.startswith('difficulty_')}
    
    print_stats_category("📊 General Metrics (e.g., Perplexity)", general_metrics)
    print_stats_category("📊 Score Statistics", score_metrics)
    print_stats_category("📊 Text Difficulty Statistics", difficulty_metrics)

def print_stats_category(title: str, trackers: dict):
    """Formats and prints a category of statistical results in a table."""
    if not trackers:
        return
        
    print(title)
    print("=================================================================")
    print(f"  {'Metric':<25} {'Mean':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10}")
    print(f"  ---------------------------------------------------------------")
    for key, tracker in sorted(trackers.items()):
        # Clean up the key for display
        display_name = key.replace('score_', '').replace('difficulty_', '').replace('_', ' ').capitalize()
        if tracker.count > 0:
            print(f"  {display_name:<25} {tracker.mean:<10.2f} {tracker.std:<10.2f} {tracker.min:<10.2f} {tracker.max:<10.2f}")
    print("=================================================================\n")

def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(
        description="A memory-efficient script to analyze a JSONL file and output summary statistics.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_file', help="Path to the JSONL file to be analyzed.")
    args = parser.parse_args()
    
    analyze_file_streamed(args.input_file)

if __name__ == '__main__':
    main()