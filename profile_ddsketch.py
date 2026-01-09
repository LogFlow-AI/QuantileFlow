import cProfile
import pstats
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

from QuantileFlow.ddsketch.core import DDSketch


BENCHMARK_DIR = Path("benchmarks")
BENCHMARK_DIR.mkdir(exist_ok=True)


def run_sketch_operations(num_values: int = 10_000_000) -> Dict:
    """Runs typical DDSketch operations for profiling.
    
    Returns:
        Dict containing operation results and metrics
    """
    sketch = DDSketch(relative_accuracy=0.01)
    
    # Generate random data
    data = np.random.rand(num_values) * 1000
    
    # Track insertion time
    for value in data:
        sketch.insert(value)
        
    # Compute quantiles
    quantiles_to_compute = [0.5, 0.9, 0.99, 0.999]
    quantile_results = {}
    for q in quantiles_to_compute:
        try:
            quantile_value = sketch.quantile(q)
            quantile_results[q] = quantile_value
        except ValueError as e:
            quantile_results[q] = f"Error: {e}"
    
    return {
        'num_values': num_values,
        'quantiles': quantile_results
    }


def extract_profile_stats(profiler: cProfile.Profile, top_n: int = 20) -> List[Dict]:
    """Extract key statistics from profiler."""
    stats = pstats.Stats(profiler)
    stats_data = []
    
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, line, func_name = func
        stats_data.append({
            'function': f"{Path(filename).name}:{line}({func_name})",
            'ncalls': nc,
            'tottime': tt,
            'cumtime': ct,
            'percall_tot': tt / nc if nc > 0 else 0,
            'percall_cum': ct / nc if nc > 0 else 0,
        })
    
    # Sort by cumulative time
    stats_data.sort(key=lambda x: x['cumtime'], reverse=True)
    return stats_data[:top_n]


def print_comparison_table(current: List[Dict], baseline: List[Dict]):
    """Print side-by-side comparison of current run vs baseline."""
    print(f"\n{'=' * 130}")
    print(f"{'Performance Comparison (Current vs Baseline)':^130}")
    print('=' * 130)
    
    header = (f"{'Function':<45} "
              f"{'Curr Time':>12} {'Base Time':>12} {'Diff':>10} {'Change %':>12} {'Calls Δ':>10}")
    print(header)
    print('-' * 130)
    
    # Create lookup dict for baseline
    baseline_dict = {stat['function']: stat for stat in baseline}
    
    for curr_stat in current:
        func_name = curr_stat['function']
        if len(func_name) > 42:
            func_name = "..." + func_name[-42:]
        
        baseline_stat = baseline_dict.get(curr_stat['function'])
        
        if baseline_stat:
            time_diff = curr_stat['cumtime'] - baseline_stat['cumtime']
            time_change = (time_diff / baseline_stat['cumtime'] * 100) if baseline_stat['cumtime'] > 0 else 0
            calls_diff = curr_stat['ncalls'] - baseline_stat['ncalls']
            
            # Color coding for terminal (green for improvement, red for regression)
            change_str = f"{time_change:+.1f}%"
            if time_change < -5:  # Significant improvement
                change_str = f"\033[92m{change_str}\033[0m"
            elif time_change > 5:  # Significant regression
                change_str = f"\033[91m{change_str}\033[0m"
            
            print(f"{func_name:<45} "
                  f"{curr_stat['cumtime']:>12.4f} "
                  f"{baseline_stat['cumtime']:>12.4f} "
                  f"{time_diff:>+10.4f} "
                  f"{change_str:>12} "
                  f"{calls_diff:>+10}")
        else:
            print(f"{func_name:<45} "
                  f"{curr_stat['cumtime']:>12.4f} "
                  f"{'N/A':>12} "
                  f"{'N/A':>10} "
                  f"{'NEW':>12} "
                  f"{'N/A':>10}")
    
    print('=' * 130)


def save_benchmark(stats_data: List[Dict], name: str, metadata: Dict):
    """Save benchmark results to file."""
    benchmark = {
        'timestamp': datetime.now().isoformat(),
        'name': name,
        'metadata': metadata,
        'stats': stats_data
    }
    
    filepath = BENCHMARK_DIR / f"{name}.json"
    with open(filepath, 'w') as f:
        json.dump(benchmark, f, indent=2)
    
    print(f"\n✓ Benchmark saved to: {filepath}")


def load_benchmark(name: str) -> Tuple[List[Dict], Dict]:
    """Load benchmark from file."""
    filepath = BENCHMARK_DIR / f"{name}.json"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Benchmark '{name}' not found at {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data['stats'], data['metadata']


def list_benchmarks():
    """List all available benchmarks."""
    benchmarks = sorted(BENCHMARK_DIR.glob("*.json"))
    
    if not benchmarks:
        print("\nNo benchmarks found.")
        return
    
    print(f"\n{'Available Benchmarks':^90}")
    print('=' * 90)
    print(f"{'Name':<25} {'Date':<22} {'Values':<15} {'Trials':<10}")
    print('-' * 90)
    
    for bm_file in benchmarks:
        with open(bm_file, 'r') as f:
            data = json.load(f)
        name = bm_file.stem
        timestamp = datetime.fromisoformat(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        num_values = data.get('metadata', {}).get('num_values', 'N/A')
        num_trials = data.get('metadata', {}).get('num_trials', 1)
        
        values_str = f"{num_values:,}" if isinstance(num_values, int) else str(num_values)
        print(f"{name:<25} {timestamp:<22} {values_str:<15} {num_trials:<10}")
    
    print('=' * 90)


def merge_trial_stats(all_trial_stats: List[List[Dict]]) -> Tuple[List[Dict], List[Dict]]:
    """Merge statistics from multiple trials, computing mean and std dev.
    
    Args:
        all_trial_stats: List of stats from each trial
        
    Returns:
        Tuple of (mean_stats, std_stats)
    """
    # Build a dict mapping function -> list of stats from each trial
    func_stats = {}
    
    for trial_stats in all_trial_stats:
        for stat in trial_stats:
            func = stat['function']
            if func not in func_stats:
                func_stats[func] = []
            func_stats[func].append(stat)
    
    # Compute mean and std dev for each function
    mean_stats = []
    std_stats = []
    
    for func, stats_list in func_stats.items():
        if not stats_list:
            continue
        
        # Compute means
        mean_stat = {
            'function': func,
            'ncalls': int(np.mean([s['ncalls'] for s in stats_list])),
            'tottime': np.mean([s['tottime'] for s in stats_list]),
            'cumtime': np.mean([s['cumtime'] for s in stats_list]),
            'percall_tot': np.mean([s['percall_tot'] for s in stats_list]),
            'percall_cum': np.mean([s['percall_cum'] for s in stats_list]),
        }
        
        # Compute standard deviations
        std_stat = {
            'function': func,
            'tottime_std': np.std([s['tottime'] for s in stats_list]),
            'cumtime_std': np.std([s['cumtime'] for s in stats_list]),
        }
        
        mean_stats.append(mean_stat)
        std_stats.append(std_stat)
    
    # Sort by cumulative time
    mean_stats.sort(key=lambda x: x['cumtime'], reverse=True)
    
    # Reorder std_stats to match mean_stats
    func_to_std = {s['function']: s for s in std_stats}
    std_stats = [func_to_std[s['function']] for s in mean_stats]
    
    return mean_stats, std_stats


def print_profile_table_with_std(stats_data: List[Dict], std_data: List[Dict], 
                                  title: str = "Profile Results", num_trials: int = 1):
    """Print profile statistics with standard deviation."""
    print(f"\n{'=' * 120}")
    print(f"{title:^120}")
    if num_trials > 1:
        print(f"{'(Averaged over ' + str(num_trials) + ' trials)':^120}")
    print('=' * 120)
    
    if num_trials > 1:
        header = f"{'Function':<45} {'Calls':>10} {'TotTime':>12} {'CumTime':>12} {'±StdDev':>10} {'Per Call':>12}"
    else:
        header = f"{'Function':<45} {'Calls':>10} {'TotTime':>12} {'CumTime':>12} {'Per Call':>12}"
    print(header)
    print('-' * 120)
    
    std_dict = {s['function']: s for s in std_data} if std_data else {}
    
    for stat in stats_data:
        func_name = stat['function']
        if len(func_name) > 42:
            func_name = "..." + func_name[-42:]
        
        if num_trials > 1 and stat['function'] in std_dict:
            std = std_dict[stat['function']]
            print(f"{func_name:<45} "
                  f"{stat['ncalls']:>10} "
                  f"{stat['tottime']:>12.4f} "
                  f"{stat['cumtime']:>12.4f} "
                  f"±{std['cumtime_std']:>9.4f} "
                  f"{stat['percall_cum']:>12.6f}")
        else:
            print(f"{func_name:<45} "
                  f"{stat['ncalls']:>10} "
                  f"{stat['tottime']:>12.4f} "
                  f"{stat['cumtime']:>12.4f} "
                  f"{stat['percall_cum']:>12.6f}")
    
    print('=' * 120)


def profile(num_values: int = 10_000_000, 
            save_as: str = None, 
            compare_to: str = None,
            top_n: int = 20,
            num_trials: int = 1):
    """Profiles the run_sketch_operations function.
    
    Args:
        num_values: Number of values to insert per trial
        save_as: If provided, save results as a benchmark with this name
        compare_to: If provided, compare results to this benchmark
        top_n: Number of top functions to display
        num_trials: Number of trials to run and average
    """
    print(f"\n{'Starting Profile Run':^60}")
    print(f"{'=' * 60}")
    print(f"Values per trial: {num_values:,}")
    print(f"Number of trials: {num_trials}")
    print(f"Total operations: {num_values * num_trials:,}")
    print(f"Top functions to show: {top_n}")
    print('=' * 60)
    
    all_trial_stats = []
    
    for trial in range(num_trials):
        if num_trials > 1:
            print(f"\n▶ Running trial {trial + 1}/{num_trials}...")
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        results = run_sketch_operations(num_values)
        
        profiler.disable()
        
        # Extract statistics for this trial
        trial_stats = extract_profile_stats(profiler, top_n=top_n * 2)  # Get more to ensure we have enough after averaging
        all_trial_stats.append(trial_stats)
    
    # Merge stats from all trials
    if num_trials > 1:
        print(f"\n📊 Computing averages across {num_trials} trials...")
        stats_data, std_data = merge_trial_stats(all_trial_stats)
        stats_data = stats_data[:top_n]  # Trim to requested top_n
        std_data = std_data[:top_n]
    else:
        stats_data = all_trial_stats[0][:top_n]
        std_data = []
    
    # Print results
    print("\n📊 Operation Results:")
    print(f"  • Values inserted per trial: {results['num_values']:,}")
    if num_trials > 1:
        print(f"  • Total values processed: {results['num_values'] * num_trials:,}")
    print("  • Sample quantiles from last trial:")
    for q, val in results['quantiles'].items():
        print(f"    - Q({q}): {val}")
    
    # Show profile table
    print_profile_table_with_std(stats_data, std_data, 
                                  "Current Run - Top Functions by Cumulative Time",
                                  num_trials=num_trials)
    
    # Total time summary
    total_time = sum(stat['cumtime'] for stat in stats_data[:5])
    if num_trials > 1 and std_data:
        total_std = np.sqrt(sum(std['cumtime_std']**2 for std in std_data[:5]))
        print(f"\n⏱️  Top 5 functions avg time: {total_time:.4f}s (±{total_std:.4f}s)")
        cv = (total_std / total_time * 100) if total_time > 0 else 0
        print(f"   Coefficient of variation: {cv:.2f}%", end="")
        if cv < 5:
            print(" ✓ (Very stable)")
        elif cv < 10:
            print(" (Stable)")
        elif cv < 20:
            print(" ⚠ (Moderate variance)")
        else:
            print(" ⚠ (High variance - consider more trials)")
    else:
        print(f"\n⏱️  Top 5 functions total time: {total_time:.4f}s")
    
    # Compare to baseline if requested
    if compare_to:
        try:
            baseline_stats, baseline_metadata = load_benchmark(compare_to)
            baseline_trials = baseline_metadata.get('num_trials', 1)
            print(f"\n📈 Comparing to baseline: '{compare_to}'")
            print(f"   Baseline date: {baseline_metadata.get('timestamp', 'N/A')}")
            print(f"   Baseline values per trial: {baseline_metadata.get('num_values', 'N/A'):,}")
            print(f"   Baseline trials: {baseline_trials}")
            print_comparison_table(stats_data, baseline_stats)
            
            # Summary statistics
            curr_total = sum(s['cumtime'] for s in stats_data[:5])
            base_total = sum(s['cumtime'] for s in baseline_stats[:5])
            change = ((curr_total - base_total) / base_total * 100) if base_total > 0 else 0
            
            print(f"\n📊 Overall Performance Change: {change:+.2f}%")
            if change < -5:
                print("   ✓ \033[92mSignificant improvement!\033[0m")
            elif change > 5:
                print("   ⚠ \033[91mPerformance regression detected\033[0m")
            else:
                print("   ≈ Similar performance")
                
        except FileNotFoundError as e:
            print(f"\n⚠️  {e}")
    
    # Save as benchmark if requested
    if save_as:
        metadata = {
            'num_values': num_values,
            'num_trials': num_trials,
            'timestamp': datetime.now().isoformat(),
        }
        save_benchmark(stats_data, save_as, metadata)
    
    print("\n✅ Profiling complete.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Profile DDSketch operations with benchmarking capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic profile with multiple trials
  python profile_ddsketch.py --num-trials 5
  
  # Save averaged baseline (recommended: use 3-5 trials)
  python profile_ddsketch.py --num-trials 5 --save-as baseline
  
  # Compare against baseline
  python profile_ddsketch.py --num-trials 5 --compare-to baseline
  
  # Save and compare in one run
  python profile_ddsketch.py --num-trials 5 --save-as optimized --compare-to baseline
  
  # List all saved benchmarks
  python profile_ddsketch.py --list
  
  # Quick test with fewer values
  python profile_ddsketch.py --num-values 1000000 --num-trials 3
        """
    )
    
    parser.add_argument('--num-values', type=int, default=10_000_000,
                        help='Number of values to insert per trial (default: 10,000,000)')
    parser.add_argument('--num-trials', type=int, default=1,
                        help='Number of trials to run and average (default: 1, recommended: 3-5)')
    parser.add_argument('--save-as', type=str,
                        help='Save results as a benchmark with this name')
    parser.add_argument('--compare-to', type=str,
                        help='Compare results to this benchmark')
    parser.add_argument('--list', action='store_true',
                        help='List all available benchmarks')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Number of top functions to display (default: 20)')
    
    args = parser.parse_args()
    
    if args.list:
        list_benchmarks()
    else:
        profile(
            num_values=args.num_values,
            save_as=args.save_as,
            compare_to=args.compare_to,
            top_n=args.top_n,
            num_trials=args.num_trials
        )


if __name__ == "__main__":
    main() 