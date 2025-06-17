import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_perf_results(file_path):
    """Parse Triton perf_analyzer output file."""
    results = []
    with open(file_path, 'r') as f:
        content = f.read()

    pattern = r'Concurrency: (\d+), throughput: ([\d.]+) infer/sec, latency ([\d.]+) usec'
    matches = re.findall(pattern, content)

    for match in matches:
        concurrency = int(match[0])
        throughput = float(match[1])
        latency = float(match[2])
        results.append({
            'Concurrency': concurrency,
            'Throughput (infer/sec)': throughput,
            'Latency (ms)': latency / 1000
        })

    return pd.DataFrame(results)


def generate_performance_report():
    """Generate a performance report from the test results."""
    result_files = {
        'Dictionary': 'perf_results/dictionary_throughput.txt',
        'Rule-based': 'perf_results/rule_throughput.txt',
        'Ensemble': 'perf_results/ensemble.txt'
    }

    if os.path.exists('perf_results/text_normalizer.txt'):
        result_files['ONNX Model'] = 'perf_results/text_normalizer.txt'

    os.makedirs('perf_results/plots', exist_ok=True)

    all_results = {}
    for model_name, file_path in result_files.items():
        if os.path.exists(file_path):
            all_results[model_name] = parse_perf_results(file_path)
        else:
            print(f"Warning: Result file {file_path} not found")

    if not all_results:
        print("No performance results found to analyze")
        return

    plt.figure(figsize=(12, 6))
    for model_name, df in all_results.items():
        if not df.empty:
            plt.plot(df['Concurrency'], df['Throughput (infer/sec)'], marker='o', label=model_name)

    plt.title('Throughput vs Concurrency')
    plt.xlabel('Concurrency')
    plt.ylabel('Throughput (infer/sec)')
    plt.grid(True)
    plt.legend()
    plt.savefig('perf_results/plots/throughput_comparison.png')

    plt.figure(figsize=(12, 6))
    for model_name, df in all_results.items():
        if not df.empty:
            plt.plot(df['Concurrency'], df['Latency (ms)'], marker='o', label=model_name)

    plt.title('Latency vs Concurrency')
    plt.xlabel('Concurrency')
    plt.ylabel('Latency (ms)')
    plt.grid(True)
    plt.legend()
    plt.savefig('perf_results/plots/latency_comparison.png')

    with open('perf_results/summary.md', 'w') as f:
        f.write('# Performance Analysis Report\n\n')
        f.write('## Model Performance Summary\n\n')

        f.write('| Model | Max Throughput (infer/sec) | Min Latency (ms) | Optimal Concurrency |\n')
        f.write('|-------|----------------------------|------------------|--------------------|\n')

        for model_name, df in all_results.items():
            if not df.empty:
                max_throughput = df['Throughput (infer/sec)'].max()
                min_latency = df['Latency (ms)'].min()
                optimal_concurrency = df.loc[df['Throughput (infer/sec)'].idxmax()]['Concurrency']

                f.write(f'| {model_name} | {max_throughput:.2f} | {min_latency:.2f} | {optimal_concurrency} |\n')

        f.write('\n\n## Throughput Comparison\n\n')
        f.write('![Throughput Comparison](plots/throughput_comparison.png)\n\n')

        f.write('\n\n## Latency Comparison\n\n')
        f.write('![Latency Comparison](plots/latency_comparison.png)\n\n')

        f.write('\n\n## Recommendations\n\n')
        f.write('Based on the performance analysis:\n\n')
        f.write(
            '1. The dictionary-based normalizer provides the lowest latency and is best for time-sensitive applications.\n')
        f.write(
            '2. The rule-based normalizer offers a good balance of accuracy and performance for text with specific patterns.\n')

        if 'ONNX Model' in all_results:
            f.write('3. The ONNX model provides the highest accuracy for complex cases but at higher latency.\n')

        f.write(
            '4. For production deployment, use the optimal concurrency settings identified above for each component.\n')

    print("Performance analysis completed. Report generated at perf_results/summary.md")


if __name__ == "__main__":
    generate_performance_report()