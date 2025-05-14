import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define your strategies (policies) here; update this list to match your actual experiment
STRATEGIES = ['DQN', 'A2C', 'PPO', 'LRU', 'LFU', 'TTL', 'NONE']

def load_data(path='cache_metrics.json'):
    """
    Load the cache metrics data from JSON, parse timestamps, extract policy, and ensure numeric columns.
    """
    date_fmt = "%Y-%m-%d %H:%M:%S.%f"
    df = pd.read_json(path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format=date_fmt, errors='coerce')
    if 'rl_action_taken' in df.columns:
        df['policy'] = df['rl_action_taken'].str.extract(r'policy=([^,]+)')
    elif 'policy' not in df.columns:
        raise ValueError("Cannot find 'policy' or 'rl_action_taken' in data columns.")
    df['size_bytes'] = pd.to_numeric(df.get('size_bytes', 0), errors='coerce').fillna(0).astype(int)
    df['hit_ratio'] = pd.to_numeric(df.get('hit_ratio', 0), errors='coerce').fillna(0)
    df['load_time_ms'] = pd.to_numeric(df.get('load_time_ms', 0), errors='coerce').fillna(0)
    # Optionally: fill missing workload_pattern with "unknown"
    if 'workload_pattern' not in df.columns:
        df['workload_pattern'] = "unknown"
    return df

def plot_rq1(df):
    """
    RQ1: Hit Ratio vs Storage Used (MB), colored by policy, with Pareto front for each policy.
    """
    plt.figure(figsize=(9,7))
    for policy in STRATEGIES:
        grp = df[df['policy'] == policy]
        if not grp.empty:
            plt.scatter(grp['size_bytes']/1e6, grp['hit_ratio'], label=policy, alpha=0.7)
            # Pareto front: sorted by storage, keep best-so-far hit_ratio
            pareto = grp.sort_values('size_bytes')
            best = []
            max_hr = -np.inf
            for _, row in pareto.iterrows():
                if row['hit_ratio'] > max_hr:
                    best.append(row)
                    max_hr = row['hit_ratio']
            if best:
                pareto_points = pd.DataFrame(best)
                plt.plot(pareto_points['size_bytes']/1e6, pareto_points['hit_ratio'],
                         linestyle='--', alpha=0.4, label=f"{policy} Pareto")
    plt.xlabel('Storage Used (MB)')
    plt.ylabel('Cache Hit Ratio')
    plt.title('RQ1: Hit Ratio vs Storage by Strategy')
    plt.legend(title='Policy', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

def plot_rq2(df):
    """
    RQ2: Hit Ratio over Time, for each policy, optionally faceted by workload pattern.
    """
    workloads = df['workload_pattern'].unique()
    for workload in workloads:
        plt.figure(figsize=(10,6))
        subset = df[df['workload_pattern'] == workload]
        for policy in STRATEGIES:
            grp = subset[subset['policy'] == policy].sort_values('timestamp')
            if not grp.empty:
                plt.plot(grp['timestamp'], grp['hit_ratio'], marker='o', label=policy)
        plt.xlabel('Time')
        plt.ylabel('Cache Hit Ratio')
        title = f'RQ2: Hit Ratio Over Time by Strategy'
        if workload != "unknown":
            title += f' (Workload: {workload})'
        plt.title(title)
        plt.legend(title='Policy', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

def plot_rq3(df):
    """
    RQ3: Average metrics per strategy (hit_ratio, storage, latency), grouped bar plot, by workload.
    """
    workloads = df['workload_pattern'].unique()
    for workload in workloads:
        subset = df[df['workload_pattern'] == workload]
        summary = (subset.groupby('policy')
                         .agg(avg_hit_ratio=('hit_ratio','mean'),
                              avg_storage=('size_bytes','mean'),
                              avg_latency=('load_time_ms','mean'))
                         .reindex(STRATEGIES)
                         .dropna()
                         .reset_index())
        x = np.arange(len(summary))
        width = 0.25
        plt.figure(figsize=(11,6))
        plt.bar(x - width, summary['avg_hit_ratio'], width, label='Avg Hit Ratio')
        plt.bar(x, summary['avg_storage']/1e6, width, label='Avg Storage (MB)')
        plt.bar(x + width, summary['avg_latency'], width, label='Avg Latency (ms)')
        plt.xlabel('Strategy')
        title = 'RQ3: Average Metrics per Strategy'
        if workload != "unknown":
            title += f' (Workload: {workload})'
        plt.title(title)
        plt.xticks(x, summary['policy'])
        plt.legend()
        plt.tight_layout()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rq', type=int, choices=[1,2,3], default=None,
                        help='Research question number to plot (1, 2, or 3). If omitted, will plot all.')
    parser.add_argument('--file', type=str, default='cache_metrics.json',
                        help='Path to cache metrics JSON file.')
    args = parser.parse_args()

    df = load_data(args.file)

    if args.rq == 1:
        plot_rq1(df)
    elif args.rq == 2:
        plot_rq2(df)
    elif args.rq == 3:
        plot_rq3(df)
    else:
        plot_rq1(df)
        plot_rq2(df)
        plot_rq3(df)

    plt.show()

if __name__ == '__main__':
    main()