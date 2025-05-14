import argparse
import pandas as pd
import matplotlib.pyplot as plt

# only consider these strategies, in this order
STRATEGIES = ['DQN', 'A2C', 'PPO', 'LRU', 'LFU', 'TTL', 'NONE']

def load_data():
    date_fmt = "%Y-%m-%d %H:%M:%S.%f"
    df = pd.read_csv(
        'cache_metrics.csv',
        parse_dates=['timestamp'],
        date_parser=lambda x: pd.to_datetime(x, format=date_fmt)
    )
    df['size'] = pd.to_numeric(df['size'], errors='coerce')
    df['size_bytes'] = df['size'].fillna(0).astype(int)
    return df

def plot_rq1(df):
    plt.figure(figsize=(8,6))
    for policy in STRATEGIES:
        grp = df[df['policy'] == policy]
        if not grp.empty:
            plt.scatter(grp['size_bytes'], grp['hit_ratio'], label=policy, alpha=0.7)
    plt.xlabel('Storage (bytes)')
    plt.ylabel('Hit Ratio')
    plt.title('RQ1: Hit Ratio vs Storage by Strategy')
    plt.legend(title='Strategy')
    plt.tight_layout()
    plt.show()

def plot_rq2(df):
    plt.figure(figsize=(10,6))
    for policy in STRATEGIES:
        grp = df[df['policy'] == policy].sort_values('timestamp')
        if not grp.empty:
            plt.plot(grp['timestamp'], grp['hit_ratio'], marker='o', label=policy)
    plt.xlabel('Timestamp')
    plt.ylabel('Hit Ratio')
    plt.title('RQ2: Hit Ratio Over Time by Strategy')
    plt.legend(title='Strategy')
    plt.tight_layout()
    plt.show()

def plot_rq3(df):
    # include observed=False to suppress the FutureWarning
    summary = df.groupby('policy', observed=False).agg(
        avg_hit_ratio=('hit_ratio', 'mean'),
        avg_storage=('size_bytes', 'mean'),
        avg_latency=('load_time_ms', 'mean')
    ).reindex(STRATEGIES).dropna().reset_index()

    x = range(len(summary))
    width = 0.25

    plt.figure(figsize=(8, 6))
    plt.bar([i - width for i in x], summary['avg_hit_ratio'], width, label='Hit Ratio')
    plt.bar(x,                     summary['avg_storage'],   width, label='Storage')
    plt.bar([i + width for i in x], summary['avg_latency'],  width, label='Latency')
    plt.xlabel('Strategy')
    plt.title('RQ3: Average Metrics per Strategy')
    plt.xticks(x, summary['policy'])
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rq', type=int, choices=[1,2,3],
                        help='Research question number to plot')
    args = parser.parse_args()

    df = load_data()

    if args.rq == 1:
        plot_rq1(df)
    elif args.rq == 2:
        plot_rq2(df)
    else:
        # default and rq=3
        plot_rq1(df)
        plot_rq2(df)
        plot_rq3(df)

if __name__ == '__main__':
    main()