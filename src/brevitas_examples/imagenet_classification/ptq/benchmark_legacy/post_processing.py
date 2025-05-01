import os

import pandas as pd


def main():
    main_dir = './multirun'

    evals = next(os.walk(main_dir))[1]
    df = None
    for eval in evals:
        full_path = os.path.join(main_dir, eval, 'RESULTS_TORCHVISION.csv')
        if not os.path.exists(full_path):
            continue
        if df is None:
            df = pd.read_csv(full_path)
        else:
            single_df = pd.read_csv(full_path)
            df = pd.concat([df, single_df])
    df = df.sort_values(by=list(df.columns))
    df.to_csv('RESULTS_TORCHVISION.csv', index=False, mode='w')

    grouped_df = df.groupby([
        'Model name',
        'Target backend',
        'Scale factor type',
        'Weight bit width',
        'Act bit width',
        'Bias bit width',
        'Weight quant granularity',
        'Act quant type'])
    idx = grouped_df['Top 1% quant accuracy'].transform(max) == df['Top 1% quant accuracy']
    best_config_df = df[idx]
    best_config_df = best_config_df.sort_values(by=['Model name', 'Top 1% quant accuracy'])
    best_config_df.to_csv('RESULTS_TORCHVISION_BEST_CONFIGS.csv', index=False, mode='w')


if __name__ == '__main__':
    main()
