import time

import pandas as pd
import yaml

from crystalsizer3d import LOGS_PATH, logger

SEQUENCES_LOGS_PATH = LOGS_PATH.parent / 'track_sequence'


def collect_args_data():
    """
    Collect args data from all runs in the sequences logs.
    """
    args_data = []
    for sequence_path in SEQUENCES_LOGS_PATH.iterdir():
        if not sequence_path.is_dir():
            continue
        runs_path = sequence_path / 'runs'
        if not runs_path.exists():
            continue

        # Loop over the 'runs' within each sequence
        for run_path in runs_path.iterdir():
            if not run_path.is_dir():
                continue
            args_path = run_path / 'args.yml'
            if not args_path.exists():
                continue

            # Load args
            with open(args_path, 'r') as f:
                args = yaml.load(f, Loader=yaml.FullLoader)

            # Add identifiers
            args['sequence'] = sequence_path.name
            args['run_dir'] = run_path.name
            args['path'] = str(run_path)

            # Check the refiner cache directory to see how many images have been started
            if 'refiner_dir' in args:
                refiner_cache_dir = run_path.parent.parent / 'refiner' / args['refiner_dir']
                if refiner_cache_dir.exists():
                    args['n_images_processed'] = sum(1 for _ in refiner_cache_dir.iterdir() if _.is_dir())

            args_data.append(args)

    df = pd.DataFrame(args_data)

    return df


def track_sequence_runs():
    """
    Collate all run data into a single spreadsheet.
    """
    start_time = time.time()
    spreadsheet_path = SEQUENCES_LOGS_PATH / 'runs.xlsx'

    # Load existing spreadsheet or create new DataFrame if not found
    try:
        df = pd.read_excel(spreadsheet_path, sheet_name='Runs')
    except FileNotFoundError:
        df = pd.DataFrame()

    # Collect new data and merge
    new_df = collect_args_data()
    if df.empty:
        df = new_df
        writer_args = dict(mode='w')
    else:
        merged_df = pd.merge(df, new_df, on='path', how='outer', suffixes=('', '_new'))

        # Overwrite existing columns with new data where available
        for col in df.columns:
            if col in new_df.columns and col + '_new' in merged_df:
                merged_df[col] = merged_df[col + '_new'].combine_first(merged_df[col])

        # Drop the extra columns that were created during the merge
        merged_df.drop(
            columns=[col + '_new' for col in df.columns if col in new_df.columns and col + '_new' in merged_df.columns],
            inplace=True
        )
        df = merged_df
        writer_args = dict(mode='a', if_sheet_exists='overlay')

    # Sort the dataframe by created time
    df = df.sort_values(by=['created'])

    # Sort the columns alphabetically
    df = df.reindex(sorted(df.columns), axis=1)

    # Put specified columns first
    first_cols = ['created', 'sequence', 'run_dir', 'n_images_processed', 'path',
                  'refiner_dir', 'denoiser_dir', 'keypoints_dir', 'predictor_dir']
    valid_first_cols = [col for col in first_cols if col in df.columns]
    col_order = valid_first_cols + [col for col in df.columns if col not in valid_first_cols]
    df = df[col_order]

    # Write updated data back to spreadsheet
    with pd.ExcelWriter(spreadsheet_path, **writer_args) as writer:
        df.to_excel(writer, index=False, sheet_name='Runs')

    # Print how long this took - split into hours, minutes, seconds
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f'Finished in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.')


if __name__ == '__main__':
    track_sequence_runs()
