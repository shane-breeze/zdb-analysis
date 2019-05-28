#!/usr/bin/env python
import argparse
import pandas as pd
import pysge
import tqdm
import glob
import os

from zdb.modules.df_process import df_merge

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to temp dir")
    parser.add_argument(
        "--sge-opts", default="-q hep.q", type=str,
        help="Options to pass onto qsub",
    )
    parser.add_argument(
        "-o", "--output", default="output.pkl", type=str, help="Output file",
    )
    return parser.parse_args()

def main():
    options = parse_args()

    njobs = len(glob.glob(os.path.join(options.path, "task_*")))
    pbar = tqdm.tqdm(total=njobs, desc="Merged", dynamic_ncols=True)

    df = pd.DataFrame()
    merged_idx = []
    for results in pysge.sge_resume(
        "zdb", options.path, options=options.sge_opts, sleep=5,
        request_resubmission_options=True,
    ):
        for idx, r in enumerate(results):
            if r is None or idx in merged_idx:
                continue
            df = df_merge(df, r)
            merged_idx.append(idx)
            pbar.update()

    pbar.close()

    print(df)
    path, table = options.output.split(":")
    df.to_hdf(
        path, table, format='table', append=True, complevel=9,
        complib='blosc:lz4hc',
    )

if __name__ == "__main__":
    main()
