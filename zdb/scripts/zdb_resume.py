#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
import pysge
import glob
import os

from zdb.modules.df_process import df_merge, df_open_merge

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to temp dir")
    parser.add_argument(
        "-m", "--mode", default="multiprocessing", type=str,
        help="Parallelisation: 'multiprocessing', 'sge', 'htcondor'",
    )
    parser.add_argument(
        "-j", "--ncores", default=0, type=int,
        help="Number of cores for 'multiprocessing' jobs",
    )
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

    results = pysge.sge_resume(
        "zdb", options.path, options=options.sge_opts, sleep=5,
        request_resubmission_options=True,
    )

    njobs = options.ncores
    if options.mode in ["multiprocessing"] or options.ncores < 0:
        njobs = len(results)

    grouped_args = [list(x) for x in np.array_split(results, njobs)]
    tasks = [
        {"task": df_open_merge, "args": (args,), "kwargs": {"quiet": True}}
        for args in grouped_args
    ]

    if options.mode=="multiprocessing" and options.ncores==0:
        merge_results = pysge.local_submit(tasks)
        df = pd.DataFrame()
        for result in merge_results:
            df = df_merge(df, result)
    elif options.mode=="multiprocessing":
        merge_results = pysge.mp_submit(tasks, ncores=options.ncores)
        df = pd.DataFrame()
        for result in merge_results:
            df = df_merge(df, result)
    elif options.mode=="sge":
        merge_results = pysge.sge_submit(
            "zdb-merge", "_ccsp_temp/", tasks=tasks, options=options.sge_opts,
            sleep=5, request_resubmission_options=True,
        )
        df = df_open_merge(merge_results)
    else:
        df = pd.DataFrame()

    print(df)
    path, table = options.output.split(":")
    df.to_hdf(
        path, table, format='table', append=True, complevel=9,
        complib='blosc:lz4hc',
    )

if __name__ == "__main__":
    main()
