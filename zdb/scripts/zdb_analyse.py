#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
import pysge
import oyaml as yaml
import functools

from zdb.modules.df_process import df_process, df_merge

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to yaml file")
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
        "-n", "--nfiles", default=-1, type=int,
        help="Number of files to process. -1 = all",
    )
    parser.add_argument(
        "-o", "--output", default="output.pkl", type=str, help="Output file",
    )
    return parser.parse_args()

def main():
    options = parse_args()
    mode = options.mode
    njobs = options.ncores

    # setup jobs
    with open(options.config, 'r') as f:
        cfg = yaml.full_load(f)

    # group jobs
    files = cfg["files"]
    if options.nfiles > 0:
        files = files[:options.nfiles]
    if mode in ["multiprocessing"]:
        njobs = len(files)

    grouped_files = [list(x) for x in np.array_split(files, njobs)]
    tasks = [
        {"task": df_process, "args": (fs, cfg["query"]), "kwargs": {}}
        for fs in grouped_files
    ]

    if mode=="multiprocessing" and options.ncores==0:
        results = pysge.local_submit(tasks)
        df = functools.reduce(lambda x, y: df_merge(x, y), results)
    elif mode=="multiprocessing":
        results = pysge.mp_submit(tasks, ncores=options.ncores)
        df = functools.reduce(lambda x, y: df_merge(x, y), results)
    elif mode=="sge":
        df = pd.DataFrame()
        merged_idx = []
        for results in pysge.sge_submit_yield(
            "zdb", "_ccsp_temp/", tasks=tasks, options=options.sge_opts,
            sleep=5, request_resubmission_options=True,
        ):
            for idx, r in enumerate(results):
                if r is None or idx in merged_idx:
                    continue
                df = df_merge(df, r)
                merged_idx.append(idx)
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
