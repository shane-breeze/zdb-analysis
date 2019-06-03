#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
import pysge
import oyaml as yaml
import functools
import tqdm

from zdb.modules.df_process import df_process, df_merge, df_open_merge

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
    if mode in ["multiprocessing"] or njobs < 0:
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
        results = pysge.sge_submit(
            "zdb", "_ccsp_temp/", tasks=tasks, options=options.sge_opts,
            sleep=5, request_resubmission_options=True,
        )

        grouped_args = [list(x) for x in np.array_split(results, 75)]
        tasks = [
            {"task": df_open_merge, "args": (args,), "kwargs": {"quiet": True}}
            for args in grouped_args
        ]
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
        path, table, format='table', append=False, complevel=9,
        complib='blosc:lz4hc',
    )

if __name__ == "__main__":
    main()
