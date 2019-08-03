#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
import pysge
import oyaml as yaml

from zdb.modules.df_skim import df_skim

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
        "--batch-opts", default="", type=str,
        help="Options to pass onto the batch submission",
    )
    parser.add_argument(
        "-n", "--nfiles", default=-1, type=int,
        help="Number of files to process. -1 = all",
    )
    parser.add_argument(
        "-o", "--output", default="output.h5", type=str, help="Output file",
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
        {"task": df_skim, "args": (fs,cfg,options.output.format(idx)), "kwargs": {}}
        for idx, fs in enumerate(grouped_files)
    ]

    if mode=="multiprocessing" and options.ncores==0:
        results = pysge.local_submit(tasks)
    elif mode=="multiprocessing":
        results = pysge.mp_submit(tasks, ncores=options.ncores)
    elif mode=="sge":
        results = pysge.sge_submit(
            "zdb", "_ccsp_temp/", tasks=tasks, options=options.sge_opts,
            sleep=5, request_resubmission_options=True,
        )
    print("Finished!")

if __name__ == "__main__":
    main()
