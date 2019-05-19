#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
import pysge
import tqdm

from zdb.modules.yaml_process import yaml_read, create_query_string
from zdb.modules.db_query_to_frame import db_query_to_frame, merge_results
from zdb.modules.multirun import multirun

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

    # setup queries
    cfg = yaml_read(options.config)
    cfg_q = cfg["query"]

    queries = []
    for hist_label, hist_dict in cfg_q["histograms"].items():
        queries.extend(create_query_string(
            cfg_q["template"], hist_dict, aliases=cfg_q["aliases"],
        ))
    queries = " UNION ".join(queries)

    databases = cfg["database"]
    if options.nfiles >= 0 and options.nfiles < len(databases):
        databases = databases[:options.nfiles]
    jobs = [
        (db_query_to_frame, (dbpath, queries))
        for dbpath in databases
    ]

    # group jobs
    if mode in ["multiprocessing"]:
        njobs = len(jobs)+1

    jobs = [list(x) for x in np.array_split(jobs, njobs)]
    tasks = [
        {"task": multirun, "args": args, "kwargs": {"index": cfg_q["index"]}}
        for args in jobs
    ]

    if mode=="multiprocessing" and options.ncores==0:
        results = pysge.local_submit(tasks)
        df = merge_results(results, cfg_q["index"]).set_index(cfg_q["index"])
    elif mode=="multiprocessing":
        results = pysge.mp_submit(tasks, ncores=options.ncores)
        df = merge_results(results, cfg_q["index"]).set_index(cfg_q["index"])
    elif mode=="sge":
        df, merged_idx = None, []
        for results in pysge.sge_submit_yield(
            "zdb", "_ccsp_temp/", tasks=tasks,
            options=options.sge_opts, sleep=5,
            request_resubmission_options=True,
        ):
            to_merge = []
            for idx, r in enumerate(results):
                if r is None or idx in merged_idx:
                    continue
                to_merge.append(r)
                merged_idx.append(idx)
            if len(to_merge)>0:
                if df is not None:
                    to_merge = [(df.reset_index(),)] + to_merge
                df = (
                    merge_results(to_merge, cfg_q["index"], disable=True)
                    .set_index(cfg_q["index"])
                )

    else:
        df = merge_results([], cfg_q["index"]).set_index(cfg_q["index"])
    print(df)
    df.to_pickle(options.output)

if __name__ == "__main__":
    main()
