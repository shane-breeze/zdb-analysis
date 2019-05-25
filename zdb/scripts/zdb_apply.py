#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
import pysge
import tqdm

from zdb.modules.yaml_process import yaml_read, create_query_string
from zdb.modules.db_query_to_frame import db_query_to_frame_processing, merge_results
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

    query_eval_groupby = []
    for hist_label, hist_dict in cfg_q["histograms"].items():
        query = " UNION ".join(create_query_string(
            cfg_q["template"], hist_dict, aliases=cfg_q["aliases"],
        ))
        query_eval_groupby.append((
            query, hist_dict["evals"], hist_dict["groupby"],
        ))

    databases = cfg["database"]
    if options.nfiles >= 0 and options.nfiles < len(databases):
        databases = databases[:options.nfiles]

    jobs = [
        (db_query_to_frame_processing, (dbpath, query_eval_groupby))
        for dbpath in databases
    ]

    # group jobs
    if mode in ["multiprocessing"]:
        njobs = len(jobs)

    jobs = [list(x) for x in np.array_split(jobs, njobs)]
    tasks = [
        {"task": multirun, "args": args, "kwargs": {"index": cfg_q["index"]}}
        for args in jobs
    ]

    if mode=="multiprocessing" and options.ncores==0:
        results = [
            r for r in pysge.local_submit(tasks)
            if not r[0].empty
        ]
        if len(results)!=0:
            df = merge_results(results, cfg_q["index"]).set_index(cfg_q["index"])
        else:
            df = None
    elif mode=="multiprocessing":
        results = [
            r for r in pysge.mp_submit(tasks, ncores=options.ncores)
            if not r[0].empty
        ]
        if len(results)!=0:
            df = merge_results(results, cfg_q["index"]).set_index(cfg_q["index"])
        else:
            df = None
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
                if df is not None and not df.empty:
                    to_merge = [(df.reset_index(),)] + to_merge
                df = merge_results(to_merge, cfg_q["index"], disable=True)
                if df is not None:
                    df = df.set_index(cfg_q["index"])

    else:
        df = merge_results([], cfg_q["index"]).set_index(cfg_q["index"])
    print(df)
    if df is None:
        print("No events passed the selection. Nothing to save")
    else:
        df.to_pickle(options.output)
        df.to_hdf(
            options.output.split(".")[0]+".h5", 'AggEvents', format='table',
            complevel=9, complib='blosc:lz4hc',
        )

if __name__ == "__main__":
    main()
