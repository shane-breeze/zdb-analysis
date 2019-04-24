#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd

from zdb.modules.yaml_process import yaml_read, create_query_string
from zdb.modules.db_query_to_frame import db_query_to_frame, merge_results
from zdb.modules.multirun import multirun

from atsge.build_parallel import build_parallel

import logging
logging.getLogger(__name__).setLevel(logging.INFO)
logging.getLogger("atsge.SGEJobSubmitter").setLevel(logging.INFO)
logging.getLogger("atsge.WorkingArea").setLevel(logging.INFO)

logging.getLogger(__name__).propagate = False
logging.getLogger("atsge.SGEJobSubmitter").propagate = False
logging.getLogger("atsge.WorkingArea").propagate = False

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
        "-n", "--nfiles", default=-1, type=int,
        help="Number of files to process. -1 = all",
    )
    parser.add_argument(
        "-o", "--output", default="output.pkl", type=str, help="Output file",
    )
    return parser.parse_args()

def main():
    options = parse_args()
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
    if options.mode in ["multiprocessing"]:
        njobs = len(jobs)+1
    jobs = [
        jobs[i:i+len(jobs)//njobs+1]
        for i in xrange(0, len(jobs), len(jobs)//njobs+1)
    ]

    parallel = build_parallel(
        options.mode, processes=options.ncores, quiet=False,
        dispatcher_options={"vmem": 6, "walltime": 3*60*60},
    )
    parallel.begin()
    try:
        parallel.communicationChannel.put_multiple([{
            'task': multirun,
            'args': args,
            'kwargs': {"index": cfg_q["index"]},
        } for args in jobs])
        results = parallel.communicationChannel.receive()
    except KeyboardInterrupt:
        parallel.terminate()
    parallel.end()

    df = merge_results(results, cfg_q["index"]).set_index(cfg_q["index"])
    print(df)
    df.to_pickle(options.output)

if __name__ == "__main__":
    main()
