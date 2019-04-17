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

logging.getLogger(__name__).propagate = False
logging.getLogger("atsge.SGEJobSubmitter").propagate = False

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
        "-o", "--output", default="output.csv", type=str, help="Output file",
    )
    return parser.parse_args()

def main():
    options = parse_args()
    njobs = 1 if options.mode in ["multiprocessing"] else options.ncores

    # setup queries
    cfg = yaml_read(options.config)
    cfg_q = cfg["query"]

    for hist_label, hist_dict in cfg_q["histograms"].items():
        print("Setup jobs args")
        queries = create_query_string(
            cfg_q["template"], hist_dict, aliases=cfg_q["aliases"],
        )
        jobs = [
            (db_query_to_frame, (dbpath, queries))
            for dbpath in cfg["database"]
        ]

        # group jobs
        jobs = [
            jobs[i:i+len(jobs)/njobs+1]
            for i in xrange(0, len(jobs), len(jobs)/njobs+1)
        ]

        parallel = build_parallel(
            options.mode, processes=options.ncores, quiet=False,
            dispatcher_options={"vmem": 6, "walltime": 3*60*60},
        )
        parallel.begin()
        try:
            print("Submitting jobs")
            parallel.communicationChannel.put_multiple([{
                'task': multirun,
                'args': args,
                'kwargs': {"index": cfg_q["index"]},
            } for args in jobs])
            results = parallel.communicationChannel.receive()
        except KeyboardInterrupt:
            parallel.terminate()
        parallel.end()

        df = (
            merge_results(results, cfg_q["index"])
            .set_index(cfg_q["index"])
        )
        print(df)
        df.to_csv(options.output.format(hist_label), float_format="%.12f")

if __name__ == "__main__":
    main()
