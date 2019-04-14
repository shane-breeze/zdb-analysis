#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd

from zdb.modules.yaml_process import yaml_read, create_query_string
from zdb.modules.db_query_to_frame import db_query_to_frame

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

    # setup queries
    cfg = yaml_read(options.config)
    cfg_q = cfg["query"]
    hist_dict = cfg_q["histograms"]["METnoX_pt"]
    print("Setup jobs args")
    queries = create_query_string(
        cfg_q["template"], hist_dict, aliases=cfg_q["aliases"],
    )
    args = [
        (dbpath, queries)
        for dbpath in cfg["database"]
    ]

    parallel = build_parallel(
        options.mode, processes=options.ncores, quiet=False,
        dispatcher_options={"vmem": 6, "walltime": 3*60*60},
    )
    parallel.begin()
    try:
        print("Submitting jobs")
        parallel.communicationChannel.put_multiple([{
            'task': db_query_to_frame,
            'args': sub_args,
            'kwargs': {},
        } for sub_args in args])
        results = parallel.communicationChannel.receive()
    except KeyboardInterrupt:
        parallel.terminate()
    parallel.end()

    df = None
    for result in results:
        dfr = result[0].set_index(cfg_q["index"])
        if df is None:
            df = dfr
        else:
            df = (
                df.reindex_like(df+dfr).fillna(0)
                + dfr.reindex_like(df+dfr).fillna(0)
            )

    print(df)
    df.to_csv(options.output, float_format="%.12f")

if __name__ == "__main__":
    main()
