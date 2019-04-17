#!/usr/bin/env python
import argparse
import glob
import gzip
import pandas as pd
import yaml

try:
    import cPickle as pickle
except ImportError:
    import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Wildcard path to results")
    parser.add_argument("cfg", type=str, help="Config file")
    parser.add_argument(
        "-o", "--output", default="output.csv", type=str, help="Output file",
    )
    return parser.parse_args()

def read_result(path):
    results = []
    for p in glob.glob(path):
        print("Reading {}".format(p))
        with gzip.open(p, 'rb') as f:
            yield pickle.load(f)

def read_cfg(path):
    with open(path, 'r') as f:
        return yaml.load(f)

def merge_results(path, cfg, output):
    df = None
    for result in read_result(path):
        dfr = result[0]
        if dfr.empty:
            continue
        if df is None:
            df = dfr
        else:
            df = (
                df.reindex_like(df+dfr).fillna(0)
                + dfr.reindex_like(df+dfr).fillna(0)
            )

    print(df)
    df.to_csv(output, float_format="%.12f")

def main():
    options = parse_args()
    merge_results(options.path, read_cfg(options.cfg), options.output)

if __name__ == "__main__":
    main()
