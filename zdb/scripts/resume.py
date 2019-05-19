#!/usr/bin/env python
import argparse
import glob
import gzip
import yaml
import tqdm

from zdb.modules.db_query_to_frame import merge_results

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
    for p in tqdm.tqdm(glob.glob(path), desc="Reading", dynamic_ncols=True, unit=" files"):
        #print("Reading {}".format(p))
        with gzip.open(p, 'rb') as f:
            yield pickle.load(f)

def read_cfg(path):
    with open(path, 'r') as f:
        return yaml.load(f)

def main():
    options = parse_args()

    index = read_cfg(options.cfg)["query"]["index"]
    df = merge_results(list(read_result(options.path)), index).set_index(index)
    print(df)
    df.to_pickle(options.output)

if __name__ == "__main__":
    main()
