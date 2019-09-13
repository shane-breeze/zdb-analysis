#!/usr/bin/env python
import argparse
import importlib
from zdb.modules.draw import submit_draw_data_mc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Input file")
    parser.add_argument("drawer", type=str, help="Path to drawing function")
    parser.add_argument("cfg", type=str, help="Plotting config file")
    parser.add_argument("outdir", type=str, help="Output directory")
    parser.add_argument(
        "-n", "--nplots", default=-1, type=int,
        help="Number of plots to draw. -1 = all",
    )
    parser.add_argument(
        "-m", "--mode", default="multiprocessing", type=str,
        help="Parallelisation: 'multiprocessing', 'sge', 'htcondor'",
    )
    parser.add_argument(
        "-j", "--ncores", default=0, type=int,
        help="Number of cores for 'multiprocessing' jobs",
    )
    parser.add_argument(
        "--batch-opts", default="-q hep.q", type=str,
        help="SGE job options",
    )
    return parser.parse_args()

def main():
    options = parse_args()
    module_name, function_name = options.drawer.split(":")
    drawer = getattr(importlib.import_module(module_name), function_name)
    options.drawer = drawer
    submit_draw_data_mc(**vars(options))

if __name__ == "__main__":
    main()
