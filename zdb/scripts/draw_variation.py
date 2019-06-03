#!/usr/bin/env python
import os
import re
import argparse
import pandas as pd
import importlib
import yaml
import copy

from zdb.modules.multirun import multidraw
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
    parser.add_argument("mc", help="Path to MC pandas pickle")
    parser.add_argument("drawer", help="Path to drawing function")
    parser.add_argument("cfg", help="Plotting config file")
    parser.add_argument(
        "-m", "--mode", default="multiprocessing", type=str,
        help="Parallelisation: 'multiprocessing', 'sge', 'htcondor'",
    )
    parser.add_argument(
        "-j", "--ncores", default=0, type=int,
        help="Number of cores for 'multiprocessing' jobs",
    )
    parser.add_argument(
        "-n", "--nplots", default=-1, type=int,
        help="Number of plots to draw. -1 = all",
    )
    parser.add_argument(
        "-o", "--outdir", default="temp", type=str, help="Output directory",
    )
    return parser.parse_args()

def rename_df_index(df, index, rename_list):
    if df is None:
        return df
    indexes = df.index.names
    tdf = df.reset_index()
    for new_val, selection in rename_list:
        tdf.loc[tdf.eval(selection),index] = new_val
    return tdf.set_index(indexes)

def parallel_draw(draw, jobs, options):
    if len(jobs)==0:
        return

    njobs = options.ncores
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
            'task': multidraw,
            'args': (draw, args),
            'kwargs': {},
        } for args in jobs])
        results = parallel.communicationChannel.receive()
    except KeyboardInterrupt:
        parallel.terminate()
    parallel.end()

def main():
    options = parse_args()

    # Setup drawer function
    module_name, function_name = options.drawer.split(":")
    draw = getattr(importlib.import_module(module_name), function_name)

    # open cfg
    with open(options.cfg, 'r') as f:
        cfg = yaml.load(f)

    # Read in dataframes
    df = pd.read_pickle(options.mc) if options.mc is not None else None
    df = rename_df_index(df, "parent", [
        ("WJetsToENu", "(parent=='WJetsToLNu') & (LeptonIsElectron==1)"),
        ("WJetsToMuNu", "(parent=='WJetsToLNu') & (LeptonIsMuon==1)"),
        #("WJetsToTauNu", "(parent=='WJetsToLNu') & (LeptonIsTau==1)"),
        ("WJetsToTauHNu", "(parent=='WJetsToLNu') & (LeptonIsTau==1) & (nGenTauL==0)"),
        ("WJetsToTauLNu", "(parent=='WJetsToLNu') & (LeptonIsTau==1) & (nGenTauL==1)"),
        ("DYJetsToEE", "(parent=='DYJetsToLL') & (LeptonIsElectron==1)"),
        ("DYJetsToMuMu", "(parent=='DYJetsToLL') & (LeptonIsMuon==1)"),
        #("DYJetsToTauTau", "(parent=='DYJetsToLL') & (LeptonIsTau==1)"),
        ("DYJetsToTauHTauH", "(parent=='DYJetsToLL') & (LeptonIsTau==1) & (nGenTauL==0)"),
        ("DYJetsToTauHTauL", "(parent=='DYJetsToLL') & (LeptonIsTau==1) & (nGenTauL==1)"),
        ("DYJetsToTauLTauL", "(parent=='DYJetsToLL') & (LeptonIsTau==1) & (nGenTauL==2)"),
    ]).reset_index(["LeptonIsElectron", "LeptonIsMuon", "LeptonIsTau", "nGenTauL"], drop=True)
    df = df.groupby(df.index.names).sum()

    # cutflows
    cutflows = df.index.get_level_values("selection").unique()

    # variations
    varnames = df.index.get_level_values("varname").unique()
    regex = re.compile("^(?P<varname>[a-zA-Z0-9_]+)_(?P<variation>[a-zA-Z0-9]+)(Up|Down)$")

    varname_variations = {}
    for v in varnames:
        match = regex.search(v)
        if match:
            varname = match.group("varname")
            variation = match.group("variation")
            if varname not in varname_variations:
                varname_variations[varname] = []
            if variation not in varname_variations[varname]:
                varname_variations[varname].append(variation)

    # group into histograms
    jobs = []
    for cutflow in cutflows:
        for varname, variations in varname_variations.items():
            for variation in variations:
                job_cfg = copy.deepcopy(cfg[variation])
                job_cfg.update(cfg.get("defaults", {}))
                job_cfg.update(cfg.get(cutflow, {}))
                job_cfg.update(cfg.get(varname, {}))
                outdir = os.path.join(options.outdir, cutflow, varname)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                job_cfg["outpath"] = os.path.abspath(
                    os.path.join(outdir, cfg[variation]["outpath"])
                )

                df_loc = df.loc[
                    (
                        (df.index.get_level_values("selection")==cutflow)
                        & (df.index.get_level_values("varname").isin(
                            [varname+"_nominal", varname+"_"+variation+"Up", varname+"_"+variation+"Down"]
                        ))
                    ), :
                ]
                jobs.append((df_loc, copy.deepcopy(job_cfg)))

    if options.nplots >= 0 and options.nplots < len(jobs):
        jobs = jobs[:options.nplots]
    parallel_draw(draw, jobs, options)

if __name__ == "__main__":
    main()
