#!/usr/bin/env python
import os
import argparse
import numpy as np
import pandas as pd
import importlib
import yaml
import copy
import pysge

from zdb.modules.multirun import multidraw

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input file")
    parser.add_argument("cfg", help="Plotting config file")
    parser.add_argument("drawer", help="Path to drawing function")
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
        help="SGE job options",
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
        njobs = len(jobs)

    grouped_jobs = [list(x) for x in np.array_split(jobs, njobs)]
    tasks = [
        {"task": multidraw, "args": (draw, args), "kwargs": {}}
        for args in grouped_jobs
    ]

    if options.mode=="multiprocessing" and options.ncores==0:
        pysge.local_submit(tasks)
    elif options.mode=="multiprocessing":
        pysge.mp_submit(tasks, ncores=options.ncores)
    elif options.mode=="sge":
        pysge.sge_submit(
            "zdb-draw", "_ccsp_temp/", tasks=tasks, options=options.sge_opts,
            sleep=5, request_resubmission_options=True,
        )

def main():
    options = parse_args()

    # Setup drawer function
    module_name, function_name = options.drawer.split(":")
    draw = getattr(importlib.import_module(module_name), function_name)

    # open cfg
    with open(options.cfg, 'r') as f:
        cfg = yaml.full_load(f)

    # Read in dataframes
    df_data = pd.read_hdf(options.input, "DataAggEvents")
    df_data = df_data.loc[("central",), :]
    df_mc = pd.read_hdf(options.input, "MCAggEvents")
    df_mc = df_mc.loc[("central",), :]

    # process MC dataframe
    if df_mc is not None:
        df_mc = rename_df_index(df_mc, "parent", [
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
        df_mc = df_mc.groupby(df_mc.index.names).sum()

    # dfs
    dfs = []
    if df_data is not None:
        dfs.append(df_data)
    if df_mc is not None:
        dfs.append(df_mc)

    # varnames
    varnames = pd.concat(dfs).index.get_level_values("varname0").unique()

    # datasets
    if df_data is not None:
        datasets = df_data.index.get_level_values("parent").unique()
    else:
        datasets = ["None"]

    # cutflows
    cutflows = pd.concat(dfs).index.get_level_values("selection").unique()

    # group into histograms
    jobs = []
    for varname in varnames:
        for dataset in datasets:
            for cutflow in cutflows:
                job_cfg = copy.deepcopy(cfg[varname])
                job_cfg.update(cfg.get("defaults", {}))
                job_cfg.update(cfg.get(dataset+"_dataset", {}))
                job_cfg.update(cfg.get(cutflow, {}))
                job_cfg.update(cfg.get(dataset+"_dataset", {}).get(cutflow, {}))
                job_cfg.update(cfg.get(dataset+"_dataset", {}).get(cutflow, {}).get(varname, {}))
                outdir = os.path.join(options.outdir, dataset, cutflow)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                job_cfg["outpath"] = os.path.abspath(
                    os.path.join(outdir, cfg[varname]["outpath"])
                )

                # data selection
                if df_data is None or (varname, cutflow, dataset) not in df_data.index:
                    df_data_loc = None
                else:
                    df_data_loc = df_data.loc[(varname, cutflow, dataset),:]

                # mc selection
                if df_mc is None or (varname, cutflow) not in df_mc.index:
                    df_mc_loc = None
                else:
                    df_mc_loc = df_mc.loc[(varname, cutflow),:]

                jobs.append((df_data_loc, df_mc_loc, copy.deepcopy(job_cfg)))

    if options.nplots >= 0 and options.nplots < len(jobs):
        jobs = jobs[:options.nplots]
    parallel_draw(draw, jobs, options)

if __name__ == "__main__":
    main()
