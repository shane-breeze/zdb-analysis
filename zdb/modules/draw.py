import os
import copy
import pysge
import oyaml as yaml
import numpy as np
import pandas as pd

from zdb.modules.multirun import multidraw

def parallel_draw(drawer, jobs, mode, ncores, batch_opts):
    if len(jobs)==0:
        return

    njobs = ncores
    if mode in ["multiprocessing"]:
        njobs = len(jobs)

    grouped_jobs = [list(x) for x in np.array_split(jobs, njobs)]
    tasks = [
        {"task": multidraw, "args": (drawer, args), "kwargs": {}}
        for args in grouped_jobs
    ]

    if mode=="multiprocessing" and ncores==0:
        pysge.local_submit(tasks)
    elif mode=="multiprocessing":
        pysge.mp_submit(tasks, ncores=ncores)
    elif mode=="sge":
        pysge.sge_submit(
            tasks, "zdb-draw", "_ccsp_temp/", options=batch_opts,
            sleep=5, request_resubmission_options=True, return_files=True,
        )

def submit_draw_data_mc(
    infile, drawer, cfg, outdir, nplots=-1, mode="multiprocessing", ncores=0,
    batch_opts="-q hep.q",
):
    with open(cfg, 'r') as f:
        cfg = yaml.full_load(f)

    # Read in dataframes
    df_data = pd.read_hdf(infile, "DataAggEvents")
    df_data = df_data.loc[("central",), :]
    df_mc = pd.read_hdf(infile, "MCAggEvents")
    df_mc = df_mc.loc[("central",), :]

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
                if varname not in cfg:
                    continue
                job_cfg = copy.deepcopy(cfg[varname])
                job_cfg.update(cfg.get("defaults", {}))
                job_cfg.update(cfg.get(dataset+"_dataset", {}))
                job_cfg.update(cfg.get(cutflow, {}))
                job_cfg.update(cfg.get(dataset+"_dataset", {}).get(cutflow, {}))
                job_cfg.update(cfg.get(dataset+"_dataset", {}).get(cutflow, {}).get(varname, {}))
                toutdir = os.path.join(outdir, dataset, cutflow)
                if not os.path.exists(toutdir):
                    os.makedirs(toutdir)
                job_cfg["outpath"] = os.path.abspath(
                    os.path.join(toutdir, cfg[varname]["outpath"])
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

    if nplots >= 0 and nplots < len(jobs):
        jobs = jobs[:nplots]
    parallel_draw(drawer, jobs, mode, ncores, batch_opts)
