import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

def process_mc(df, cfg):
    if df is None:
        return None

    indexes = df.index.names
    df = df.reset_index()
    for old_name, new_name in cfg["rename_samples"].items():
        df.loc[df[cfg["sample"]]==old_name,cfg["sample"]] = new_name
    df = df.groupby(indexes).sum()

    binning = np.arange(*cfg["binning"])
    indexes = df.index.names
    df = df.reset_index()
    mask = (df[cfg["binvar"]] < binning[0])
    df.loc[mask,cfg["binvar"]] = 2*binning[0] - binning[1]
    df.loc[~mask,cfg["binvar"]] = binning[binning.searchsorted(df.loc[~mask,cfg["binvar"]], side='right')-1]
    df = df.groupby(indexes).sum()

    return df

def create_mc_counts_variance(df, cfg):
    if df is None:
        return None, None

    # Pivot MC tables on samples
    df_counts = pd.pivot_table(
        df, values=cfg["counts"], index=cfg["binvar"], columns=cfg["sample"],
        aggfunc='sum', fill_value=0.,
    )
    df_variance = pd.pivot_table(
        df, values=cfg["variance"], index=cfg["binvar"],
        columns=cfg["sample"], aggfunc='sum', fill_value=0.,
    )

    # Sort MC by sum across bins
    columns = df_counts.sum().sort_values().index.tolist()
    if cfg["process_at_bottom"] in columns:
        columns.remove(cfg["process_at_bottom"])
        columns = [cfg["process_at_bottom"]]+columns
    df_counts = df_counts.reindex(columns, axis=1)

    return df_counts, df_variance

def process_data(df, cfg):
    if df is None:
        return None

    binning = np.arange(*cfg["binning"])
    indexes = df.index.names
    df = df.reset_index()
    mask = (df[cfg["binvar"]] < binning[0])
    df.loc[mask,cfg["binvar"]] = 2*binning[0] - binning[1]
    df.loc[~mask,cfg["binvar"]] = binning[binning.searchsorted(df.loc[~mask,cfg["binvar"]], side='right')-1]
    df = df.groupby(indexes).sum()

    return df

def draw_data(ax, df, cfg):
    if df is None:
        return

    binning = np.arange(*cfg["binning"])
    binvars = df.index.get_level_values(cfg["binvar"])
    bincents = (binning[1:] + binning[:-1])/2.
    idx_dummy = pd.DataFrame({
        "binvar": binning[:-1], cfg["counts"]: [0]*len(binning[:-1]),
        cfg["variance"]: [0]*len(binning[:-1]),
    }).set_index("binvar")
    df = df.reindex_like(idx_dummy)

    if not cfg["blind"]:
        ax.errorbar(
            bincents, df[cfg["counts"]],
            yerr=np.sqrt(df[cfg["variance"]]), fmt='o', ms=4, lw=0.6,
            capsize=2.5, color='black', label="Data",
        )

def draw_mc_counts(ax, df, cfg):
    if df is None:
        return

    binning = np.arange(*cfg["binning"])
    binvars = df.index.get_level_values(cfg["binvar"])
    ax.hist(
        binvars, binning, weights=df.sum(axis=1),
        histtype='step', color='black', label=r'',#label=r'SM Total',
    )
    columns = df.sum().sort_values().index.tolist()
    ax.hist(
        [binvars]*df.shape[1], binning, weights=df.values,
        histtype='stepfilled', stacked=True, log=True,
        color=[cfg["process_colors"][p] for p in columns],
        label=[cfg["process_names"][p] for p in columns],
    )

def draw_legend(axt, axb, df_mc, df_data, cfg):
    handles, labels = axt.get_legend_handles_labels()

    if df_mc is not None:
        fractions = (df_mc.sum(axis=0) / df_mc.sum().sum()).values[::-1]
        fraction_labels = ["{:.3f}".format(f) for f in fractions]

        if df_data is not None and not cfg["blind"]:
            fraction_labels = [
                "{:.3f}".format(df_data[cfg["counts"]].sum().sum()/df_mc.sum().sum())
            ] + fraction_labels

            data_idx = labels.index("Data")
            data_label = labels.pop(data_idx)
            labels = [data_label]+labels
            data_handle = handles.pop(data_idx)
            handles = [data_handle]+handles

        blank_handles = [
            ptch.Rectangle((0,0), 0, 0, fill=False, edgecolor='none', visible=False)
        ]*len(fraction_labels)

        if cfg["legend_off_axes"]:
            box = axt.get_position()
            axt.set_position([box.x0, box.y0, box.width*0.8, box.height])
            axt.legend(
                handles+blank_handles, labels+fraction_labels, ncol=2,
                bbox_to_anchor=(1, 1), handleheight=1.6, labelspacing=0.05,
                columnspacing=-2,
            )
            box = axb.get_position()
            axb.set_position([box.x0, box.y0, box.width*0.8, box.height])
        else:
            axt.legend(
                handles+blank_handles, labels+fraction_labels, ncol=2,
                handleheight=0.8, labelspacing=0.05, columnspacing=-2,
            )
    else:
        handles, labels = axt.get_legend_handles_labels()
        if cfg["legend_off_axes"]:
            box = axt.get_position()
            axt.set_position([box.x0, box.y0, box.width*0.8, box.height])
            axt.legend(handles, labels, bbox_to_anchor=(1, 1))
            box = axb.get_position()
            axb.set_position([box.x0, box.y0, box.width*0.8, box.height])
        else:
            axt.legend(handles, labels)

    handles, labels = axb.get_legend_handles_labels()
    if cfg["legend_off_axes"]:
        axb.legend(handles, labels, bbox_to_anchor=(1, 1))
    else:
        axb.legned(handles, labels)

def draw_cms_header(ax):
    ax.text(
        0, 1, r'$\mathbf{CMS}\ \mathit{Preliminary}$',
        ha='left', va='bottom', transform=ax.transAxes, fontsize=12,
    )
    ax.text(
        1, 1, r'$35.9\ \mathrm{fb}^{-1}(13\ \mathrm{TeV})$',
        ha='right', va='bottom', transform=ax.transAxes, fontsize=12,
    )

def draw_ratio(ax, df, cfg):
    if df is None or cfg["blind"]:
        return

    binning = np.arange(*cfg["binning"])
    binvars = df.index.get_level_values(cfg["binvar"])
    bincents = (binning[1:] + binning[:-1])/2.
    idx_dummy = pd.DataFrame({
        "binvar": binning[:-1], "ratio": [0]*len(binning[:-1]),
        "data_err": [0]*len(binning[:-1]), "mc_err": [0]*len(binning[:-1]),
    }).set_index("binvar")
    df = df.reindex_like(idx_dummy)

    ax.errorbar(
        bincents, df["ratio"], yerr=df["data_err"], fmt='o', ms=4, lw=0.6,
        capsize=2.5, color='black', label='',
    )

    ax.fill_between(
        binning, list(1.-df["mc_err"])+[1.], list(1.+df["mc_err"])+[1.],
        step='post', color='#aaaaaa', label='MC stat. unc.',
    )

def draw_data_mc(df_data, df_mc, cfg):
    plt.style.use('cms')

    df_mc = process_mc(df_mc, cfg)
    df_mc_counts, df_mc_variance = create_mc_counts_variance(df_mc, cfg)
    df_data = process_data(df_data, cfg)

    fig, (axt, axb) = plt.subplots(
        figsize = (4.8, 6), nrows=2, ncols=1, sharex='col', sharey=False,
        gridspec_kw={'height_ratios': [2.5, 1], 'wspace': 0.1, 'hspace': 0.1},
    )

    draw_data(axt, df_data, cfg)
    draw_mc_counts(axt, df_mc_counts, cfg)

    if df_mc is not None and df_data is not None:
        df_ratio = pd.DataFrame({
            "ratio": df_data["sum_w"] / df_mc_counts.sum(axis=1),
            "data_err": np.sqrt(df_data["sum_ww"]) / df_mc_counts.sum(axis=1),
            "mc_err": np.sqrt(df_mc_variance.sum(axis=1)) / df_mc_counts.sum(axis=1),
        }, index=df_mc_counts.index)
    else:
        df_ratio = None
    draw_ratio(axb, df_ratio, cfg)

    draw_cms_header(axt)
    draw_legend(axt, axb, df_mc_counts, df_data, cfg)

    ylims = axt.get_ylim()
    binning = np.arange(*cfg["binning"])
    axt.set_xlim(binning[0], binning[-1])
    axt.set_ylim(max(ylims[0], 0.5), ylims[1])
    axt.set_ylabel("Number of events", fontsize=12)

    ylims = axb.get_ylim()
    axb.set_ylim(max(ylims[0], 0.5), min(ylims[1], 1.5))
    axb.set_xlabel(cfg["label"], fontsize=12)
    axb.set_ylabel("Data/Simulation", fontsize=12)

    axb.axhline(1., lw=0.8, ls='--', color='black')

    print("Creating {}".format(cfg["outpath"]))
    fig.align_ylabels([axt, axb])
    fig.savefig(cfg["outpath"], format="pdf", bbox_inches="tight")
    plt.close(fig)

    return True
