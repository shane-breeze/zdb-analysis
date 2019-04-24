import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

def draw_data_mc(df_data, df_mc, cfg):
    plt.style.use('cms')

    # Rename samples
    indexes = df_mc.index.names
    df_mc = df_mc.reset_index()
    for old_name, new_name in cfg["rename_samples"].items():
        df_mc.loc[df_mc[cfg["sample"]]==old_name,cfg["sample"]] = new_name
    df_mc = df_mc.groupby(indexes).sum()

    # Rebin
    binning = np.arange(*cfg["binning"])

    indexes = df_data.index.names
    df_data = df_data.reset_index()
    df_data.loc[:,cfg["binvar"]] = binning[binning.searchsorted(df_data[cfg["binvar"]])-1]
    df_data = df_data.groupby(indexes).sum()

    indexes = df_mc.index.names
    df_mc = df_mc.reset_index()
    df_mc.loc[:,cfg["binvar"]] = binning[binning.searchsorted(df_mc[cfg["binvar"]])-1]
    df_mc = df_mc.groupby(indexes).sum()

    # Pivot MC tables on samples
    df_mc_counts = pd.pivot_table(
        df_mc, values=cfg["counts"], index=cfg["binvar"], columns=cfg["sample"],
        aggfunc='sum', fill_value=0.,
    )
    df_mc_variance = pd.pivot_table(
        df_mc, values=cfg["variance"], index=cfg["binvar"],
        columns=cfg["sample"], aggfunc='sum', fill_value=0.,
    )

    # Sort MC by sum across bins
    columns = df_mc_counts.sum().sort_values().index.tolist()
    if cfg["process_at_bottom"] in columns:
        columns.remove(cfg["process_at_bottom"])
        columns = [cfg["process_at_bottom"]]+columns
    df_mc_counts = df_mc_counts.reindex(columns, axis=1)

    fig, (axt, axb) = plt.subplots(
        figsize = (4.8, 6), nrows=2, ncols=1, sharex='col', sharey=False,
        gridspec_kw={'height_ratios': [2.5, 1], 'wspace': 0.1, 'hspace': 0.1},
    )

    # Draw data
    binvars = df_data.index.get_level_values(cfg["binvar"])
    bincents = (binning[1:] + binning[:-1])/2.
    idx_dummy = pd.DataFrame({
        "binvar": binning[:-1], cfg["counts"]: [0]*len(binning[:-1]),
        cfg["variance"]: [0]*len(binning[:-1]),
    }).set_index("binvar")
    df_data = df_data.reindex_like(idx_dummy)

    if not cfg["blind"]:
        axt.errorbar(
            bincents, df_data[cfg["counts"]],
            yerr=np.sqrt(df_data[cfg["variance"]]), fmt='o', ms=4, lw=0.6,
            capsize=2.5, color='black', label="Data",
        )

    # Draw MC
    binvars = df_mc_counts.index.get_level_values(cfg["binvar"])
    axt.hist(
        binvars, binning, weights=df_mc_counts.sum(axis=1),
        histtype='step', color='black', label=r'',#label=r'SM Total',
    )
    axt.hist(
        [binvars]*df_mc_counts.shape[1], binning, weights=df_mc_counts.values,
        histtype='stepfilled', stacked=True, log=True,
        color=[cfg["process_colors"][p] for p in columns],
        label=[cfg["process_names"][p] for p in columns],
    )

    handles, labels = axt.get_legend_handles_labels()
    fractions = (df_mc_counts.sum(axis=0) / df_mc_counts.sum().sum()).values[::-1]
    fraction_labels = ["{:.3f}".format(f) for f in fractions]

    if not cfg["blind"]:
        fraction_labels = [
            "{:.3f}".format(df_data[cfg["counts"]].sum().sum()/df_mc_counts.sum().sum())
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

    axt.text(
        0, 1, r'$\mathbf{CMS}\ \mathit{Preliminary}$',
        ha='left', va='bottom', transform=axt.transAxes, fontsize=12,
    )
    axt.text(
        1, 1, r'$35.9\ \mathrm{fb}^{-1}(13\ \mathrm{TeV})$',
        ha='right', va='bottom', transform=axt.transAxes, fontsize=12,
    )

    ylims = axt.get_ylim()
    axt.set_xlim(binning[0], binning[-1])
    axt.set_ylim(max(ylims[0], 0.5), ylims[1])
    axt.set_ylabel("Number of events", fontsize=12)
    axb.set_xlabel(cfg["label"], fontsize=12)
    axb.set_ylabel("Data/Simulation", fontsize=12)

    print("Creating {}".format(cfg["outpath"]))
    fig.align_ylabels([axt, axb])
    fig.savefig(cfg["outpath"], format="pdf", bbox_inches="tight")
    plt.close(fig)

    return True
