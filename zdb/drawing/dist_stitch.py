import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def dist_stitch(df, bins, filepath, cfg):
    all_columns = list(df.index.names)
    columns_noproc = [c for c in all_columns if c != "process"]
    columns_nobins = [c for c in all_columns if "bin" not in c]
    columns_nobins_noproc = [c for c in columns_nobins if c != "process"]

    # Remove under and overflow bins (add overflow into final bin)
    bins = np.array(bins[0][1:-1])

    df_pivot_proc = df.pivot_table(
        values='yield',
        index=columns_noproc,
        columns='process',
        aggfunc=np.sum,
    )

    # Ratio
    y = np.log10(df_pivot_proc.sum(axis=1))
    df_ratio = 0.5*(np.roll(y, 1) + np.roll(y, -1))/y
    df_ratio.iloc[0] = 1.
    df_ratio.iloc[-1] = 1.

    # Ratio uncertainty
    df_pivot_proc_var = df.pivot_table(
        values = 'variance',
        index = columns_noproc,
        columns = 'process',
        aggfunc = np.sum,
    )
    df_ratio_err = np.sqrt(df_pivot_proc_var.sum(axis=1))/df_pivot_proc.sum(axis=1)

    # Split axis into top and bottom with ratio 3:1
    # Share the x axis, not the y axis
    # Figure size is 4.8 by 6.4 inches
    fig, (axtop, axbot) = plt.subplots(
        nrows=2, ncols=1, sharex='col', sharey=False,
        gridspec_kw={'height_ratios': [3, 1]},
        figsize = (4.8, 6.4),
    )
    if cfg.log:
        locmin = ticker.LogLocator(
            base = 10.0,
            subs = (0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10),
        )
        axtop.yaxis.set_minor_locator(locmin)
        axtop.yaxis.set_minor_formatter(ticker.NullFormatter())
        axtop.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        axtop.set_yscale('log')

    # Get the global bins
    xlow = df_pivot_proc.index.get_level_values("bin0_low").values
    xupp = df_pivot_proc.index.get_level_values("bin0_upp").values
    xcenters = (xupp + xlow)/2

    # Stacked plot
    sorted_processes = list(df_pivot_proc.sum().sort_values(ascending=False).index)
    df_pivot_proc = df_pivot_proc.fillna(0)
    axtop.hist(
        [xcenters]*len(sorted_processes),
        bins = bins,
        weights = [df_pivot_proc[proc].values for proc in sorted_processes],
        histtype='step',
        stacked = True,
        color = [cfg.sample_colours.get(proc, "blue")
                 for proc in sorted_processes],
        label = sorted_processes,
    )
    axtop.set_xlim(bins[0], bins[-1])

    # Add CMS text to top + energy + lumi
    axtop.text(0, 1, r'$\mathbf{CMS}\ \mathit{Preliminary}$',
               horizontalalignment='left',
               verticalalignment='bottom',
               transform=axtop.transAxes,
               fontsize='large')
    axtop.text(1, 1, r'$35.9\ \mathrm{fb}^{-1}(13\ \mathrm{TeV})$',
               horizontalalignment='right',
               verticalalignment='bottom',
               transform=axtop.transAxes,
               fontsize='large')

    # Legend - reverse the labels
    handles, labels = axtop.get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    labels = [cfg.sample_names.get(l, l) for l in labels]
    axtop.legend(handles, labels)

    # Ratio plot
    if df_ratio.shape[0] > 2:
        df_ratio.iloc[-2] = 1.
    axbot.hist(
        xcenters,
        bins = bins,
        weights = df_ratio,
        color = "black",
        histtype = 'step',
    )

    axbot.fill_between(
        xlow,
        1. - df_ratio_err.values,
        1. + df_ratio_err.values,
        step = 'post',
        color = "#aaaaaa",
        label = "MC stat. unc.",
    )
    ylim_ratio = max(1.-df_ratio.min(), df_ratio.max()-1.)*1.1
    ylim_ratio_err = df_ratio_err.max()*1.1
    ylim = max(ylim_ratio, ylim_ratio_err)
    if not (np.isinf(ylim) or np.isnan(ylim)):
        axbot.set_ylim((1.-ylim, 1.+ylim))
    axbot.axhline(1., ls=':', color='black')

    # Add labels
    name = cfg.name
    axbot.set_xlabel(cfg.axis_label.get(name, name),
                     fontsize='large')
    axbot.set_ylabel("Closure", fontsize='large')

    # Legend
    handles, labels = axbot.get_legend_handles_labels()
    axbot.legend(handles, labels)

    # Report
    print("Creating {}.pdf".format(filepath))

    # Actually save the figure
    plt.tight_layout()
    fig.savefig(filepath+".pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    return df
