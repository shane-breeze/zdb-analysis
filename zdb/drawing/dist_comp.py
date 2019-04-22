import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dist_comp(df, bins, filepath, cfg):
    # Define columns
    all_columns = list(df.index.names)
    columns_noname = [c for c in all_columns if c != "name"]
    columns_nobins = [c for c in all_columns if "bin" not in c]

    # Remove under and overflow bins (add overflow into final bin)
    def truncate(indf):
        indf.iloc[-2] += indf.iloc[-1]
        indf = indf.iloc[1:-1]
        indf = indf.reset_index(columns_nobins, drop=True)
        return indf
    df = df.groupby(columns_nobins).apply(truncate)
    bins = bins[0][1:-1]

    df_pivot_name = df.pivot_table(
        values = 'yield',
        index = columns_noname,
        columns = 'name',
        aggfunc = np.sum,
    )

    name_uncorr = [c for c in df_pivot_name.columns if "corrected" not in c].pop()
    name_corr = [c for c in df_pivot_name.columns if "corrected" in c].pop()
    df_uncorr = df_pivot_name[name_uncorr]
    df_corr = df_pivot_name[name_corr]
    df_uncorr.columns = ['yield']
    df_corr.columns = ['yield']

    df_pivot_uncorr = df_uncorr.unstack(level='key')
    df_pivot_corr = df_corr.unstack(level='key')
    df_pivot_ratio = df_pivot_corr / df_pivot_uncorr

    # Get the global bins
    xlow = df_pivot_uncorr.index.get_level_values("bin0_low").values
    xupp = df_pivot_uncorr.index.get_level_values("bin0_upp").values
    xcenters = (xupp+xlow)/2

    # Split axis into top and bottom with ratio 3:1
    # Share the x axis, not the y axis
    # Figure size is 4.8 by 6.4 inches
    fig, (axtop, axbot) = plt.subplots(
        nrows=2, ncols=1, sharex='col', sharey=False,
        gridspec_kw={'height_ratios': [3, 1]},
        figsize = (4.8, 6.4),
    )
    if cfg.log: axtop.set_yscale('log')

    # Draw hists
    all_keys = list(df.index.get_level_values("key").unique())
    axtop.hist(
        [xcenters]*len(all_keys),
        bins = bins,
        weights = [df_pivot_uncorr[k] for k in all_keys],
        histtype = 'step',
        color = [cfg.sample_colours.get(k, "blue") for k in all_keys],
        label = [cfg.sample_names.get(k, k) for k in all_keys],
        ls = '--',
    )
    handles, labels = axtop.get_legend_handles_labels()
    axtop.hist(
        [xcenters]*len(all_keys),
        bins = bins,
        weights = [df_pivot_corr[k] for k in all_keys],
        histtype = 'step',
        color = [cfg.sample_colours.get(k, "blue") for k in all_keys],
        label = [cfg.sample_names.get(k, k) for k in all_keys],
    )
    handles_corr, labels_corr = axtop.get_legend_handles_labels()
    handles_corr = handles_corr[len(handles):]
    labels_corr = labels_corr[len(labels):]
    labels_corr = [l+" corr." for l in labels_corr]
    labels = reduce(lambda x,y: x+y, list(zip(labels, labels_corr)))
    handles = reduce(lambda x,y: x+y, list(zip(handles, handles_corr)))

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
    axtop.legend(handles, labels)

    # Ratio in bottom panel
    axbot.hist(
        [xcenters]*len(all_keys),
        bins = bins,
        weights = [df_pivot_ratio[k] for k in all_keys],
        histtype = 'step',
        color = [cfg.sample_colours.get(k, 'blue') for k in all_keys],
    )
    axbot.axhline(1, ls='--', color='gray')

    name = cfg.name
    ymin = min(df_pivot_ratio[all_keys].min())
    ymax = max(df_pivot_ratio[all_keys].max())
    padding = (ymax - ymin)*0.05
    if not (np.isinf(ymin) or np.isnan(ymin) or np.isinf(ymax) or np.isnan(ymax)):
        axbot.set_ylim((ymin-padding, ymax+padding))
    axbot.set_xlabel(cfg.axis_label.get(name, name),
                     fontsize='large')
    axbot.set_ylabel(r'$1 + \kappa_{EW}$', fontsize='large')

    # Report
    print("Creating {}.pdf".format(filepath))

    # Actually save the figure
    plt.tight_layout()
    fig.savefig(filepath+".pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    return df
