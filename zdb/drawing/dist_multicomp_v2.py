import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
logging.basicConfig()

from zinv.utils.Colours import colours_dict

def dist_multicomp_v2(df, bins, filepath, cfg):
    mll_range = [71, 111]
    all_keys = list(df.index.get_level_values("key").unique())
    all_idxs = list(df.index.names)

    # remove over/under-flow
    bins = np.array(bins)
    bins = bins[1:-1]

    df_pivot_key = df.pivot_table(
        values = "yield",
        index = [k for k in all_idxs if k not in ["key"]],
        columns = "key",
        aggfunc = np.sum,
    )

    df_pivot_key_var = df.pivot_table(
        values = "variance",
        index = [k for k in all_idxs if k not in ["key"]],
        columns = "key",
        aggfunc = np.sum,
    )

    df_pivot_ratio = df_pivot_key[all_keys]
    df_pivot_ratio = df_pivot_ratio.div(df_pivot_key[all_keys[0]], axis=0)
    df_pivot_ratio_var = df_pivot_key_var[all_keys]
    df_pivot_ratio_var = df_pivot_ratio_var.div(df_pivot_key[all_keys[0]]**2, axis=0)

    df_pivot_key_sum = df_pivot_key.groupby([k for k in all_idxs if "bin0" not in k and k!="key"]).sum()
    df_pivot_key_var_sum = df_pivot_key_var.groupby([k for k in all_idxs if "bin0" not in k and k!="key"]).sum()
    df_pivot_ratio_sum = df_pivot_key_sum[all_keys]
    df_pivot_ratio_sum = df_pivot_ratio_sum.div(df_pivot_key_sum[all_keys[0]], axis=0)
    df_pivot_ratio_var_sum = df_pivot_key_var_sum[all_keys]
    df_pivot_ratio_var_sum = df_pivot_ratio_var_sum.div(df_pivot_key_sum[all_keys[0]]**2, axis=0)

    fig, (axtop, axbot) = plt.subplots(
        nrows=2, ncols=1, sharex='col', sharey=False,
        gridspec_kw={'height_ratios': [3, 1],
                     'wspace': 0.1,
                     'hspace': 0.1},
        figsize = (4.8, 6),
    )

    axtop.set_xlim(bins[0], bins[-1])
    axtop.set_ylabel("Event yield", fontsize=12)
    axtop.set_yscale('log')

    xaxis_label = df.index.get_level_values("name")[0]
    axbot.set_xlabel(cfg.axis_label.get(xaxis_label, xaxis_label.replace("_", " ")),
                     fontsize=12)
    axbot.set_ylabel("Ratio", fontsize=12)

    # Add CMS text to top + energy + lumi
    axtop.text(0.01, 1, r'$\mathbf{CMS}\ \mathit{Preliminary}$',
               ha='left', va='bottom', transform=axtop.transAxes,
               fontsize=12)
    axtop.text(0.99, 1, r'$35.9\ \mathrm{fb}^{-1}(13\ \mathrm{TeV})$',
               ha='right', va='bottom', transform=axtop.transAxes,
               fontsize=12)

    for key in all_keys:
        tdf = df_pivot_key[key]
        bin_low = np.array(tdf.index.get_level_values("bin0_low"))
        bin_upp = np.array(tdf.index.get_level_values("bin0_upp"))
        bin_cent = (bin_low + bin_upp)/2

        axtop.hist(
            bin_cent,
            bins = bins,
            weights = tdf.values,
            histtype = 'step',
            color = cfg.sample_colours.get(key, "black"),
            label = cfg.sample_names.get(key, key.replace("_", " ")),
        )

        tdf_key_var = df_pivot_key_var[key]
        central = np.zeros(bins.shape[0]-1)
        variance = np.zeros(bins.shape[0]-1)
        central[np.isin(bins[:-1], bin_low)] = tdf[tdf.index.get_level_values("bin0_low").isin(bins[:-1])].values
        variance[np.isin(bins[:-1], bin_low)] = tdf_key_var[tdf_key_var.index.get_level_values("bin0_low").isin(bins[:-1])].values
        axtop.fill_between(
            bins,
            list(central - np.sqrt(variance)) + [central[-1]],
            list(central + np.sqrt(variance)) + [central[-1]],
            step = 'post',
            color = cfg.sample_colours.get(key, "black"),
            alpha = 0.5,
        )

        tdf_ratio = df_pivot_ratio[key]
        axbot.hist(
            bin_cent,
            bins = bins,
            weights = tdf_ratio.values,
            histtype = 'step',
            color = cfg.sample_colours.get(key, "black"),
            label = "",
        )

        tdf_ratio_var = df_pivot_ratio_var[key]
        central = np.zeros(bins.shape[0]-1)
        variance = np.zeros(bins.shape[0]-1)
        central[np.isin(bins[:-1], bin_low)] = tdf_ratio[tdf_ratio.index.get_level_values("bin0_low").isin(bins[:-1])].values
        variance[np.isin(bins[:-1], bin_low)] = tdf_ratio_var[tdf_ratio_var.index.get_level_values("bin0_low").isin(bins[:-1])].values
        axbot.fill_between(
            bins,
            list(central - np.sqrt(variance)) + [1.],
            list(central + np.sqrt(variance)) + [1.],
            step = 'post',
            color = cfg.sample_colours.get(key, "black"),
            alpha = 0.5,
        )

    axtop.axvspan(bins[0],  mll_range[0],  alpha=0.3, color='gray')
    axtop.axvspan(mll_range[-1], bins[-1], alpha=0.3, color='gray')
    axbot.axvspan(bins[0],  mll_range[0],  alpha=0.3, color='gray')
    axbot.axvspan(mll_range[-1], bins[-1], alpha=0.3, color='gray')

    axbot.set_ylim((0.9, 1.1))

    # Legend
    handles, labels = axtop.get_legend_handles_labels()
    labels = [
        labels[idx]+r' ${:.3f}$'.format( # \pm {:.3f}$'.format(
            df_pivot_ratio_sum[all_keys[idx]].values[0],
            np.sqrt(df_pivot_ratio_var_sum[all_keys[idx]].values[0]),
        ) for idx in range(len(all_keys))
    ]
    legend = axtop.legend(handles, labels, framealpha=0.8)
    plt.setp(legend.get_title(), fontsize=12)

    # Report
    print("Creating {}.pdf".format(filepath))

    # Actually save the figure
    fig.savefig(filepath+".pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    return df
