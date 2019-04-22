import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.width', None)

def dist_test(*args, **kwargs):
    dfs = []
    for arg in args:
        df, _, cfg = arg
        name = df.index.get_level_values("name").unique()[0]

        df = df.reset_index(["weight", "name", "variable0"], drop=True)\
                .reset_index(["bin0_low", "bin0_upp"])
        df = df.loc[[("MET", "Monojet", "ZJetsToNuNu"),
                     ("MET", "SingleMuon", "WJetsToMuNu"),
                     ("MET", "DoubleMuon", "DYJetsToMuMu"),
                     ("MET", "None", "ZJetsToNuNu"),
                     ("MET", "None", "WJetsToMuNu"),
                     ("MET", "None", "DYJetsToMuMu"),
                     ("MET", "Monojet_remove_muon_selection_fmt_0", "ZJetsToNuNu"),
                     ("MET", "Monojet_remove_muon_selection_fmt_0", "WJetsToMuNu"),
                     ("MET", "Monojet_remove_muon_selection_fmt_0", "DYJetsToMuMu")]].dropna()
        df = df.set_index(["bin0_low", "bin0_upp"], append=True)
        dfs.append(df)

    df = pd.concat(dfs)

    def truncate(indf):
        indf.iloc[-2] += indf.iloc[-1]
        indf = indf.iloc[1:-1]
        indf = indf.reset_index(["dataset", "region", "process"], drop=True)
        return indf

    # truncate and normalise
    df = df.groupby(["dataset", "region", "process"]).apply(truncate)\
            .reset_index(["dataset", "region"], drop=True)
    df = df.div(df.groupby("process").sum(axis=0), axis=0)

    # binning
    bin_low = df.index.get_level_values("bin0_low").unique()
    bin_upp = df.index.get_level_values("bin0_upp").unique()
    bin_cents = (bin_low+bin_upp)/2
    bins = np.array(list(bin_low)+[bin_upp[-1]])

    fig, (axtop, axbot) = plt.subplots(
        nrows=2, ncols=1, sharex='col', sharey=False,
        gridspec_kw={'height_ratios': [3, 1]},
        figsize = (4.8, 6.4),
    )

    df_piv = df.pivot_table(values = 'yield',
                            index = ["bin0_low", "bin0_upp"],
                            columns = 'process',
                            aggfunc = np.sum)

    axtop.hist(
        [bin_cents]*df_piv.shape[1],
        bins = bins,
        weights = df_piv.values,
        histtype = 'step',
        label = [cfg.sample_names.get(c, c) for c in df_piv.columns],
        #color = [cfg.sample_colours.get(c, 'black') for c in df_piv.columns],
    )
    #axtop.set_yscale('log')
    axtop.set_xlim(bins[0], bins[-1])
    axtop.legend(*axtop.get_legend_handles_labels())
    if "eta" in name or "phi" in name:
        axtop.set_ylabel("Event density / {:.1f}".format(stats.mode(bin_upp-bin_low).mode[0]),
                         fontsize = 'large')
    else:
        axtop.set_ylabel("Event density / {} GeV".format(int(stats.mode(bin_upp-bin_low).mode[0])),
                         fontsize = 'large')

    # Add CMS text to top + energy + lumi
    axtop.text(0, 1, r'$\mathbf{CMS}\ \mathit{Preliminary}$', ha='left',
               va='bottom', transform=axtop.transAxes, fontsize='large')
    axtop.text(1, 1, r'$35.9\ \mathrm{fb}^{-1}(13\ \mathrm{TeV})$', ha='right',
               va='bottom', transform=axtop.transAxes, fontsize='large')

    # Ratio
    df_ratio = df_piv.div(df_piv[df_piv.columns[0]], axis=0)
    #df_ratio = df_ratio[[c for c in df_piv.columns[1:]]]
    df_ratio[np.isnan(df_ratio)] = 1.
    df_ratio[np.isinf(df_ratio)] = 1.
    axbot.hist(
        [bin_cents]*df_ratio.shape[1],
        bins = bins,
        weights = df_ratio.values,
        histtype = 'step',
    )

    dy = min(1.1*np.max([1-df_ratio.min(), df_ratio.max()-1]), 0.5)
    axbot.set_ylim(1-dy, 1+dy)
    axbot.axhline(1, lw=1, ls=':', color='black')

    axbot.set_xlabel(cfg.axis_label.get(name, name), fontsize='large')
    axbot.set_ylabel("Ratio", fontsize='large')

    filepath = kwargs["filepath"]
    print("Creating {}.pdf".format(filepath))
    plt.tight_layout()
    fig.savefig(filepath+".pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
