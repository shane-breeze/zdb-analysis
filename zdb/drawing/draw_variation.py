import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

def process_dataframe(df, cfg):
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
    # Pivot MC tables on samples
    df_counts = pd.pivot_table(
        df, values=cfg["counts"], index=cfg["binvar"],
        columns=[cfg["sample"], cfg["varname"]], aggfunc='sum', fill_value=0.,
    )
    df_variance = pd.pivot_table(
        df, values=cfg["variance"], index=cfg["binvar"],
        columns=[cfg["sample"], cfg["varname"]], aggfunc='sum', fill_value=0.,
    )

    new_cols = []
    for c in df_counts.columns.levels[1]:
        new_cols.append(
            "Up" if c.endswith("Up") else
            "Down" if c.endswith("Down") else
            "Nominal"
        )
    df_counts.columns = df_counts.columns.set_levels([df_counts.columns.levels[0], new_cols])
    df_variance.columns = df_variance.columns.set_levels([df_counts.columns.levels[0], new_cols])

    return df_counts, df_variance

def draw_ratios(ax, df, cfg):
    binning = np.arange(*cfg["binning"])
    binvars = df.index.get_level_values(cfg["binvar"]).tolist()
    labels = [c[0] for c in df.columns.to_flat_index()]
    ax.hist(
        [binvars]*df.shape[1], bins=binning, weights=df.values, histtype='step',
        color=[cfg["process_colors"].get(c, c) for c in labels],
        label=[cfg["process_names"].get(labels[idx], labels[idx]) if idx%2==0 else "" for idx in range(len(labels))],
    )

def draw_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    return ax.legend(handles, labels)

def draw_cms_header(ax):
    ax.text(
        0, 1, r'$\mathbf{CMS}\ \mathit{Preliminary}$',
        ha='left', va='bottom', transform=ax.transAxes, fontsize=12,
    )
    ax.text(
        1, 1, r'$35.9\ \mathrm{fb}^{-1}(13\ \mathrm{TeV})$',
        ha='right', va='bottom', transform=ax.transAxes, fontsize=12,
    )

def draw_variation(df, cfg):
    plt.style.use('cms')

    df = process_dataframe(df, cfg)
    df_counts, df_variance = create_mc_counts_variance(df, cfg)

    dfs = []
    for c in df_counts.columns.levels[0]:
        dfs.append(
            df_counts.loc[:, (c, ("Up", "Down"))]
            .divide(df_counts.loc[:, (c, "Nominal")], axis='index')
            - 1.
        )
    df_ratio = (
        pd.concat(dfs, axis='columns')
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.)
    )

    fig, ax = plt.subplots(figsize = (5.4, 4.8))

    draw_ratios(ax, df_ratio, cfg)
    draw_cms_header(ax)
    draw_legend(ax)

    ylim = min(0.5, max(map(abs, ax.get_ylim())))
    ax.set_ylim(-ylim, +ylim)
    ax.axhline(1., ls='--', lw=0.8, color='black')

    print("Creating {}".format(cfg["outpath"]))
    fig.savefig(cfg["outpath"], format="pdf", bbox_inches="tight")
    plt.close(fig)

    return True
