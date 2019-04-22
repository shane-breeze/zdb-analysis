import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dist_2d(df, bins, path, cfg):
    all_columns = list(df.index.names)
    columns_bins = [c for c in all_columns if "bin" in c]

    # Drop overflow and underflow
    df = df.reset_index(columns_bins)
    df = df.loc[(~np.isinf(df)).all(axis=1)]\
            .set_index(columns_bins, append=True)
    bins = [b[1:-1] for b in bins]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(5.76, 4.8))

    xlow = df.index.get_level_values("bin0_low")
    xupp = df.index.get_level_values("bin0_upp")
    ylow = df.index.get_level_values("bin1_low")
    yupp = df.index.get_level_values("bin1_upp")

    x = (xlow + xupp)/2
    y = (ylow + yupp)/2

    h = ax.hist2d(
        x, y,
        bins = bins,
        weights = df["yield"],
        cmap = "Blues",
    )
    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label('Events', fontsize='large')

    ax.set_xlabel(cfg.xlabel, fontsize='large')
    ax.set_ylabel(cfg.ylabel, fontsize='large')

    # Add CMS text to top + energy + lumi
    ax.text(0, 1, r'$\mathbf{CMS}\ \mathit{Preliminary}$', ha='left',
            va='bottom', transform=ax.transAxes, fontsize='large')
    ax.text(1, 1, r'$35.9\ \mathrm{fb}^{-1}(13\ \mathrm{TeV})$', ha='right',
            va='bottom', transform=ax.transAxes, fontsize='large')

    print("Creating {}.pdf".format(path))
    #plt.tight_layout()
    fig.savefig(path+".pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    return df
