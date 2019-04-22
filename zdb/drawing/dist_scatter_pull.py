import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.stats import norm
from scipy.interpolate import spline

def make_error_boxes(ax, xlow, xhigh, ylow, yhigh, facecolor='r',
                     edgecolor='r', alpha=1.0):
    errorboxes = [
        Rectangle((xl, yl),
                  xh-xl, yh-yl,
                  alpha = alpha)
        for xl, xh, yl, yh in zip(xlow, xhigh, ylow, yhigh)
    ]

    pc = PatchCollection(errorboxes,
                         facecolor = facecolor,
                         edgecolor = edgecolor,
                         alpha = alpha)
    ax.add_collection(pc)
    return errorboxes[0]

def dist_scatter_pull(df, bins, variations, filepath, cfg):
    # # change stuff
    # bin1_low = df.index.get_level_values("bin1_low").values
    # bin1_upp = df.index.get_level_values("bin1_upp").values

    # infidx = np.where(np.isinf(bin1_upp))[0]
    # bin1_upp[infidx] = 2*bin1_upp[infidx-1] - bin1_upp[infidx-2]
    # df = df.divide((bin1_upp+bin1_low)/2, axis=0)
    # df["mean"] = df["mean"] + 1
    # cfg.ylabel = r'$\mu(E_{T,\parallel}^{miss}-p_{T}(\mu\mu)) / \langle p_{T}(\mu\mu) \rangle - 1$'
    # cfg.ylabel = r'$\mu(E_{T,\perp}^{miss}) / \langle p_{T}(\mu\mu) \rangle + 1$'

    datasets = ["MET", "SingleMuon", "SingleElectron"]
    attr = df.columns[0]

    # Remove underflow bin
    bins = bins[1:-1]
    df = df[~np.isinf(df.index.get_level_values("bin1_low"))]
    df = df[~np.isinf(df.index.get_level_values("bin1_upp"))]

    # Split into data and MC
    df_data = df[df.index.get_level_values('process').isin(datasets)]\
            .reset_index("process", drop=True)
    df_mc = df[df.index.get_level_values('process').isin(["MCSum"])]\
            .reset_index("process", drop=True)

    # Get binning
    bins = list(bins)
    bins.append(2*bins[-1]-bins[-2])
    bins = np.array(bins)
    ylow = df_data.index.get_level_values("bin1_low").values
    yupp = df_data.index.get_level_values("bin1_upp").values
    ycents = (yupp + ylow)/2
    ywidths = (yupp - ylow)

    # total uncertainty
    df_mc["{}_unc_up_total".format(attr)] = df_mc["{}_unc".format(attr)]**2
    df_mc["{}_unc_down_total".format(attr)] = df_mc["{}_unc".format(attr)]**2
    for var in variations:
        if "up" in var.lower():
            df_mc["{}_unc_up_total".format(attr)] += df_mc["{}_unc_{}".format(attr, var)]**2
        elif "down" in var.lower():
            df_mc["{}_unc_down_total".format(attr)] += df_mc["{}_unc_{}".format(attr, var)]**2
    df_mc["{}_unc_up_total".format(attr)] = np.sqrt(df_mc["{}_unc_up_total".format(attr)])
    df_mc["{}_unc_down_total".format(attr)] = np.sqrt(df_mc["{}_unc_down_total".format(attr)])
    df_data["{}_unc_total".format(attr)] = df_data["{}_unc".format(attr)]

    # Ratios
    df_ratio = pd.concat(
        [df_data[[attr, "{}_unc_total".format(attr)]],
         df_mc[[attr, "{}_unc_up_total".format(attr), "{}_unc_down_total".format(attr)]]],
        axis = 1,
    )
    df_ratio.columns = ["data", "data_unc", "mc", "mc_unc_up", "mc_unc_down"]
    df_ratio["mc_unc"] = df_ratio.apply(lambda x: x["mc_unc_up"] if x["data"]>=x["mc"] else x["mc_unc_down"], axis=1)
    df_ratio["ratio"] = df_ratio["data"]/df_ratio["mc"]
    df_ratio["ratio_data_unc"] = df_ratio["data_unc"]/df_ratio["mc"]
    df_ratio["ratio_mc_unc"] = df_ratio["mc_unc"]/df_ratio["mc"]
    df_ratio = df_ratio[["ratio", "ratio_data_unc", "ratio_mc_unc"]]

    # Pulls
    df_pulls = pd.concat(
        [df_data[attr], df_mc[attr],
         df_data["{}_unc_total".format(attr)],
         df_mc["{}_unc_up_total".format(attr)],
         df_mc["{}_unc_down_total".format(attr)]],
        axis = 1,
    )
    df_pulls.columns = ["data", "mc", "data_unc", "mc_unc_up", "mc_unc_down"]
    df_pulls["diff"] = (df_pulls["data"] - df_pulls["mc"])
    df_pulls["mc_unc"] = df_pulls.apply(lambda x: x["mc_unc_up"] if x["data"]>=x["mc"] else x["mc_unc_down"], axis=1)
    df_pulls["pull"] = df_pulls["diff"] / np.sqrt(df_pulls["data_unc"]**2 + df_pulls["mc_unc"]**2)
    df_pulls = df_pulls[["pull"]]

    # Create figure and axes
    fig, axes = plt.subplots(
        nrows=3, ncols=2, sharex='col', sharey='row',
        gridspec_kw={'height_ratios': [3, 1, 1],
                     'width_ratios': [6, 1],
                     'wspace': 0.05,
                     'hspace': 0.05},
        figsize = (5.6, 6.4),
    )
    axtop, axtopnull = axes[0]
    axmid, axmidnull = axes[1]
    axbot, axbotrig = axes[2]
    axtopnull.axis('off')
    axmidnull.axis('off')

    # top axes
    axtop.text(0, 1, r'$\mathbf{CMS}\ \mathit{Preliminary}$',
               ha='left', va='bottom', transform=axtop.transAxes,
               fontsize='large')
    axtop.text(1, 1, r'$35.9\ \mathrm{fb}^{-1}(13\ \mathrm{TeV})$',
               ha='right', va='bottom', transform=axtop.transAxes,
               fontsize='large')

    # absolute MC boxes
    rect_eg = make_error_boxes(
        axtop,
        ycents - ywidths/2,
        ycents + ywidths/2,
        df_mc[attr] - df_mc["{}_unc_down_total".format(attr)],
        df_mc[attr] + df_mc["{}_unc_up_total".format(attr)],
        facecolor="#80b1d3",
        edgecolor="#5a9ac6",
    )

    # absolute data points
    axtop.errorbar(
        ycents, df_data[attr],
        xerr=ywidths/2, yerr=df_data["{}_unc_total".format(attr)],
        fmt='o', markersize=3, linewidth=1,
        capsize=1.8, color="black", label="Data",
    )

    mc_down = (df_mc[attr] - df_mc["{}_unc_down_total".format(attr)]).values
    mc_up = (df_mc[attr] + df_mc["{}_unc_up_total".format(attr)]).values
    data_down = (df_data[attr] - df_data["{}_unc_total".format(attr)]).values
    data_up = (df_data[attr] + df_data["{}_unc_total".format(attr)]).values
    mc_down = mc_down[~(np.isinf(mc_down) | np.isnan(mc_down))]
    mc_up = mc_up[~(np.isinf(mc_up) | np.isnan(mc_up))]
    data_down = data_down[~(np.isinf(data_down) | np.isnan(data_down))]
    data_up = data_up[~(np.isinf(data_up) | np.isnan(data_up))]

    ylims = axtop.get_ylim()
    axtop.set_ylim((
        min(list(mc_down)+list(data_down)+[ylims[0]]),
        max(list(mc_up)+list(data_up)+[ylims[1]]),
    ))

    # Set minimum y range
    ylims = axtop.get_ylim()
    dylims = ylims[1]-ylims[0]
    if dylims < 0.1:
        add_dylims = 0.5*(0.1 - dylims)
        axtop.set_ylim((ylims[0]-add_dylims, ylims[1]+add_dylims))
    # TODO
    #axtop.set_xlim((50., 350.))
    #axtop.set_ylim((0, 40))
    #axtop.axhline(1., ls='--', color='grey', lw=1)

    handles, labels = axtop.get_legend_handles_labels()
    handles += [rect_eg]
    labels += ["MC"]
    axtop.legend(handles, labels)
    axtop.set_ylabel(cfg.ylabel, fontsize='large')

    # TODO
    #axtop.set_ylabel(r'$\mu(E_{T,\parallel}^{miss}-p_{T}(\mu\mu)) / \langle p_{T}(\mu\mu) \rangle + 1$')
    #axtop.set_ylabel(r'$\mu(E_{T,\perp}^{miss}) / \langle p_{T}(\mu\mu) \rangle$')

    # Middle axes
    axmid.axhline(1, ls='--', color='grey', lw=1)
    axmid.set_ylabel("Data / MC", fontsize='large')
    # TODO
    #axmid.set_ylim((0.75, 1.25))

    xlow = df_ratio.index.get_level_values("bin1_low").values
    xupp = df_ratio.index.get_level_values("bin1_upp").values
    xbins = np.array(list(xlow) + [xupp[-1]])
    axmid.fill_between(
        xbins,
        list(1.-df_ratio["ratio_mc_unc"])+[1.],
        list(1.+df_ratio["ratio_mc_unc"])+[1.],
        step = 'post',
        color = '#aaaaaa',
        label = "MC unc.",
    )

    axmid.errorbar(
        ycents, df_ratio["ratio"],
        xerr=ywidths/2, yerr=df_ratio["ratio_data_unc"],
        fmt='o', markersize=3, linewidth=1,
        capsize=1.8, color="black",
    )

    # bottom axes
    axbot.plot(ycents, df_pulls["pull"], 'o', ms=3, mfc='black', mec='black')
    axbot.set_xlabel(cfg.xlabel, fontsize='large')
    axbot.set_ylabel("Pull", fontsize='large')

    ylim = max(map(abs, axbot.get_ylim()))
    ylim = ylim if ylim>1.1 else 1.1
    axbot.set_ylim(-ylim, ylim)
    axbot.axhline(-1, ls='--', color='grey', lw=1)
    axbot.axhline(1, ls='--', color='grey', lw=1)

    # bottom right axes
    pull_bins = [-np.inf] + list(np.linspace(-5., 5., 21)) + [np.inf]
    pull_hist, _ = np.histogram(df_pulls["pull"], pull_bins)
    pull_hist[1] += pull_hist[0]
    pull_hist[-2] += pull_hist[-1]
    pull_hist = pull_hist[1:-1]
    pull_bins = pull_bins[1:-1]

    axbotrig.hist(
        (np.array(pull_bins[1:])+np.array(pull_bins[:-1]))/2,
        bins = pull_bins,
        weights = pull_hist,
        histtype = 'step',
        orientation = 'horizontal',
        color = "k",
    )
    axbotrig.set_xlim(0., pull_hist.max()+1)
    axbotrig.set_ylim(-ylim, ylim)
    axbotrig.axhline(-1, ls='--', color='grey', lw=1)
    axbotrig.axhline(1, ls='--', color='grey', lw=1)

    # Add gaussian fit
    (mu, sigma) = norm.fit(df_pulls["pull"])
    pull_bins = np.array(pull_bins)
    xs = (pull_bins[1:] + pull_bins[:-1])/2.
    gaus = mlab.normpdf(xs, mu, sigma)
    xnew = np.linspace(xs.min(), xs.max(), xs.shape[0]*4)
    ynew = spline(xs, pull_hist.sum()*gaus/gaus.sum(), xnew)
    axbotrig.plot(ynew, xnew, 'r--', lw=2)
    axbotrig.set_xticklabels([])

    # Create the damn plots
    print("Creating {}.pdf".format(filepath))
    plt.tight_layout()
    fig.savefig(filepath+".pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    return df
