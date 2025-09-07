
#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on: 2020-12-02 10:15:01


import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

DATE = datetime.now().strftime("%Y-%m-%d")


import matplotlib.pyplot as plt
import seaborn as sns
import proplot as plot
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from adjustText import adjust_text

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

# %%
# PATHS
ROOT_DIR = Path(
    "/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/manuscript/phecode_assoc_w_ptb/"
)

DATA_DIR = ROOT_DIR.joinpath("icd9_assoc_only")
white_assoc_file = DATA_DIR.joinpath("results_white_assoc_w_ptb_icd9_only.tsv")
black_assoc_file = DATA_DIR.joinpath("results_black_assoc_w_ptb_icd9_only.tsv")


# phe_descrip_file = ROOT_DIR.joinpath("phecode_icd9_rolled.csv")
phe_descrip_file = ROOT_DIR.joinpath("phecode_definitions1.2.csv")


OUTPUT_DIR = DATA_DIR.joinpath('figures')


# %%
# -----------
# FUNCTIONS
# -----------


def load_phe_results(assoc_file, phe_df):
    keepcols = [
        "predictor",
        "beta",
        "SE",
        "OR",
        "p",
        "type",
        "n_total",
        "n_cases",
        "n_controls",
    ]
    dtype_vals = {
        "predictor": str,
        "beta": np.float64,
        "SE": np.float64,
        "OR": np.float64,
        "p": np.float64,
        "type": str,
        "n_total": np.float64,
        "n_cases": np.float64,
        "n_controls": np.float64,
    }

    results_df = pd.read_csv(assoc_file, sep="\t", usecols=keepcols, dtype=dtype_vals)
    no_na_df = results_df.loc[
        ~results_df["p"].isna(),
    ].copy()
    print(f"loaded dataframe from: {assoc_file}")
    print(
        f"Removed {results_df.shape[0]-no_na_df.shape[0]:,} na rows out of {results_df.shape[0]:,}"
    )

    phewas_df = pd.merge(
        no_na_df, phe_df, left_on="predictor", right_on="phecode", how="inner"
    )

    phewas_df["-log10_p"] = -1 * np.log10(phewas_df["p"])
    return phewas_df


def split_preg_chap(phe_df):
    preg_df = phe_df.loc[phe_df["category"] == "pregnancy complications"]
    non_preg_df = phe_df.loc[phe_df["category"] != "pregnancy complications"]
    return preg_df, non_preg_df


def get_xtick_locations(raw_phe_df):
    phe_df = raw_phe_df.copy()

    category_edges_pos = np.concatenate(
        (
            phe_df.groupby("category")["x_ind"].min().values,
            phe_df.groupby("category")["x_ind"].max().values,
        )
    )

    category_label_pos = (
        phe_df.groupby("category")["x_ind"].min().values
        + phe_df.groupby("category")["x_ind"].max().values
    ) / 2

    category_labels = phe_df.groupby("category").size().reset_index()["category"].values

    return category_edges_pos, category_label_pos, category_labels


def get_phe_color(category):

    cm_dict = dict(
        zip(
            non_preg_phe_df["category"].unique(),
            [
                "#d078d6",
                "#53ba5f",
                "#b440a3",
                "#8fb53c",
                "#6d63ce",
                "#db9130",
                "#599cd7",
                "#cd562a",
                "#50b399",
                "#ce434b",
                "#5e7e3a",
                "#df588c",
                "#bfa84b",
                "#917cbe",
                "#94652e",
                "#a4506f",
                "#de8b73",
            ],
        )
    )

    return cm_dict[category]

def set_xticks(ax, preg_edge_pos, preg_lab_pos, black_preg_df, preg_labels, xrot=0):
    _ = ax.tick_params(axis="x", which="minor", length=0)
    _ = ax.set_xticks(ticks=preg_edge_pos)
    _ = ax.set_xticks(ticks=preg_lab_pos, minor=True)
    _ = ax.set_xlim(black_preg_df["x_ind"].min(), black_preg_df["x_ind"].max())
    _ = ax.set_xticklabels(labels=preg_labels, minor=True, rotation=xrot)
    _ = ax.set_xticklabels("", minor=False)

# %%
phe_df = pd.read_csv(phe_descrip_file, sep=",", dtype={"phecode": str})
phe_df.sort_values(["category", "phenotype", "phecode"], inplace=True)
phe_df.reset_index(drop=True, inplace=True)
phe_df.loc[phe_df["category"].isna(), "category"] = "other"


shorter_name = dict(zip(['circulatory system', 'congenital anomalies', 'dermatologic',
       'digestive', 'endocrine/metabolic', 'genitourinary',
       'hematopoietic', 'infectious diseases', 'injuries & poisonings',
       'mental disorders', 'musculoskeletal', 'neoplasms', 'neurological',
       'pregnancy complications', 'respiratory', 'sense organs',
       'symptoms', 'other'],
        ['circulatory', 'congenital', 'dermatologic',
      'digestive', 'endocrine', 'genitourinary',
      'hematopoietic', 'infectious', 'injuries',
      'mental health', 'musculoskeletal', 'neoplasms', 'neurological',
      'pregnancy', 'respiratory', 'sensory',
      'symptoms', 'other']))
phe_df['category'] = phe_df['category'].map(shorter_name)

# split phecode description
non_preg_phe_df = phe_df.loc[phe_df["category"] != "pregnancy"].copy()
preg_phe_df = phe_df.loc[phe_df["category"] == "pregnancy"].copy()

chap_order={'circulatory': 1,
    'congenital':2,
    'dermatologic':3,
    'digestive':4,
    'endocrine':5,
    'genitourinary':6,
    'hematopoietic': 7,
    'infectious':8,
    'injuries': 9,
    'mental health': 10,
    'musculoskeletal':11,
    'neoplasms': 12,
    'neurological':13,
    'pregnancy':14,
    'respiratory':15,
    'sensory':16,
    'symptoms':17,
    'other':18}



preg_phe_df["x_ind"] = np.arange(0, preg_phe_df.shape[0])
non_preg_phe_df["x_ind"] = np.arange(0, non_preg_phe_df.shape[0])

# load phe results and merge with phecode descriptions
black_preg_df = load_phe_results(black_assoc_file, preg_phe_df)
black_no_preg_df = load_phe_results(black_assoc_file, non_preg_phe_df)


white_preg_df = load_phe_results(white_assoc_file, preg_phe_df)
white_no_preg_df = load_phe_results(white_assoc_file, non_preg_phe_df)

non_preg_edge_pos, non_preg_lab_pos, non_preg_labels = get_xtick_locations(
    non_preg_phe_df
)

preg_edge_pos, preg_lab_pos, preg_labels = get_xtick_locations(preg_phe_df)
# %%


shared_sig_phecodes = set(
    black_no_preg_df.loc[
        black_no_preg_df["p"] < 0.05 / black_no_preg_df.shape[0], "predictor"
    ].values
).intersection(
    set(
        white_no_preg_df.loc[
            white_no_preg_df["p"] < 0.05 / white_no_preg_df.shape[0], "predictor"
        ].values
    )
)

shared_preg_sig_phecodes = set(
    black_preg_df.loc[
        black_preg_df["p"] < 0.05 / black_preg_df.shape[0], "predictor"
    ].values
).intersection(
    set(
        white_preg_df.loc[
            white_preg_df["p"] < 0.05 / white_preg_df.shape[0], "predictor"
        ].values
    )
)


# %%
# -----------
# plot
# -----------

bl_bonferroni_threshold = 0.05/(black_no_preg_df['phecode'].nunique() + black_preg_df['phecode'].nunique())
wh_bonferroni_threshold = 0.05/(white_no_preg_df['phecode'].nunique() + white_preg_df['phecode'].nunique())



# %%

fig3 = plt.figure(constrained_layout=False, figsize=(7,4))
gs = fig3.add_gridspec(2, 6)
bl_phe = fig3.add_subplot(gs[0, :5])
wh_phe = fig3.add_subplot(gs[1, :5])
bl_pre = fig3.add_subplot(gs[0, -1])
wh_pre = fig3.add_subplot(gs[1, -1])

plt.subplots_adjust(top=0.95, bottom=0.18, left=0.07, right=0.99, hspace=0.2, wspace=0.4)

for ax, df, bonf_pval in zip( [bl_phe, wh_phe], [black_no_preg_df, white_no_preg_df], [bl_bonferroni_threshold, wh_bonferroni_threshold]):
    for phecat in df["category"].unique():
        base_df = df.loc[(df["category"] == phecat),:].copy() # draw all base points
        plt_df = df.loc[(df["category"] == phecat) & (df['p']<bonf_pval),:].copy()
        not_sig_plt_df = df.loc[(df["category"] == phecat) & (df['p']>=bonf_pval),:].copy()

        for this_df, color, ssize in zip([base_df, plt_df], ['gray', get_phe_color(phecat)], (2 ,16)):
            ax.scatter(
                this_df["x_ind"],
                this_df["-log10_p"],
                s=ssize,
                marker="o",
                color=color,
            )


        ax.axhline(-1*np.log10(bonf_pval), color='indianred', linestyle=":", linewidth=0.5)
        ax.grid(b=False)
        ax.set_xlim(0,1789)


# plot pregnancy
for ax, df, preg_bonf_pval in zip([bl_pre, wh_pre], [black_preg_df, white_preg_df],  [bl_bonferroni_threshold, wh_bonferroni_threshold]):


    plt_df = df.loc[df['p']<bonf_pval,:].copy()
    base_df = df.loc[df['p']>=bonf_pval,:].copy()

    for this_df, color, ssize in zip([base_df, plt_df], ['gray', 'royalblue'], (2 ,16)):

        ax.scatter(
            this_df["x_ind"],
            this_df["-log10_p"],
            s=ssize,
            marker="o",
            color=color,
        )
        ax.axhline(-1*np.log10(preg_bonf_pval), color='indianred', linestyle=":", linewidth=1)
        ax.grid(b=False)
        ax.set_xlim(0,57)


# -----------
# xticks
# -----------
# nonpregnancy
# remove other
_ = wh_phe.set_xticks(non_preg_lab_pos[non_preg_labels!='other'])
_ = wh_phe.set_xticklabels(non_preg_labels[non_preg_labels!='other'], rotation=310, ha='left' )
_ = bl_phe.set_xticks([])
_ = bl_phe.set_xticklabels('')
_ = bl_phe.set_xticks([], minor=True)
_ = wh_phe.set_xticks([], minor=True)


# pregnancy
_ = bl_pre.set_xticklabels('')
_ = bl_pre.set_xticks([])
_ = wh_pre.set_xticks(preg_lab_pos)
_ = wh_pre.set_xticklabels(['pregnancy'])


# add chapter shading
cat_minor = np.sort(non_preg_edge_pos)[0::2]
for this_ax in [bl_phe, wh_phe]:
    patches_1 = []
    patches_2 = []
    for i in np.arange(0, len(cat_minor)):
        if i ==16:
            continue
        else:
            rect = mpatches.Rectangle(xy=(cat_minor[i], 0), width=cat_minor[i+1] - cat_minor[i], height=this_ax.get_ylim()[-1])
            if (i%2) == 0:
                patches_1.append(rect)
            else:
                patches_2.append(rect)

    collection1 = PatchCollection(patches_1, color='lightgray', alpha=0.4, zorder=-1)
    collection2 = PatchCollection(patches_2, color='white', alpha=1, zorder=-1)
    this_ax.add_collection(collection1)
    this_ax.add_collection(collection2)

# add yaxis labels
wh_phe.set_ylabel("-log10 P-value")
bl_phe.set_ylabel("-log10 P-value")
wh_phe.set_title("White")
bl_phe.set_title("Black")


# LABEL

black_no_preg_df.query('p<@bl_bonferroni_threshold').sort_values(['category', '-log10_p'])

# label
for ax_ind,ax , df, bonf_pval in zip(np.arange(0,5),
                                      [bl_phe, bl_pre, wh_phe, wh_pre],
                                      [black_no_preg_df, black_preg_df, white_no_preg_df, white_preg_df],
                                      [bl_bonferroni_threshold, bl_bonferroni_threshold, wh_bonferroni_threshold, wh_bonferroni_threshold]):


    if ((ax_ind == 1) | (ax_ind== 3)):
        to_label_df = df.loc[df['p']<bonf_pval,:].sort_values('p', ascending=True).head(8)
    else:
        to_label_df = df.loc[df['p']<bonf_pval,:].sort_values('p', ascending=True)
        # to_label_df = temp_to_label_df.sort_values('p', ascending=True).groupby('category').head(5)


    annotations = [ax.text(row['x_ind'], row['-log10_p'],row['phenotype'], fontsize=4) for ind, row in to_label_df.iterrows()]
    adjust_text(annotations,  ax=ax,
                force_text=(0.25,0.25), only_move={'points':'y', 'text':'y', 'objects':'y'})

for xtick in wh_phe.get_xticklabels():
    xtick.set_color(get_phe_color(xtick.get_text()))



plt.savefig(os.path.join(OUTPUT_DIR, f"{DATE}_assoc_w_ptb_label_v4.pdf"), transparent=True)






# %%
#
#
# layout = [[1, 1, 1, 1, 1, 2], [3, 3, 3, 3, 3, 4]]
# fig, axs = plot.subplots(array=layout, height=4, width=8.5 - 2, sharex=3, sharey=0, hspace=0.2, wspace=0.2, left='3.5em', tight=False)
#
#
# # no pregnancy
# for ax_ind, df, bonf_pval in zip([0, 2], [black_no_preg_df, white_no_preg_df], [bl_bonferroni_threshold, wh_bonferroni_threshold]):
#     for phecat in df["category"].unique():
#         base_df = df.loc[(df["category"] == phecat),:].copy() # draw all base points
#         plt_df = df.loc[(df["category"] == phecat) & (df['p']<bonf_pval),:].copy()
#         not_sig_plt_df = df.loc[(df["category"] == phecat) & (df['p']>=bonf_pval),:].copy()
#
#
#         axs[ax_ind].scatter(
#             base_df["x_ind"],
#             base_df["-log10_p"],
#             size=8,
#             marker=".",
#             color='gray',
#         )
#
#
#
#         axs[ax_ind].scatter(
#             plt_df["x_ind"],
#             plt_df["-log10_p"],
#             size=8,
#             marker=".",
#             color=get_phe_color(phecat),
#         )
#
#         axs[ax_ind].scatter(
#             not_sig_plt_df["x_ind"],
#             not_sig_plt_df["-log10_p"],
#             size=8,
#             marker=".",
#             color='gray',
#         )
#
#         axs[ax_ind].axhline(-1*np.log10(bonf_pval), color='indianred', linestyle=":", linewidth=0.5)
#
#
#         sig_plt_df = plt_df[plt_df["predictor"].isin(shared_sig_phecodes)]
#         axs[ax_ind].scatter(
#             sig_plt_df["x_ind"],
#             sig_plt_df["-log10_p"],
#             size=10,
#             marker="D",
#             markeredgecolor="k",
#             color=get_phe_color(phecat),
#         )
#         # axs[ax_ind].set_ylim(0,40)
#
# axs[0].set_title("Black")
# axs[2].set_title("White")
#
# # pregnacy
# for ax_ind, df, preg_bonf_pval in zip([1, 3], [black_preg_df, white_preg_df],  [bl_bonferroni_threshold, wh_bonferroni_threshold]):
#
#     plt_df = df.loc[df['p']<bonf_pval,:].copy()
#     not_sig_plt_df = df.loc[df['p']>=bonf_pval,:].copy()
#
#
#     sig_plt_df = df[df["predictor"].isin(shared_preg_sig_phecodes)]
#     axs[ax_ind].scatter(
#         plt_df["x_ind"], plt_df["-log10_p"], size=8, marker=".", color="royalblue"
#     )
#
#     axs[ax_ind].scatter(
#         not_sig_plt_df["x_ind"], not_sig_plt_df["-log10_p"], size=8, marker=".", color="gray"
#     )
#
#
#     axs[ax_ind].scatter(
#         sig_plt_df["x_ind"],
#         sig_plt_df["-log10_p"],
#         size=10,
#         marker="D",
#         markeredgecolor="k",
#         color="royalblue",
#     )
#
#     axs[ax_ind].axhline(-1*np.log10(preg_bonf_pval), color='indianred', linestyle=":", linewidth=0.5)
#     axs[ax_ind].axhline(-1*np.log10(0.05), color='black', linestyle="-", linewidth=0.5)
#
#
#
#
#
# # set yticks
# for ax_ind, df in zip(np.arange(0,5), [black_no_preg_df, black_preg_df, white_no_preg_df, white_preg_df]):
#     ymax=np.round(np.max(df['-log10_p'])+0.10*np.max(df['-log10_p']))
#     axs[ax_ind].format(ylim=(0, ymax) , ylocator=[np.round(x) for x in np.linspace(0,ymax , 4)])
#
#     # label
#     if ax_ind % 2 ==0:
#         to_label_df = df.sort_values('p').head(35)
#     else:
#         to_label_df = df.sort_values('p').head(15)
#
#
# _ = [axs[x].set_ylabel("") for x in [1, 3]]
# [ax.format(xtickminor=False, xgrid=False, ygrid=True, ytickminor=False) for ax in axs]
#
#
# for ind in [0,2]:
#     ax.set_xticks(ticks = non_preg_lab_pos[0::2])
#     ax.set_xticks(ticks = non_preg_lab_pos[1::2], minor=True)
#     ax.set_xlim(white_no_preg_df["x_ind"].min(), white_no_preg_df["x_ind"].max())
#     _ = axs[0].set_xticklabels(labels=non_preg_labels[::2], minor=False, rotation=0)
#     _ = axs[0].set_xticklabels(labels=non_preg_labels[1::2], minor=True, rotation=0)
#     _ = axs[0].tick_params(axis="x", which="minor", length=2)
#     _ = axs[0].tick_params(axis="x", which="major", length=14)
#
# for xtick in axs[2].get_xticklabels():
#     xtick.set_color(get_phe_color(xtick.get_text()))
# for xtick in axs[2].get_xticklabels(minor=True):
#     xtick.set_color(get_phe_color(xtick.get_text()))
#
#
# # set_xticks(axs[3], preg_edge_pos, preg_lab_pos, black_preg_df, preg_labels)
#
# # label
# for ax_ind, df, bonf_pval in zip(np.arange(0,5), [black_no_preg_df, black_preg_df, white_no_preg_df, white_preg_df], [bl_bonferroni_threshold, bl_bonferroni_threshold, wh_bonferroni_threshold, wh_bonferroni_threshold]):
#
#
#     if ((ax_ind == 1) | (ax_ind== 3)):
#         to_label_df = df.loc[df['p']<bonf_pval,:].sort_values('p', ascending=True).head(8)
#     else:
#         temp_to_label_df = df.loc[df['p']<bonf_pval,:].copy()
#         to_label_df = temp_to_label_df.groupby('category').head(5)
#
#     annotations = [axs[ax_ind].text(row['x_ind'], row['-log10_p'],row['phenotype'], fontsize=4) for ind, row in to_label_df.iterrows()]
#     adjust_text(annotations,arrowprops=dict(arrowstyle='->', color='gray'), ax=axs[ax_ind],
#                 force_text=(0.1,0.25), only_move={'points':'y', 'text':'y', 'objects':'y'}, color='red')
#
# # plt.savefig(os.path.join(OUTPUT_DIR, f"{DATE}_assoc_w_ptb_label_v2.pdf"), transparent=True)
#
# # plt.savefig(os.path.join(OUTPUT_DIR, f"{DATE}_assoc_w_ptb_label_v1.pdf"), transparent=True)
# # plt.savefig(os.path.join(OUTPUT_DIR, f"{DATE}_assoc_w_ptb_no_label_v1.pdf"), transparent=True)
#
# # %%
#
#
#
# # %%
# # write to excel
# keep_cols = ['predictor',  'phenotype', 'beta', 'SE', 'OR', 'p', '-log10_p', 'pass_bonf','bonf', 'n_total', 'n_cases', 'n_controls', 'category', 'x_ind']
# with pd.ExcelWriter(OUTPUT_DIR.joinpath('ptb_association_black_and_white.xlsx')) as writer:
#     for name, df in zip(['black_no_preg', 'black_preg',], [black_no_preg_df, black_preg_df]):
#
#         df['bonf'] = bl_bonferroni_threshold
#         df['pass_bonf'] = df['p']<bl_bonferroni_threshold
#         df.loc[:, keep_cols].to_excel(writer, sheet_name=name)
#
#     for name, df in zip(['white_no_preg','white_preg'], [ white_no_preg_df, white_preg_df]):
#         df['bonf'] = wh_bonferroni_threshold
#         df['pass_bonf'] = df['p']<wh_bonferroni_threshold
#         df.loc[:, keep_cols].to_excel(writer, sheet_name=name)
#
#
