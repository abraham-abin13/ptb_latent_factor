#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on: 2021-03-02 08:49:12

# %% 
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection


sys.path.append("/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/r_tensor_decomp")
from helper_analyze_ranks import latentAnalysis, latentWeights

DATE = datetime.now().strftime('%Y-%m-%d')

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
%matplotlib inline

import matplotlib
import matplotlib.font_manager as font_manager
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

fpath='/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf'
font_dirs = ['/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf', ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)


mpl.rcParams['font.family'] = 'Arial'


data_root=Path("/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/input_tensors/preterm_term_all_race_icd9_no_preg")
td_filepaths={
        "analysis_label": "preterm_term_all_race_icd9_no_preg",
        "input_tensor_file": data_root.joinpath("tensor_preterm_term_all_race_icd9_no_preg_within_5yr.pickle"),
        "phe_axis_file": data_root.joinpath("phecode_axis_preterm_term_all_race_icd9_no_preg_within_5yr.pickle"),
        "time_axis_file": data_root.joinpath("binned_years_axis_preterm_term_all_race_icd9_no_preg_within_5yr.pickle"),
        "grid_axis_file": data_root.joinpath("grids_axis_preterm_term_all_race_icd9_no_preg_within_5yr.pickle"),
        "lf_analysis_dir": "/dors/capra_lab/users/abraha1/projects/PTB_phewas/data/2020_07_30_longit_topic_modeling/r_parafac_outputs/preterm_term_all_race_icd9_no_preg",
        "fig_output_dir": "/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/manuscript/latent_factors_ptb_term",
    }

OUTPUT_DIR=Path("/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/manuscript/latent_factors_ptb_term")
describe_output_dir = OUTPUT_DIR.joinpath("describe_lfs")
short_phe_names_dict = {'circulatory system':'circulatory',
 'congenital anomalies':'congenital',
 'dermatologic':'dermatologic',
 'digestive':'digestive',
 'endocrine/metabolic':'endocrine',
 'genitourinary':'genitourinary',
 'hematopoietic':'hematopoietic',
 'infectious diseases':'infectious',
 'injuries & poisonings': 'injuries',
 'mental disorders':'mental health',
 'musculoskeletal':'musculoskeletal',
 'neoplasms':'neoplasms',
 'neurological':'neurological',
 'pregnancy complications':'pregnancy',
 'respiratory':'respiratory',
 'sense organs':'sensory',
 'symptoms':'symptoms'}

def plot_top_phe(pheno_lf_df, phe_df, lf_colname, lf_col_ind, pax,  patch_height, hide_xticks=True,label_top_n=False, top_marker_size=5, fontsize=6):


    phe_merged_df = pd.merge(pheno_lf_df, phe_df, left_on='phecodes', right_on='phenotype')
    phe_df.sort_values('xind', inplace=True)

    cat_list = []
    cat_midpoint = []
    cat_minor = []
    for cat in phe_df['group'].unique():
        vals = phe_df.loc[phe_df['group']==cat, 'xind'].values
        midpoint = (np.min(vals) + np.max(vals))/2
        cat_list.append(cat)
        cat_midpoint.append(midpoint)
        cat_minor.append(np.min(vals))



    # update category label
    cat_list = [short_phe_names_dict[x] for x in cat_list]
    # for ind, lf in enumerate(np.arange(1,rank+1)):
    # lf_colname = 'V1'
    # lf_col_ind=0
    all_bool = phe_merged_df[f'V{lf}'] !=0
    pax.scatter(phe_merged_df.loc[all_bool, 'xind'], phe_merged_df.loc[all_bool, lf_colname],  s=1,alpha=0.8, label='', color='gray', zorder=0)

    bool = top_phe_df['latent_factor'] == lf_colname
    top_plot_df = top_phe_df.loc[bool,:]


    y = top_plot_df.loc[:, 'wt']
    x = top_plot_df.loc[:, 'xind']
    pax.scatter(x, y,  s=top_marker_size, alpha=1, label=lf_colname, color=np.tile(lf_colors[lf_col_ind], (len(x),1)), zorder=2)


    if label_top_n:
        # label the top 3 phecodes in each chapter (also in the top 10)
        top_phe_lf_df = top_phe_df.loc[top_phe_df['latent_factor']==lf_colname, ['xind', 'group','norm_wt', 'wt', 'phecodes']].copy()
        top_phe_lf_df.sort_values(['group','norm_wt'], inplace=True, ascending=False)
        for_label_df = top_phe_lf_df.groupby('group').head(label_top_n).copy()
        for ind, row in for_label_df.iterrows():
            pax.annotate(row.phecodes, xy=(row.xind, row.wt), fontsize=fontsize)


    pax.tick_params(axis="x", which="minor", length=4)
    pax.tick_params(axis="x", which="major", length=4)
    pax.set_xticks(cat_midpoint[::2],)
    pax.set_xticks(cat_midpoint[1::2], minor=True)
    pax.set_xticklabels(cat_list[::2], rotation=270, ha='center',  fontsize=fontsize)
    pax.set_xticklabels(cat_list[1::2], minor=True, rotation=270, ha='center', fontsize=fontsize)
    if hide_xticks:
        pax.tick_params(axis='x', which='both', bottom=False,top=False,labelbottom=False)

    pax.tick_params(axis='y', labelsize=6 )
    pax.set_xlim(0, phe_df.shape[0])
    pax.set_ylim(0, patch_height)
    # pax.legend(ncol=1,loc='center left', bbox_to_anchor=(1, 0.5), borderpad=0.2, labelspacing=0.5, fontsize=6,
    #       handletextpad=0.01, borderaxespad=0.1, columnspacing=0.1, title="Latent\nFactors")

    patches_1 = []
    patches_2 = []
    for i in np.arange(0, len(cat_minor)):

        if (i != (len(cat_minor)-1)):
            rect = mpatches.Rectangle(xy=(cat_minor[i], 0), width=cat_minor[i+1] - cat_minor[i], height=patch_height)
            if (i%2==0):
                patches_1.append(rect)
            else:
                patches_2.append(rect)

        elif i == (len(cat_minor)-1):
            rect = mpatches.Rectangle(xy=(cat_minor[i], 0), width=phe_df.shape[0] - cat_minor[i], height=patch_height)
            if (i%2==0):
                patches_1.append(rect)
            else:
                patches_2.append(rect)



    collection1 = PatchCollection(patches_1, color='lightgray', alpha=0.3, zorder=-1)
    collection2 = PatchCollection(patches_2, color='lightgray', alpha=0, zorder=-1)
    pax.add_collection(collection1)
    pax.add_collection(collection2)
    return pax


def phe_by_chapter(pheno_lf_df, phe_df, rank, lf_colors, fontsize=8, height=4, aspect=1.75, min_dot_size=1, max_dot_size=200):


    phecodes_df = pd.merge(pheno_lf_df, phe_df, left_on='phecodes', right_on='phenotype', how='inner')
    phecodes_df['short_name_group'] = phecodes_df['group'].map(short_phe_names_dict)

    mean_wt_per_chapter_df = phecodes_df.groupby('short_name_group').mean().reset_index()
    mean_wt_per_chapter_df.drop(columns={'xind'}, inplace=True)
    mean_wt_per_chapter_df.set_index('short_name_group',inplace=True)
    norm_mean_wt_per_chapter_df = mean_wt_per_chapter_df.apply(lambda x: (x - x.min())/(x.max()-x.min()), axis=0)
    long_norm_wt_df= pd.melt(norm_mean_wt_per_chapter_df.reset_index(), id_vars=['short_name_group'], value_vars=[f'V{rank}' for rank in np.arange(1,rank+1)], value_name='weight', var_name='factor')

    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': fontsize})
    g = sns.relplot(
        data=long_norm_wt_df,
        x="short_name_group", y="factor", hue="weight", size="weight",
        palette="Grays", hue_norm=(0, 1), edgecolor="black",
        height=height, aspect=aspect, sizes=(min_dot_size, max_dot_size), size_norm=(0, 1),
    )

    # Tweak the figure to finalize
    g.set(xlabel="", ylabel="", aspect="equal")
    g.despine(left=True, bottom=True)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)

    lg = g._legend.remove()
    plt.legend(ncol=6, borderaxespad=0, handletextpad=0, bbox_to_anchor= (0.95, 1.45), fontsize=fontsize, mode=None,
                title='Weight', title_fontsize=fontsize, labelspacing=0.5, columnspacing=1)
    # for artist in g.legend.legendHandles:
    #     artist.set_edgecolor("1")
    # lg = g._legend
    # lg.set_bbox_to_anchor((0.15, .65))
    # plt.setp(g.legend.get_title(), fontsize=fontsize+1)
    # plt.setp(g.legend.get_texts(), fontsize=fontsize)



    # modify y tick labels
    _ = g.ax.set_yticklabels([x.get_text().replace('V', 'Factor ') for x in g.ax.get_yticklabels()], fontsize=fontsize)
    _ = g.ax.set_xticklabels(g.ax.get_xticklabels(), fontsize=fontsize)
    g.ax.set_ylim(rank-1,0)

    # set grids
    g.ax.grid(True, which='major', axis='x', alpha=1, linewidth=0.25 )
    g.ax.grid(True, which='major', axis='y', alpha=0)

    for lf, col in enumerate(lf_colors):
        g.ax.axhline(lf ,color=col, linewidth=0.75, zorder=0)

    plt.subplots_adjust(left=0.3, top=1, bottom =0.3, right=0.95)


# %%
###
###    laod data
###


rank=33
this_experiment = latentWeights(td_filepaths['phe_axis_file'],
            td_filepaths['grid_axis_file'],
            td_filepaths['time_axis_file'],
            td_filepaths['lf_analysis_dir'], rank,
            td_filepaths['fig_output_dir'],
            td_filepaths['analysis_label'])

# prepare variables
top_phe_df = this_experiment.get_top_n_phenotypes()
pheno_lf_df = this_experiment.pheno_lf_df
time_lf_df = this_experiment.time_lf_df
individ_lf_df = this_experiment.individ_lf_df
phe_df = this_experiment.phe_df
analysis_label = this_experiment.analysis_label
lf_cols = [f"V{lf}" for lf in np.arange(1,rank+1)]


# %%
### COMBINE TIME AND PHE

lfs_to_plot = [1, 10, 33]
lf_cols = [f"V{lf}" for lf in lfs_to_plot]

lf_colors = sns.color_palette("husl", len(lfs_to_plot))

time_y_max = np.ceil(time_lf_df.loc[:, lf_cols].max().max())+0.2
time_y_min= 0
time_xticks = time_lf_df.loc[:, 'time_from_delivery'].values
time_yticks = np.arange(0, time_y_max+1 ,1)

pheno_y_max = pheno_lf_df[lf_cols].max().max().round()+10
pheno_yticks = np.arange(0, pheno_y_max,10)

individ_lf_df[lf_cols].max().max()

indv_max = np.round(individ_lf_df[lf_cols].max().max(),3)
indv_min = np.round(individ_lf_df[lf_cols].min().min(),3)

ncols=3
fontsize=8
fig, axs = plt.subplots(nrows=len(lfs_to_plot), ncols=ncols, sharex=False, sharey=False, figsize=(2*ncols, len(lfs_to_plot)*0.7))
for lfi, lf in enumerate(lfs_to_plot):

    tax = axs[lfi, 0]
    tax.plot(time_lf_df['time_from_delivery'], time_lf_df[f'V{lf}'], marker='o', markersize=2, linewidth=1, color=lf_colors[lfi])
    tax.set_ylim(0,time_y_max)
    tax.set_yticks(time_yticks)
    tax.set_yticklabels([int(x) for x in time_yticks], fontsize=fontsize)
    tax.set_xticks(time_xticks)
    tax.fill_between(time_lf_df['time_from_delivery'], time_lf_df[f'V{lf}'], interpolate=True, color=lf_colors[lfi], alpha=0.2)
    # axs[lf-1,0].tick_params(axis='both', which='major', length=2)
    if lf == rank:
        axs[lfi,0].set_xticklabels(time_xticks, fontsize=fontsize)
    else:
        axs[lfi,0].set_xticklabels('', fontsize=fontsize)


    pax = axs[lfi, 1]
    lf_name=f"V{lf}"
    # lf_ind = lf - 1
    _ = plot_top_phe(pheno_lf_df, phe_df, lf_name, lfi ,pax,  patch_height=pheno_y_max, hide_xticks=(lf!=rank), label_top_n= False, top_marker_size=2, fontsize=fontsize)
    pax.set_yticks(pheno_yticks)
    pax.set_yticklabels([int(x) for x in pheno_yticks], fontsize=fontsize)
    pax.set_yticklabels([int(x) for x in pheno_yticks], fontsize=fontsize)


    hax = axs[lfi, 2]
    sns.heatmap(np.expand_dims(individ_lf_df[f'V{lf}'].values, axis=-1).T, cbar=True, vmin=indv_min, vmax=indv_max, cmap=sns.light_palette(lf_colors[lfi], as_cmap=True), ax=hax, rasterized=True)
    hax.set_xticklabels('')
    hax.set_yticklabels('')
    hax.tick_params(length=0, width=0, axis='x', which='major')
    hax.tick_params(length=0, width=0, axis='y', which='major')
    [hax.spines[side].set_linewidth(2) for side in hax.spines.keys()]

    for ax in [pax, tax]:

        # for ax in axs_.ravel():
        [ax.spines[side].set_linewidth(0.25) for side in ax.spines.keys()]
        ax.tick_params(axis='both', which='both', length=2, width=0.25, direction='in')
        ax.tick_params(axis='x', which='both', length=2, width=0.25, direction='out')



plt.subplots_adjust(hspace=0.25, left=0.05, top=0.98, bottom=0.3)
# plt.tight_layout()
# plt.savefig(OUTPUT_DIR.joinpath(f"{DATE}_lf_profile.pdf"))

# %% 
## TOP PHECODES PER LATENT FACTOR 

# Identify latent factor columns
lf_cols = [col for col in pheno_lf_df.columns if col.startswith('V')]

top_phecodes = []
for lf in lf_cols:
    top10 = pheno_lf_df[['phecodes', lf]].nlargest(10, lf)
    top10 = top10.rename(columns={lf: 'weight'})
    top10['latent_factor'] = lf
    top_phecodes.append(top10)

top_phecodes_df = pd.concat(top_phecodes, ignore_index=True)
top_phecodes_df = pd.merge(top_phecodes_df, phe_df.loc[:, ['phenotype', 'description', 'group']], left_on='phecodes', right_on='phenotype', how='left')

top_phecodes_df[top_phecodes_df['latent_factor'].isin(['V6'])].to_csv(describe_output_dir.joinpath("top_phecodes_V6.tsv"), sep='\t', index=False)
top_phecodes_df[top_phecodes_df['latent_factor'].isin(['V13'])].to_csv(describe_output_dir.joinpath("top_phecodes_V13.tsv"), sep='\t', index=False)


print(top_phecodes_df[['latent_factor', 'phecodes', 'weight']])


# %%

# phe_by_chapter(pheno_lf_df, phe_df, rank, lf_colors, fontsize=8, height=5.5, aspect=0.8, min_dot_size=1, max_dot_size=120)
# plt.savefig(OUTPUT_DIR.joinpath(f"{DATE}_summary_by_chap.pdf"), transparent=True)
# %%
# %%

# # -----------
# # plot time and phecodes seperately
# # -----------
# 7.5/3
# y_max = np.round(time_lf_df.loc[:, lf_cols].max().max())
# y_min= 0
# xticks = time_lf_df.loc[:, 'time_from_delivery'].values
#
# fig, axs = plt.subplots(nrows=rank, sharex=True, sharey=True, figsize=(2.5, 4))
# for lf in np.arange(1,9):
#     axs[lf-1].plot(time_lf_df['time_from_delivery'], time_lf_df[f'V{lf}'], marker='o', markersize=2, linewidth=1, color=lf_colors[lf-1])
#
#     axs[lf-1].set_ylim(0,y_max)
#     axs[lf-1].set_yticks(np.arange(0, y_max ,1))
#     axs[lf-1].set_yticklabels([int(x) for x in np.arange(0, y_max ,1)], fontsize=6)
#     axs[lf-1].set_xticks(xticks)
#     axs[lf-1].set_xticklabels(xticks, fontsize=6)
#     axs[lf-1].fill_between(time_lf_df['time_from_delivery'], time_lf_df[f'V{lf}'], interpolate=True, color=lf_colors[lf-1], alpha=0.2)
#
# # %%
# y_max = 30
# y_min= 0
# yticks = np.arange(0, y_max,10)
# xticks = pheno_lf_df.loc[:, 'phecodes'].values
#
# fig, axs = plt.subplots(nrows=rank, sharex=False, sharey=True, figsize=(2.5, 4))
# for lf in np.arange(1,9):
#     pax = axs[lf-1]
#     pax.set_yticks(yticks)
#     pax.set_yticklabels([int(x) for x in yticks], fontsize=6)
#
#     lf_name=f"V{lf}"
#     lf_ind = lf - 1
#     _ = plot_top_phe(pheno_lf_df, phe_df, lf_name, lf_ind ,pax,  patch_height=y_max, hide_xticks=(lf!=8), label_top_n= False)
#
# %%


pheno_y_max = 30
pheno_yticks = np.arange(0, pheno_y_max+10,10)


ncols=2
fontsize=8
fig, pax = plt.subplots( sharex=False, sharey=False, figsize=(6, 2))
for lf in np.arange(1,9):

    lf_name=f"V{lf}"
    lf_ind = lf - 1
    _ = plot_top_phe(pheno_lf_df, phe_df, lf_name, lf_ind ,pax,  patch_height=pheno_y_max, hide_xticks=(lf!=8), label_top_n= False, top_marker_size=5, fontsize=fontsize)
    pax.set_yticks(pheno_yticks)
    pax.set_yticklabels([int(x) for x in pheno_yticks], fontsize=fontsize)
    pax.set_yticklabels([int(x) for x in pheno_yticks], fontsize=fontsize)

    ax=pax
    [ax.spines[side].set_linewidth(0.25) for side in ax.spines.keys()]
    ax.tick_params(axis='both', which='both', length=2, width=0.25, direction='in')
    ax.tick_params(axis='x', which='both', length=2, width=0.25, direction='out')


# %%
### wriet top to excel
# top_phe_to_write_df = top_phe_df.round(2).copy()
# top_phe_to_write_df.drop(columns=['color','xind'], inplace=True)
# top_phe_to_write_df.to_excel(OUTPUT_DIR.joinpath("top_10_phe.xlsx"))





# %%
