#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on: 2022-05-05 14:05:42
# %% - imports 
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import pickle
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle
DATE = datetime.now().strftime('%Y-%m-%d')

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib
import matplotlib.font_manager as font_manager
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist


fpath='/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf'
font_dirs = ['/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf', ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)
mpl.rcParams['font.family'] = 'Arial'




# %% - paths
OUTPUT_DIR=Path("/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/manuscript/latent_factors_ptb_term/prs_assoc")
this_label="preterm_term_all_race_icd9"
this_constraint="phe_ortho"

# %% - FUNCTION

def process_and_cluster(lf_prs_df):

   
    # 1. Remove rows with missing Specific_Trait_Category
    df = lf_prs_df.loc[:, ['xlab','lf_num','coef','Specific_Trait_Category']].copy()
    df_clean = df.loc[df['Specific_Trait_Category']!=""].copy()
    
    # 2. Process each unique trait category
    processed_dfs = []
    for trait in df_clean['Specific_Trait_Category'].unique():
        # Subset data for current trait
        trait_df = df_clean[df_clean['Specific_Trait_Category'] == trait]
        
        # Create wide format dataframe
        wide_df = trait_df.pivot(index='lf_num', 
                               columns='xlab', 
                               values='coef')
        
        # Fill NaN values with 0 for clustering
        wide_df_filled = wide_df.fillna(0)
        
        # Perform hierarchical clustering on columns
        if wide_df_filled.shape[1] > 1:  # Only cluster if there's more than one column
            # Calculate distance matrix for columns
            dist_matrix = pdist(wide_df_filled.T, metric='euclidean')
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(dist_matrix, method='complete')
            
            # Get the order of columns based on clustering
            column_order = leaves_list(linkage_matrix)
            
            # Reorder columns based on clustering
            reordered_columns = wide_df.columns[column_order]
            wide_df = wide_df[reordered_columns]
        
        # Append to list of processed dataframes
        processed_dfs.append(wide_df)
    
    # Concatenate all processed dataframes
    wide_coef_df = pd.concat(processed_dfs, axis=1)

    # create p-value wide df that is correctly ordered
    pval_wide_df = lf_prs_df.pivot(index='xlab', columns='lf_num', values='sig_marker').T.copy()
    wide_pval_df = pval_wide_df.loc[:, clust_df.columns].copy() # reorder



    return wide_coef_df, wide_pval_df

def clustered_heatmap(ptb_wide_coef_df, ptb_wide_pval_df, plot_title): 
    cm = sns.clustermap(
        ptb_wide_coef_df, figsize=(ptb_wide_coef_df.shape[1]*0.3, 14), cmap=cmap,  center=0,
        dendrogram_ratio=(0.01, 0.01), col_cluster=False, row_cluster=False,
        xticklabels=1, linewidths=.2, linecolor='white',
        cbar_pos=(0.3, -0.1, 0.4, 0.03),  # Move the legend to the bottom
        cbar_kws={"orientation": "horizontal"},
        vmin=-0.04, vmax=0.04     # Make the colorbar horizontal
    )

    # Get the axes to draw the rectangles and circles
    ax = cm.ax_heatmap
    cm.ax_heatmap.set_yticklabels(cm.ax_heatmap.get_yticklabels(), rotation=0, horizontalalignment='right')
    cm.fig.suptitle(plot_title, y=1.05, fontsize=16)


    # Loop through the DataFrame to find cells to highlight
    for i in range(ptb_wide_pval_df.shape[0]):  # Rows
        for j in range(ptb_wide_pval_df.shape[1]):  # Columns
            value = ptb_wide_pval_df.iloc[i, j]


            if value == "***":  # Highly significant
                # Draw the regular rectangle
                rect = Rectangle(
                    (j, i), 1, 1,  # (x, y), width, height
                    edgecolor='#e60000', linewidth=1.5, facecolor='none'
                )
                ax.add_patch(rect)

                # Add a circle in the center of the rectangle
                circle = Circle(
                    (j + 0.5, i + 0.5),  # Center of the cell
                    radius=0.19,  # Circle size
                    color='#e60000',  # Circle color
                    alpha=1  # Slight transparency
                )
                ax.add_patch(circle)


            elif value == "*":  # Moderately significant
                # Draw the regular rectangle
                rect = Rectangle(
                    (j, i), 1, 1,  # (x, y), width, height
                    edgecolor='#636363', linewidth=1.5, facecolor='none'
                )
                ax.add_patch(rect)

                # Add a circle in the center of the rectangle
                circle = Circle(
                    (j + 0.5, i + 0.5),  # Center of the cell
                    radius=0.12,  # Circle size
                    color='#636363',  # Circle color
                    alpha=1  # Slight transparency
                )
                ax.add_patch(circle)


    plt.show()
    return cm

def unclust_heatmap(coef_df, pval_df, plot_title): 
    sns.set_theme(style="ticks",  font_scale=1.0, rc={"figure.figsize": (27, 24)})
    cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True)

    # Create your clustermap
    cm = sns.clustermap(
        coef_df, figsize=(27, 14), cmap=cmap,  center=0,
        dendrogram_ratio=(0.01, 0.01), col_cluster=False, row_cluster=False,
        xticklabels=1, linewidths=.2, linecolor='white',
        cbar_pos=(0.3, -0.1, 0.4, 0.03),  # Move the legend to the bottom
        cbar_kws={"orientation": "horizontal"},
        vmin=-0.04, vmax=0.04     # Make the colorbar horizontal
    )



    # Get the axes to draw the rectangles and circles
    ax = cm.ax_heatmap
    cm.ax_heatmap.set_yticklabels(cm.ax_heatmap.get_yticklabels(), rotation=0, horizontalalignment='right')
    cm.fig.suptitle(plot_title, y=1.05, fontsize=16)

    # Loop through the DataFrame to find cells to highlight
    for i in range(pval_df.shape[0]):  # Rows
        for j in range(pval_df.shape[1]):  # Columns
            value = pval_df.iloc[i, j]
            if value == "***":  # Highly significant
                # Draw the regular rectangle
                rect = Rectangle(
                    (j, i), 1, 1,  # (x, y), width, height
                    edgecolor='#b20000', linewidth=1.5, facecolor='none'
                )
                ax.add_patch(rect)

                # Add a circle in the center of the rectangle
                circle = Circle(
                    (j + 0.5, i + 0.5),  # Center of the cell
                    radius=0.15,  # Circle size
                    color='#b20000',  # Circle color
                    alpha=0.8  # Slight transparency
                )
                ax.add_patch(circle)

            elif value == "*":  # Moderately significant
                # Draw the regular rectangle
                rect = Rectangle(
                    (j, i), 1, 1,  # (x, y), width, height
                    edgecolor='#636363', linewidth=1.5, facecolor='none'
                )
                ax.add_patch(rect)

                # Add a circle in the center of the rectangle
                circle = Circle(
                    (j + 0.5, i + 0.5),  # Center of the cell
                    radius=0.15,  # Circle size
                    color='#636363',  # Circle color
                    alpha=0.8  # Slight transparency
                )
                ax.add_patch(circle)

    # Adjust layout and display

    plt.show()
    return cm

# %% 
def mod_unclust_heatmap(coef_df, pval_df, plot_title, label_dict, ht=14, wd=27): 
    sns.set_theme(style="ticks",  font_scale=1.0, rc={"figure.figsize": (27, 24)})
    cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True)


    # Create your clustermap
    cm = sns.clustermap(
        coef_df, figsize=(wd, ht), cmap=cmap,  center=0,
        dendrogram_ratio=(0.01, 0.01), col_cluster=False, row_cluster=False,
        xticklabels=1, linewidths=.2, linecolor='white',
        cbar_pos=(0.3, 0, 0.4, 0.03),  # Move the legend to the bottom
        cbar_kws={"orientation": "horizontal"},
        vmin=-0.04, vmax=0.04     # Make the colorbar horizontal
    )

    
    # Bottom Tick Labels (default)
    ax = cm.ax_heatmap
    xtick_labels = ax.get_xticklabels()
    top_labels = [label_dict[x.get_text()] for x in xtick_labels]
    bottom_labels = [x.get_text().split("_")[-1] for x in xtick_labels]
    ax.xaxis.tick_bottom()
    ax.tick_params(axis="x", labelbottom=True)

    # Top Tick Labels: Create a secondary x-axis for different labels
    top_ax = ax.figure.add_axes(ax.get_position())  # Add a new axis overlaying the heatmap
    top_ax.set_frame_on(False)  # Hide axis frame
    top_ax.yaxis.set_visible(False)  # Hide the y-axis

    # Configure ticks for the top
    # top_labels = ["Top1", "Top2", "Top3", "Top4", "Top5"]  # Custom top labels
    top_ax.set_xticks(ax.get_xticks())
    top_ax.set_xticklabels(top_labels, rotation=90, ha="center", fontsize=10)

    # Align with the bottom axis
    top_ax.xaxis.tick_top()
    # top_ax.tick_params(axis="x", labeltop=True)
    top_ax.tick_params(axis="x", labeltop=True)  # Adjust pad to move ticks closer


    # update bottom xtick labels 
    ax.set_xticklabels(bottom_labels, rotation=90, ha='center')  # Bottom labels


    # Loop through the DataFrame to find cells to highlight
    for i in range(pval_df.shape[0]):  # Rows
        for j in range(pval_df.shape[1]):  # Columns
            value = pval_df.iloc[i, j]
            if value == "***":  # Highly significant
                # Draw the regular rectangle
                rect = Rectangle(
                    (j, i), 1, 1,  # (x, y), width, height
                    edgecolor='#b20000', linewidth=1.5, facecolor='none'
                )
                ax.add_patch(rect)

                # Add a circle in the center of the rectangle
                circle = Circle(
                    (j + 0.5, i + 0.5),  # Center of the cell
                    radius=0.15,  # Circle size
                    color='#b20000',  # Circle color
                    alpha=0.8  # Slight transparency
                )
                ax.add_patch(circle)

            elif value == "*":  # Moderately significant
                # Draw the regular rectangle
                rect = Rectangle(
                    (j, i), 1, 1,  # (x, y), width, height
                    edgecolor='#636363', linewidth=1.5, facecolor='none'
                )
                ax.add_patch(rect)

                # Add a circle in the center of the rectangle
                circle = Circle(
                    (j + 0.5, i + 0.5),  # Center of the cell
                    radius=0.15,  # Circle size
                    color='#636363',  # Circle color
                    alpha=0.8  # Slight transparency
                )
                ax.add_patch(circle)

    # Adjust layout and display
    # plt.subplots_adjust(top=0.97, bottom=0.1)
    plt.show()
    return cm


# %% MAIN, load data 
all_lf_prs_plot_df = pd.read_csv(OUTPUT_DIR.joinpath('all_lf_prs_plot_df.tsv'), sep="\t")
ptb_lf_prs_plot_df = pd.read_csv(OUTPUT_DIR.joinpath('ptb_lf_prs_plot_df.tsv'), sep="\t")
no_ptb_lf_prs_plot_df = pd.read_csv(OUTPUT_DIR.joinpath('no_ptb_lf_prs_plot_df.tsv'), sep="\t")
# %%

ptb_lf_prs_plot_df['lf_num'] = ptb_lf_prs_plot_df['lf'].map(lambda x: int(x.split('V')[-1]))
ptb_lf_prs_plot_df['sig_marker'] = ''
ptb_lf_prs_plot_df.loc[ptb_lf_prs_plot_df["p-value"] <0.05, 'sig_marker'] = '*'
ptb_lf_prs_plot_df.loc[(ptb_lf_prs_plot_df["pass_fdr"]==True), 'sig_marker'] = '***'
ptb_lf_prs_plot_df['Specific_Trait_Category'].fillna('', inplace=True)
ptb_lf_prs_plot_df.sort_values(['Specific_Trait_Category','lf'], inplace=True)
ptb_lf_prs_plot_df['xlab'] = ptb_lf_prs_plot_df['Specific_Trait_Category'] + "_" + ptb_lf_prs_plot_df['Trait_ID']

no_ptb_lf_prs_plot_df['lf_num'] = no_ptb_lf_prs_plot_df['lf'].map(lambda x: int(x.split('V')[-1]))
no_ptb_lf_prs_plot_df['sig_marker'] = ''
no_ptb_lf_prs_plot_df.loc[no_ptb_lf_prs_plot_df["p-value"] <0.05, 'sig_marker'] = '*'
no_ptb_lf_prs_plot_df.loc[(no_ptb_lf_prs_plot_df["pass_fdr"]==True), 'sig_marker'] = '***'
no_ptb_lf_prs_plot_df['Specific_Trait_Category'].fillna('', inplace=True)
no_ptb_lf_prs_plot_df.sort_values(['Specific_Trait_Category','lf'], inplace=True)
no_ptb_lf_prs_plot_df['xlab'] = no_ptb_lf_prs_plot_df['Specific_Trait_Category'] + "_" + no_ptb_lf_prs_plot_df['Trait_ID']


# %%
keep_cols = ['coef','p-value', 'pass_fdr','fdr_bh_adjusted_pvalue', 'nobs', 'xlab', 'Specific_Trait_Category', 'lf_num', 'sig_marker' ]
# verify short label for trait with prior figure in manuscript 
prs_plot_file = "/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/manuscript/latent_factors_ptb_term/prs_assoc_case_control/2024-12-16_ptb_assoc_with_prs_effect_size_grouped_for_plot.csv"
prs_plot_df = pd.read_csv(prs_plot_file)
short_label_dict = prs_plot_df.loc[:,['topic','Trait Label']].set_index('topic').to_dict()['Trait Label']

# remove associations without a trait label
filt_ptb_df = ptb_lf_prs_plot_df.loc[ptb_lf_prs_plot_df['Specific_Trait_Category']!="",:].copy()
filt_no_ptb_df = no_ptb_lf_prs_plot_df.loc[no_ptb_lf_prs_plot_df['Specific_Trait_Category']!="",:].copy()

# keep PRS scores from PGS catalog that startes with "PGS" and those that are in the short label dict
filt_pgs_ptb_df = filt_ptb_df.loc[filt_ptb_df['Trait_ID'].str.startswith("PGS") & filt_ptb_df['Trait_ID'].isin(short_label_dict.keys()),:].copy()
filt_pgs_no_ptb_df = filt_no_ptb_df.loc[filt_no_ptb_df['Trait_ID'].str.startswith("PGS") & filt_no_ptb_df['Trait_ID'].isin(short_label_dict.keys()),:].copy()


# p<0.05 associations: PTB=115, no_PTB 135 
filt_pgs_ptb_df.loc[filt_pgs_ptb_df['p-value'] < 0.05, keep_cols]
filt_pgs_no_ptb_df.loc[filt_pgs_no_ptb_df['p-value'] < 0.05, keep_cols]

# pass FDR associations: PTB=6, no_PTB=7
filt_pgs_ptb_df.loc[filt_pgs_ptb_df["pass_fdr"]==True, keep_cols]
filt_pgs_no_ptb_df.loc[filt_pgs_no_ptb_df["pass_fdr"]==True, keep_cols]


# check taht there is no traits without a short label
set(filt_pgs_ptb_df['Trait_ID'].unique()).difference( short_label_dict.keys())
set(filt_pgs_no_ptb_df['Trait_ID'].unique()).difference( short_label_dict.keys())

filt_pgs_ptb_df['plot_trait_label']= filt_pgs_ptb_df['Trait_ID'].map(short_label_dict)
filt_pgs_no_ptb_df['plot_trait_label']= filt_pgs_no_ptb_df['Trait_ID'].map(short_label_dict)



# %% --- tidy data for heatmap


manual_order =['A1C', 'T1DM', 'T2DM',  'CVD', 'TGs', 'Cholesterol', 'HDL', 'LDL', 'BMI', 'WHR','SBP', 'DBP',   'Depression',  'Schizophrenia', 'BMD']
filt_pgs_ptb_df['plot_trait_label'] = pd.Categorical(filt_pgs_ptb_df['plot_trait_label'], categories=manual_order, ordered=True)
filt_pgs_ptb_df.sort_values(['plot_trait_label','coef'], ascending=True, inplace=True)

filt_pgs_no_ptb_df['plot_trait_label'] = pd.Categorical(filt_pgs_no_ptb_df['plot_trait_label'], categories=manual_order, ordered=True)
filt_pgs_no_ptb_df.sort_values(['plot_trait_label','coef'], ascending=True, inplace=True)


# create short label dict for plotting
ptb_short_dict = filt_pgs_ptb_df.loc[:,['xlab','plot_trait_label']].set_index('xlab').to_dict()['plot_trait_label']
no_ptb_short_dict = filt_pgs_no_ptb_df.loc[:,['xlab','plot_trait_label']].set_index('xlab').to_dict()['plot_trait_label']


# pivot PTB data 
ptb_wide_df = filt_pgs_ptb_df.pivot(index='xlab', columns='lf_num', values='coef').T.copy()
ptb_pval_wide_df = filt_pgs_ptb_df.pivot(index='xlab', columns='lf_num', values='sig_marker').T.copy()

# pivot no_ptb data 
no_ptb_wide_df = filt_pgs_no_ptb_df.pivot(index='xlab', columns='lf_num', values='coef').T.copy()
no_ptb_pval_wide_df = filt_pgs_no_ptb_df.pivot(index='xlab', columns='lf_num', values='sig_marker').T.copy()

# check that rows and columns match 
dfs = [ptb_wide_df, ptb_pval_wide_df, no_ptb_wide_df, no_ptb_pval_wide_df]

rows_match = all(dfs[0].index.equals(df.index) for df in dfs)
columns_match = all(dfs[0].columns.equals(df.columns) for df in dfs)
if not rows_match:
        warnings.warn("Row orders do not match across all dataframes.")
if not columns_match:
    warnings.warn("Column orders do not match across all dataframes.")




# %% 
# -----
# heatmap with columns UNCLUSTERED within trait groups 
# -----
ptb_cm = unclust_heatmap(ptb_wide_df, ptb_pval_wide_df, "PTB")
no_ptb_cm = unclust_heatmap(no_ptb_wide_df, no_ptb_pval_wide_df, "NO ptb")


ht=14
wd=28
ptb_cm = mod_unclust_heatmap(ptb_wide_df, ptb_pval_wide_df, "PTB", ptb_short_dict, ht=ht, wd=wd)
no_ptb_cm = mod_unclust_heatmap(no_ptb_wide_df, no_ptb_pval_wide_df, "NO PTB", no_ptb_short_dict, ht=ht, wd=wd)

# save
# ptb_cm.savefig(OUTPUT_DIR.joinpath(f"{DATE}_PTB_lf_assoc_prs_{ht}_{wd}.pdf"))
# no_ptb_cm.savefig(OUTPUT_DIR.joinpath(f"{DATE}_no_ptb_lf_assoc_prs_{ht}_{wd}.pdf"))


# %%s
# -----
# heatmap with columns clustered within trait groups 
# -----

# tidy data 
ptb_wide_coef_df, ptb_wide_pval_df = process_and_cluster(ptb_lf_prs_plot_df)
no_ptb_wide_coef_df, no_ptb_wide_pval_df = process_and_cluster(no_ptb_lf_prs_plot_df)

# note using a clustered heatmap means the columsn will not align between PTB and Not-PTB
clustered_heatmap(ptb_wide_coef_df, ptb_wide_pval_df, 'PTB')
clustered_heatmap(no_ptb_wide_coef_df, no_ptb_wide_pval_df, 'Not-PTB')