#!/bin/python
# This script will ...
#
# created on: 2021-12-06 17:08:48

# %%
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import statsmodels.stats.multitest as smm
os.getcwd()
sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/r_tensor_decomp')
from helper_prs import load_all_prs_data
DATE = datetime.now().strftime('%Y-%m-%d')


import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

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


# %% PATHS
DIR="/dors/capra_lab/users/abraha1/projects/PTB_phewas/data/2020_07_30_longit_topic_modeling/r_parafac_outputs/preterm_term_all_race_icd9_no_preg/"
OUTPUT_DIR=Path("/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/manuscript/latent_factors_ptb_term/prs_assoc")
this_label="preterm_term_all_race_icd9"
this_constraint="phe_ortho"

tensor_file ="/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/input_tensors/preterm_term_all_race_icd9/tensor_preterm_term_all_race_icd9_within_5yr.pickle"
phecode_axis_file = "/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/input_tensors/preterm_term_all_race_icd9_no_preg/phecode_axis_preterm_term_all_race_icd9_no_preg_within_5yr.pickle"
grid_axis_file="/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/input_tensors/preterm_term_all_race_icd9_no_preg/grids_axis_preterm_term_all_race_icd9_no_preg_within_5yr.pickle"
binned_years_axis_file="/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/input_tensors/preterm_term_all_race_icd9_no_preg/binned_years_axis_preterm_term_all_race_icd9_no_preg_within_5yr.pickle"
delivery_file = "/dors/capra_lab/users/abraha1/projects/PTB_phewas/data/2020_07_30_longit_topic_modeling/raw_data/est_delivery_date_at_least_one_icd_cpt_ega.tsv"


### PRS
COHORT_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/manuscript/phecode_assoc_w_ptb/icd9_assoc_only/white_cohort_for_ptb_assoc.tsv"
COVAR_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/manuscript/phecode_assoc_w_ptb/icd9_assoc_only/covar_for_ptb_assoc.tsv"
PCA_FILE = "/dors/capra_lab/users/abraha1/data/biovu_mega_ex_2019_02_capra_preterm_a3_v1-1/MEGAex_BioVUthru2019-02_BestOfMultipleCalls/mich_imp_server_results/merged/15pc_race_white_merged_filt_imp_v1.eigenvec"


# %% FUNCTIONS

def load_factor_matrix(NUM_LATENT_FACTORS, this_constraint):
    # load factor matrices
    phe_fmatrix = os.path.join(
        DIR,
        f"multiway_rank_{NUM_LATENT_FACTORS}",
        f"rank_{NUM_LATENT_FACTORS}_phecode_factors_{this_constraint}.csv",
    )
    time_fmatrix = os.path.join(
        DIR,    f"multiway_rank_{NUM_LATENT_FACTORS}",
        f"rank_{NUM_LATENT_FACTORS}_time_since_delivery_factors_{this_constraint}.csv",
    )
    # load input tensor axis labels
    phe_labels = pickle.load(open(phecode_axis_file, "rb"))
    grid_labels = pickle.load(open(grid_axis_file, "rb"))
    binned_years_labels = pickle.load(open(binned_years_axis_file, "rb"))
    return phe_labels, grid_labels, binned_years_labels

def load_indivd_lf(NUM_LATENT_FACTORS, this_constraint, grid_labels, ptb_grids):

    individ_fmatrix = os.path.join(
        DIR,
        f"multiway_rank_{NUM_LATENT_FACTORS}",
        f"rank_{NUM_LATENT_FACTORS}_individ_factors_{this_constraint}.csv",
    )

    indivd_lf_df = pd.read_csv(individ_fmatrix)
    indivd_lf_df['GRID']=grid_labels
    indivd_lf_df['preg_type'] = 'not-preterm'
    indivd_lf_df.loc[indivd_lf_df['GRID'].isin(ptb_grids), 'preg_type'] = 'preterm'
    lfs = [f'V{lf}' for lf in np.arange(1,NUM_LATENT_FACTORS+1)]

    return indivd_lf_df, lfs

def run_regression(analysis_df, x_colnames, pca_col_names, y_colname, transform_x=False, transform_y=False, logit=False, verbose=False):
    # lf_univar_prs_df = run_regression(analysis_df, x_colnames=[prs_column], pca_col_names=pca_col_names, y_colname=this_lf, std_x=True, std_y=True)
    # analysis_df, x_colnames=[prs_column], pca_col_names=pca_col_names, y_colname=this_lf, transform_x='std', transform_y='norm', logit=False


    assert len(y_colname) >1, "y_colname has more than one item"



    data_df = analysis_df.copy()
    standardize = lambda df, col_name: (df[col_name] - df[col_name].mean())/df[col_name].std()

    # independent variables (X)
    updated_x_colnames = []
    if transform_x == "std":
        if verbose:
            print("standardizing dependent variables (X).")
        for x_col in x_colnames:
            data_df[f'std_{x_col}'] = standardize(data_df, x_col)
            updated_x_colnames.append(f'std_{x_col}')
    elif transform_x == "norm":
        if verbose:
            print("normalizing dependent variables (X).")
        for x_col in x_colnames:
            scaler = MinMaxScaler()
            data_df[f'std_{x_col}'] = scaler.fit_transform(data_df.loc[:, [x_col]])
            updated_x_colnames.append(f'std_{x_col}')
    else:
        updated_x_colnames = x_colnames

    # dependent variable (Y)
    if transform_y =="std":
        if verbose:
            print("standardizing independent variables (Y).")
        updated_y_colname = f'std_{y_colname}'
        data_df[updated_y_colname] = standardize(data_df, y_colname)
    elif transform_y == "norm":
        if verbose:
            print("normalizing independent variables (Y).")
        scaler = MinMaxScaler()
        data_df[f'std_{y_colname}'] = scaler.fit_transform(data_df.loc[:, [y_colname]])
        updated_y_colname = f'std_{y_colname}'
    else:
        updated_y_colname = y_colname


    # set up indpeendent (X) and dependent (y) variables
    endo_x = data_df.loc[:, updated_x_colnames + pca_col_names].values
    y=data_df[updated_y_colname]
    X = sm.add_constant(endo_x)

    if logit:
        model = sm.Logit(y, X)
        results = model.fit()
        model_pval = results.llr_pvalue
        nobs = results.nobs
    else:
        model = sm.OLS(y, X)
        results = model.fit()
        model_pval = results.f_pvalue
        nobs = results.nobs

    # parse results
    lr_df = pd.read_html(results.summary(xname=['const']+updated_x_colnames+pca_col_names).tables[1].as_html(), header=0,index_col=0)[0]
    lr_df.columns = ['coef','std_err','t','p-value', 'CI_0.025', 'CI_0.975']
    lr_df = lr_df.drop('const').reset_index()
    lr_df.rename({'index':'topic'},inplace=True, axis=1)
    lr_df['lower_err'] = np.array(lr_df['coef']  - lr_df['CI_0.025'])
    lr_df['upper_err'] = np.array(lr_df['CI_0.975']  - lr_df['coef'])
    # lr_df['prs_label'] = prs_column
    lr_df['model_pval'] =model_pval
    lr_df['nobs'] = nobs
    lr_df['independent_variable'] = y_colname
    lr_df['dependent_variable'] = ','.join(x_colnames)

    return lr_df

def prep_ptb_assoc_df_for_plot(data_df, pca_col_names, trait_details_df, ):

    # remove pcs
    prs_names = set(data_df['topic'].unique()).difference(pca_col_names)
    no_pcs_df =  data_df.loc[data_df['topic'].isin(prs_names),:].copy()

    # merge with trait details
    effect_label_df = pd.merge(no_pcs_df, trait_details_df.loc[:, ['Trait_ID', 'Trait_Category', 'Specific_Trait_Category', 'Consortia']], on='Trait_ID',  how='left')

    # modify names
    # effect_label_df.loc[effect_label_df['Mapped Trait(s) (Ontology)']=='diastolic blood pressure', 'Mapped Trait(s) (Ontology)']  = 'blood pressure diastolic'
    # effect_label_df.loc[effect_label_df['Mapped Trait(s) (Ontology)']=='systolic blood pressure', 'Mapped Trait(s) (Ontology)']  = 'blood pressure systolic'
    # effect_label_df['trait_label'] = effect_label_df['Mapped Trait(s) (Ontology)'] +"_" + effect_label_df['prs_label']

    # pvalue correct
    fdr_bool,pval_corr, _, _ = smm.multipletests(    effect_label_df.loc[:, 'p-value'],  method='fdr_bh')
    effect_label_df.loc[:, 'fdr_bh_adjusted_pvalue'] = pval_corr
    effect_label_df['pass_fdr'] = effect_label_df['fdr_bh_adjusted_pvalue']<0.05

    # effect_label_df.sort_values(['trait_label', 'pass_fdr'],inplace=True)
    return effect_label_df

def set_up_for_heatmap(effect_label_df,mapped_to_each_disease_dict, order_key_each_disease_dict):


    effect_label_df['short_label'] =  effect_label_df['Mapped Trait(s) (Ontology)'].map(mapped_to_each_disease_dict)
    effect_label_df['short_label_order'] =  effect_label_df['short_label'].map(order_key_each_disease_dict)
    effect_label_df.sort_values(['short_label_order'], inplace=True)
    #
    # effect_label_df['short_label'] =  effect_label_df['Mapped Trait(s) (Ontology)'].map(mapped_to_each_disease_dict)
    # effect_label_df.sort_values(['big_category', 'Mapped Trait(s) (Ontology)'], inplace=True)
    # effect_label_df['short_label_prs']= effect_label_df['short_label'] + "_" + effect_label_df['dependent_variable']
    category_col_label='short_label'


    # pvalue correct within each category
    updated_pval_df = pd.DataFrame()
    for ind, gdf in effect_label_df.groupby(category_col_label, sort=False):

        fdr_bool,pval_corr, _, _ = smm.multipletests(    gdf.loc[:, 'p-value'],  method='fdr_bh')
        gdf.loc[:, 'p_adj_per_category'] = pval_corr
        gdf.loc[:, 'is_sig_per_category'] = gdf.loc[:, 'p_adj_per_category'] <0.05
        updated_pval_df = updated_pval_df.append(gdf)

    # repeat but correct for all tests
    _,all_pval_corr, _, _ = smm.multipletests(    updated_pval_df.loc[:, 'p-value'],  method='fdr_bh')
    updated_pval_df.loc[:, 'p_adj_all_tests'] = all_pval_corr
    updated_pval_df.loc[:, 'is_sig_all_tests'] = updated_pval_df.loc[:, 'p_adj_all_tests'] <0.05

    def anno_pval(x_row):

        if x_row['p_adj_all_tests'] < 0.05:
            return 'A'
        elif x_row['p_adj_per_category'] < 0.05:
            return 'C'
        elif x_row['p-value'] < 0.05:
            return '*'
        else:
            return ''

    updated_pval_df['p_anno'] = updated_pval_df.apply(lambda x: anno_pval(x), axis=1)

    mod_df = pd.DataFrame()
    mod_pval_df = pd.DataFrame()

    for ind, category in enumerate(updated_pval_df[category_col_label].unique()):

        cat_df = updated_pval_df.loc[updated_pval_df[category_col_label] == category, :].copy()
        cat_df.sort_values("Mapped Trait(s) (Ontology)", inplace=True)

        # sort within inner category
        for iind, inner_cat in enumerate(cat_df["Mapped Trait(s) (Ontology)"].unique()):

            inner_cat_df = cat_df.loc[cat_df["Mapped Trait(s) (Ontology)"] == inner_cat].copy()

            heat_df = (
                inner_cat_df.loc[:,["lf", "Polygenic Score (PGS) ID", "coef"],]
                .pivot(index="lf", columns="Polygenic Score (PGS) ID", values="coef")
                .copy()
            )

            pval_df = (
                inner_cat_df.loc[:,["lf", "Polygenic Score (PGS) ID", "p_anno"],]
                .pivot(index="lf", columns="Polygenic Score (PGS) ID", values="p_anno")
                .copy()
            )

            if heat_df.shape[1] != 1:
                cg = sns.clustermap(heat_df, cmap="vlag", center=0)
                plt.close()
                new_col_order = cg.dendrogram_col.reordered_ind
            else:
                new_col_order = 0

            clust_df = heat_df.iloc[:, new_col_order].copy()
            pval_clust_df = pval_df.iloc[:, new_col_order].copy()
            mod_df = pd.concat([mod_df, clust_df], axis=1)
            mod_pval_df = pd.concat([mod_pval_df, pval_clust_df], axis=1)

        if ind == (updated_pval_df[category_col_label].nunique() - 1):
            continue

        mod_df[f"blank_{ind}"] = 0
        mod_df[f"blank1_{ind}"] = 0
        mod_pval_df[f"blank_{ind}"] = ""
        mod_pval_df[f"blank1_{ind}"] = ""



    return mod_df, mod_pval_df

def plot_heatmap_lf_vs_prs(mod_df, mod_pval_df, lf_prs_plot_df, savefig=False):

    sns.set(style="ticks", font_scale=1, rc={"figure.figsize": (13, 5)})
    grid_kws = {"width_ratios": (0.999999, 0.01), "hspace": 0}
    fig, (ax, cbar_ax) = plt.subplots(ncols=2, gridspec_kw=grid_kws)
    sns.heatmap(
        mod_df,
        cmap="PRGn",
        center=0,
        vmin=np.floor(mod_df.min().min()*100)/100,
        vmax=np.floor(mod_df.max().max()*100)/100,
        ax=ax,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "vertical"},
        linewidth=0.1,
        linecolor="white",
        annot=mod_pval_df,
        fmt='s',
        xticklabels=True,
    )

    # create label
    temp_df = lf_prs_plot_df.drop_duplicates("Polygenic Score (PGS) ID").copy()
    pgsid_to_name_dict = dict(zip(temp_df["Polygenic Score (PGS) ID"], temp_df["short_label"]))
    new_xlabs = [pgsid_to_name_dict[x.get_text()] if x.get_text().startswith("PG") else "BLANK" for x in ax.get_xticklabels()]


    # cover blank columns
    cover_blank = []
    for i, ix in enumerate(np.where([x == "BLANK" for x in new_xlabs])[0]):
        rect = Rectangle((ix, 0), 1, 9)
        cover_blank.append(rect)

    pc = PatchCollection(cover_blank, facecolor="white", alpha=1, edgecolor=None)
    ax.add_collection(pc)


    ax2_ticks_pos = []
    ax2_ticks_labels = []
    for label in np.unique([x if not x.startswith("BLANK") else "" for x in new_xlabs])[1:]:

        pos_bool = np.where(np.array(new_xlabs) == label)[0]
        ax2_ticks_labels.append(label)

        if len(pos_bool) == 1:
            pos_tick_label = (pos_bool[0] + pos_bool[0] + 1) / 2
            ax2_ticks_pos.append(pos_tick_label)

        elif (len(pos_bool) % 2) != 0:
            pos_tick_label = (pos_bool[0] + pos_bool[-1] + 1) / 2
            ax2_ticks_pos.append(pos_tick_label)

        else:
            pos_tick_label = (pos_bool[0] + pos_bool[-1]) / 2

            if len(pos_bool) % 2 == 0:
                pos_tick_label = pos_tick_label + 0.5

            ax2_ticks_pos.append(pos_tick_label)

        ax.plot([pos_bool[0], pos_bool[-1]+1], [0,0], '-', color='gray', linewidth=0.5)
        ax.plot([pos_bool[0], pos_bool[-1]+1], [8,8], '-', color='gray', linewidth=1, zorder=1)

    # add vertical line at the end of each section
    ix_starts = np.where(np.array(new_xlabs)[:-1] != np.array(new_xlabs)[1:])[0].tolist()
    ix_starts.append(len(new_xlabs))
    x_start = 0
    for count, lab in enumerate(new_xlabs):

        if lab.startswith("BLANK"):
            continue
        if count == (len(new_xlabs) - 1):
            continue
        if lab != new_xlabs[count + 1]:
            ax.plot([count + 1, count + 1], [0, 8], "-", color="gray", linewidth=0.5)
            ax.plot([count + 3, count + 3], [0, 8], "-", color="gray", linewidth=0.5)

    ax.plot([0 , 0], [0, 9], "-", color="gray", linewidth=0.5)
    ax.plot([ax.get_xlim()[-1] , ax.get_xlim()[-1]], [0, 8], "-", color="gray", linewidth=0.5, clip_on=False)


    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax2_ticks_pos)
    ax2.set_xticklabels(ax2_ticks_labels, rotation=45, va="bottom", ha="left")


    sns.despine(ax=ax, top=True, bottom=True, left=True, right=True)
    sns.despine(ax=ax2, top=True, bottom=True, left=True, right=True)

    ax.tick_params(which="major", width=0.4, size=3)
    ax2.tick_params(which="major", width=0.4, size=3)
    cbar_ax.tick_params(which="major", width=0.4, size=3)


    # modify x ticks
    new_xticks = []
    new_xticklabels = []
    for xt in ax.get_xticklabels():

        xtext = xt.get_text()
        xtext_pos = xt.get_position()

        if xtext.startswith("blank"):
            continue

        new_xticks.append(xtext_pos[0])
        new_xticklabels.append(xtext)

    ax.set_xticks(new_xticks)
    ax.set_xticklabels(new_xticklabels, fontsize=8)
    ax.set_yticklabels([x.get_text().replace("V", "F") for x in ax.get_yticklabels()])
    ax.set_ylabel("Latent Factor")
    ax.set_xlabel('Polygenic Risk Score ID')
    ax.tick_params(axis='x', length=0)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.24, wspace=0.05)

    fig.text(0.937, 0.42, 'Effect Size (Beta)', ha='center', rotation='vertical', fontsize=10)

    if savefig:
        plt.savefig(OUTPUT_DIR.joinpath(f"{DATE}_latentfactor_assoc_with_prs_effect_size_heatmap.pdf"), transparent=True)

def load_white_only_covars(COHORT_FILE, COVAR_FILE):
    white_df = pd.read_csv( COHORT_FILE, sep="\t")
    white_grids = white_df['GRID'].unique()

    # load covariates
    covar_df = pd.read_csv( COVAR_FILE, sep="\t")

    # keep only white_grids
    white_covar_df = covar_df.loc[covar_df['GRID'].isin(white_grids)].copy()

    label_colname = 'binary_delivery'
    white_label_df = white_covar_df.loc[:, ['GRID','delivery']].copy()
    white_label_df[label_colname]= (white_label_df['delivery']=='all_preterm').map(int)
    white_ptb_grids = white_label_df.loc[white_label_df['delivery']=='all_preterm', 'GRID'].values

    return white_label_df, white_ptb_grids

def load_pcs(PCA_FILE, n_pcs=15):
    pca_df=pd.read_csv(PCA_FILE, sep="\t")
    pca_df.drop(columns={'IID'}, inplace=True)
    pca_col_names = [f'PC{num}' for num in np.arange(1,n_pcs+1)]

    return pca_df, pca_col_names

def univar_regress_wrapper(prs_covar_lf_df, prs_col_names, lf_cols, pca_col_names, clustermap_sig_lfs=True, scatter_sig_lf=True, exp_label=''):


    univar_lf_prs_df = pd.DataFrame()
    for ind, prs_column in enumerate(prs_col_names):
        print(f"On {ind} of {len(prs_col_names)}")
        analysis_df = prs_covar_lf_df.loc[:, lf_cols + pca_col_names + [prs_column]].copy()

        for this_lf in lf_cols:
            # LF ~ PRS + PCs  (for each LF and PRS)
            lf_univar_prs_df = run_regression(analysis_df, x_colnames=[prs_column], pca_col_names=pca_col_names, y_colname=this_lf, transform_x='std', transform_y='norm')
            lf_univar_prs_df['Trait_ID'] = prs_column
            lf_univar_prs_df['lf']= this_lf
            univar_lf_prs_df = univar_lf_prs_df.append(lf_univar_prs_df)



    lf_prs_plot_df = prep_ptb_assoc_df_for_plot(univar_lf_prs_df, pca_col_names, meta_df, )
    sig_lf_prs_plot_df = lf_prs_plot_df[lf_prs_plot_df['pass_fdr'] == True].copy()

    dep2cat = dict(zip(sig_lf_prs_plot_df['dependent_variable'], sig_lf_prs_plot_df['Trait_Category']))
    wide_sig_lf_prs_plot_df = sig_lf_prs_plot_df.pivot(index='dependent_variable', columns='independent_variable', values='pass_fdr').reset_index()
    wide_sig_lf_prs_plot_df['trait'] = wide_sig_lf_prs_plot_df['dependent_variable'].map(dep2cat)
    wide_sig_lf_prs_plot_df.drop(['dependent_variable'], axis=1, inplace=True)
    wide_sig_lf_prs_plot_df.set_index('trait', inplace=True)
    wide_sig_lf_prs_plot_df.columns.name=""
    wide_sig_lf_prs_plot_df.reset_index(inplace=True)
    wide_sig_lf_prs_plot_df.set_index('trait', inplace=True)


    # plot
    if clustermap_sig_lfs:
        sns.set(style="ticks",  font_scale=1.0, rc={"figure.figsize": (8, 8)})
        sns.clustermap((wide_sig_lf_prs_plot_df*1).fillna(0))
        plt.savefig(OUTPUT_DIR.joinpath(f'{DATE}_clustermap_fdr-sig_{exp_label}_lf_with_prs_univariate.png'))


    # plot scatter
    if scatter_sig_lf:
        n_plots = sig_lf_prs_plot_df.shape[0]
        nrows= 4
        ncols = int(np.ceil(n_plots/nrows))

        sns.set(style="ticks",  font_scale=1.0, rc={"figure.figsize": (14, 10)})
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
        axs = axs.ravel()
        prs_covar_lf_df
        for ind, (_, row) in enumerate(sig_lf_prs_plot_df.iterrows()):

            ax = axs[ind]
            row['dependent_variable']
            row['independent_variable']

            corr_df = prs_covar_lf_df[[row['dependent_variable'], row['independent_variable']]].copy()
            sns.scatterplot(data=corr_df, x=row['dependent_variable'], y=row['independent_variable'],
                            marker='.', alpha=0.5,ax=ax)
            ax.set_title(row['Trait_Category'])
            ax.set_xlabel('PRS')


        plt.tight_layout()
        plt.savefig(OUTPUT_DIR.joinpath(f'{DATE}_scatter_fdr-sig_{exp_label}_lf_with_prs_univariate.png'))



    return sig_lf_prs_plot_df, univar_lf_prs_df


# %%
# -----------
# main
# -----------

###
###    LOAD DATA
###

# tensor decomp and labels
NUM_LATENT_FACTORS = 33   # 18 was the best
input_tensor = pickle.load(open(tensor_file, 'rb'))
phe_labels, grid_labels, binned_years_labels = load_factor_matrix(NUM_LATENT_FACTORS, this_constraint)


# load delivery information
delivery_df = pd.read_csv(delivery_file, sep="\t")
ptb_grids = delivery_df.loc[delivery_df['consensus_label'] == 'preterm', 'GRID'].unique()
term_grids = delivery_df.loc[delivery_df['consensus_label'] == 'term', 'GRID'].unique()

# load latent factor (individ x lf) weight matrix
individ_lf, lf_cols = load_indivd_lf(NUM_LATENT_FACTORS, this_constraint, grid_labels, ptb_grids)

# load prs data
all_prs_df, meta_df, prs_col_names = load_all_prs_data()


# load PCS
pca_df, pca_col_names = load_pcs(PCA_FILE, n_pcs=15)

# three way merge
temp_merge_lf_df = pd.merge(all_prs_df, pca_df, on='FID', how='inner')
prs_covar_lf_df = pd.merge(temp_merge_lf_df, individ_lf, left_on='FID', right_on='GRID', how='inner')
prs_covar_lf_df.drop(columns={'GRID'}, inplace=True)
# remove PGS000116  -> opposite direction to the other PRSs


# %%
###
###    ANALYSIS
###
DATE
cache_file = OUTPUT_DIR.joinpath(f'2022-05-05_all_lf_assoc_with_delivery_type_df.pickle')
# if cache_file.exists():
#     print('loading from cache ... ')
#     plot_assoc_df = pickle.load(open(cache_file, 'rb'))
    
# else:
# LF ~ PRS + PCs  (for each LF and PRS)
sig_lf_prs_plot_df, univar_lf_prs_df = univar_regress_wrapper(prs_covar_lf_df, prs_col_names, lf_cols, pca_col_names, clustermap_sig_lfs=True, scatter_sig_lf=True, exp_label='')

# stratify by preterm vs not preterm
preterm_prs_covar_lf_df = prs_covar_lf_df[prs_covar_lf_df['preg_type']=='preterm'].copy()
not_preterm_prs_covar_lf_df = prs_covar_lf_df[prs_covar_lf_df['preg_type']=='not-preterm'].copy()

# regression
pre_sig_lf_prs_plot_df, pre_univar_lf_prs_df = univar_regress_wrapper(preterm_prs_covar_lf_df, prs_col_names, lf_cols, pca_col_names, clustermap_sig_lfs=True, scatter_sig_lf=True, exp_label='preterm')
nopre_sig_lf_prs_plot_df, nopre_univar_lf_prs_df = univar_regress_wrapper(not_preterm_prs_covar_lf_df, prs_col_names, lf_cols, pca_col_names, clustermap_sig_lfs=True, scatter_sig_lf=True, exp_label='not-preterm')

# create new trait label
pre_sig_lf_prs_plot_df.loc[pre_sig_lf_prs_plot_df['Specific_Trait_Category'].isna(), 'Specific_Trait_Category'] = pre_sig_lf_prs_plot_df.loc[pre_sig_lf_prs_plot_df['Specific_Trait_Category'].isna(), 'Trait_Category']
nopre_sig_lf_prs_plot_df.loc[nopre_sig_lf_prs_plot_df['Specific_Trait_Category'].isna(), 'Specific_Trait_Category'] = nopre_sig_lf_prs_plot_df.loc[nopre_sig_lf_prs_plot_df['Specific_Trait_Category'].isna(), 'Trait_Category']

# %%
### format data

# all preterm + term
all_lf_prs_plot_df = prep_ptb_assoc_df_for_plot(univar_lf_prs_df, pca_col_names, meta_df, )

# preterm birth results
ptb_lf_prs_plot_df = prep_ptb_assoc_df_for_plot(pre_univar_lf_prs_df, pca_col_names, meta_df, )

# not-preterm birth results
no_ptb_lf_prs_plot_df = prep_ptb_assoc_df_for_plot(nopre_univar_lf_prs_df, pca_col_names, meta_df, )

all_lf_prs_plot_df.to_csv(OUTPUT_DIR.joinpath('all_lf_prs_plot_df.tsv'), sep="\t", index=False)
ptb_lf_prs_plot_df.to_csv(OUTPUT_DIR.joinpath('ptb_lf_prs_plot_df.tsv'), sep="\t", index=False)
no_ptb_lf_prs_plot_df.to_csv(OUTPUT_DIR.joinpath('no_ptb_lf_prs_plot_df.tsv'), sep="\t", index=False)




temp_all_df = all_lf_prs_plot_df.loc[all_lf_prs_plot_df['pass_fdr']==True, ['Trait_ID','Specific_Trait_Category', 'lf', 'coef', 'std_err', 'p-value']].copy()
temp_ptb_df = ptb_lf_prs_plot_df.loc[ptb_lf_prs_plot_df['pass_fdr']==True, ['Trait_ID','Specific_Trait_Category', 'lf', 'coef', 'std_err', 'p-value']].copy()
temp_no_ptb_df = no_ptb_lf_prs_plot_df.loc[no_ptb_lf_prs_plot_df['pass_fdr']==True, ['Trait_ID','Specific_Trait_Category', 'lf', 'coef', 'std_err', 'p-value']].copy()
temp_all_df['analysis'] = "all"
temp_ptb_df['analysis'] = "ptb"
temp_no_ptb_df['analysis'] = "no_ptb"

plot_assoc_df = pd.concat([temp_all_df, temp_ptb_df, temp_no_ptb_df], axis=0)


# pickle.dump(plot_assoc_df, open(cache_file, 'wb'))


# %%
### heatmaps, plot all but show significant hits
ptb_lf_prs_plot_df.head(1)



sns.heatmap(ptb_lf_prs_plot_df.pivot(index='Trait_ID', columns='lf', values='coef').T)







# %%
### PLOT only significant hits

all_sig = sig_lf_prs_plot_df.loc[:, ['lf','Specific_Trait_Category']].copy()
all_sig['delivery'] = 'all'
all_sig['Specific_Trait_Category'] = all_sig['Specific_Trait_Category'].fillna('None')


pre_sig = pre_sig_lf_prs_plot_df.loc[:, ['lf','Specific_Trait_Category']].copy()
pre_sig['delivery'] = 'preterm'
nonpre_sig = nopre_sig_lf_prs_plot_df.loc[:, ['lf','Specific_Trait_Category']].copy()
nonpre_sig['delivery'] = 'not-pretem'


sns.set(style="ticks",  font_scale=1.0, rc={"figure.figsize": (4, 4)})

fig, ax = plt.subplots(nrows=1, ncols=1)

# ax.plot(all_sig['lf'], all_sig['Specific_Trait_Category'], '.', markersize=10, label='all')
ax.plot(nonpre_sig['lf'], nonpre_sig['Specific_Trait_Category'], 'o', markersize=20, markerfacecolor='white', label='not-preterm')
ax.plot(pre_sig['lf'], pre_sig['Specific_Trait_Category'], 'x', markersize=20, label='preterm')
ax.set_ylabel('Polygenic Risk Score')
ax.set_xlabel('Latent Topic')

ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)


# plt.savefig(OUTPUT_DIR.joinpath(f"{DATE}_LF_assoc_prs.pdf"))
# %%

# # convert wide df
# wide_preterm_df = lf_prs_plot_df.loc[~lf_prs_plot_df['topic'].isin(pca_col_names), ['independent_variable','dependent_variable', 'fdr_bh_adjusted_pvalue']].pivot(index='independent_variable', columns='dependent_variable', values='fdr_bh_adjusted_pvalue')
# wide_nopreterm_df = lf_prs_plot_df.loc[~lf_prs_plot_df['topic'].isin(pca_col_names), ['independent_variable','dependent_variable', 'fdr_bh_adjusted_pvalue']].pivot(index='independent_variable', columns='dependent_variable', values='fdr_bh_adjusted_pvalue')
#
# sns.heatmap((wide_preterm_df < 0.05))
# sns.heatmap((wide_nopreterm_df < 0.05))
