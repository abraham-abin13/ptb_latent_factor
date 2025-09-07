#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on: 2021-03-09 15:40:08

# %% PATHS
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# stats
from scipy.stats import f_oneway
import statsmodels.api as sm
import statsmodels.stats.multitest as smm
import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler

sys.path.append("/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/r_tensor_decomp")
from helper_analyze_ranks import latentAnalysis, latentWeights

# plotting
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
%matplotlib inline
%config InlineBackend.figure_format='retina'

import matplotlib as mpl
import matplotlib.font_manager as font_manager
fpath='/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf'
font_dirs = ['/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)
mpl.rcParams['font.family'] = 'Arial'

DATE = datetime.now().strftime('%Y-%m-%d')


### PATHS
ROOT_DIR = Path("/dors/capra_lab/users/abraha1/projects/PTB_phewas/")
PRS_DIR = ROOT_DIR.joinpath("data/2020_07_30_longit_topic_modeling/polygenic_risk_scores/plink_calculated_prs/")
EGG_CONSORTIA_DIR = ROOT_DIR.joinpath("data/2020_07_30_longit_topic_modeling/polygenic_risk_scores/egg_consortia")
EGG_CONSORTIA_FILE = EGG_CONSORTIA_DIR.joinpath('egg_consortia.xlsx')
EGG_PRS_DIR = EGG_CONSORTIA_DIR.joinpath("plink_calculated_prs")
TRAIT_DETAILS_FILE = ROOT_DIR.joinpath("data/2020_07_30_longit_topic_modeling/polygenic_risk_scores/pgs_scores_data.xlsx")
PCA_FILE = "/dors/capra_lab/users/abraha1/data/biovu_mega_ex_2019_02_capra_preterm_a3_v1-1/MEGAex_BioVUthru2019-02_BestOfMultipleCalls/mich_imp_server_results/merged/15pc_race_white_merged_filt_imp_v1.eigenvec"
COHORT_FILE = ROOT_DIR.joinpath("scripts/2020_07_30_longit_topic_modeling/manuscript/phecode_assoc_w_ptb/icd9_assoc_only/white_cohort_for_ptb_assoc.tsv")
COVAR_FILE = ROOT_DIR.joinpath("scripts/2020_07_30_longit_topic_modeling/manuscript/phecode_assoc_w_ptb/icd9_assoc_only/covar_for_ptb_assoc.tsv")


OUTPUT_DIR=ROOT_DIR.joinpath("scripts/2020_07_30_longit_topic_modeling/manuscript/latent_factors_ptb_term/prs_assoc_case_control")



# %% FUNCTIONS 
def load_prs(PRS_DIR):
    all_prs_df = pd.DataFrame()
    for counter, prs_file in enumerate(PRS_DIR.glob("*.profile")):
        print(prs_file.name)
        df = pd.read_csv( prs_file, sep="\s+", engine='python')
        keep_df = df.loc[:, ['FID','SCORESUM']].copy()
        keep_df.rename(columns={'SCORESUM':prs_file.name.split('.profile')[0]}, inplace=True)

        if counter == 0:
            all_prs_df = keep_df.copy()
            continue

        all_prs_df = pd.merge(all_prs_df, keep_df, on='FID', how='outer')
    return all_prs_df

def load_prs_egg_consortia(EGG_PRS_DIR):
    all_prs_df = pd.DataFrame()
    for counter, prs_file in enumerate(EGG_PRS_DIR.glob("*.profile")):

        prs_name = prs_file.parts[-1].split("_concat.tsv")[0]
        prs_df = pd.read_csv( prs_file, sep="\s+")
        prs_df.rename(columns={'SCORESUM':prs_name}, inplace=True)
        keep_df = prs_df.loc[:, ['FID',prs_name]].copy()

        if counter ==0:
            all_prs_df = keep_df.copy()
            continue

        all_prs_df = pd.merge(all_prs_df, keep_df, on="FID",how="outer")

    return all_prs_df

def run_regression(analysis_df, x_colnames, pca_col_names, y_colname, transform_x=False, transform_y=False, logit=False):
    # lf_univar_prs_df = run_regression(analysis_df, x_colnames=[prs_column], pca_col_names=pca_col_names, y_colname=this_lf, std_x=True, std_y=True)
    # analysis_df, x_colnames=[prs_column], pca_col_names=pca_col_names, y_colname=this_lf, transform_x='std', transform_y='norm', logit=False


    # pca_col_names -> no standardization is applied
    assert len(y_colname) >1, "y_colname has more than one item"

    data_df = analysis_df.copy()
    standardize = lambda df, col_name: (df[col_name] - df[col_name].mean())/df[col_name].std()

    # independent variables (X)
    updated_x_colnames = []
    if transform_x == "std":
        print("standardizing dependent variables (X).")
        for x_col in x_colnames:
            data_df[f'std_{x_col}'] = standardize(data_df, x_col)
            updated_x_colnames.append(f'std_{x_col}')
    elif transform_x == "norm":
        print("normalizing dependent variables (X).")
        for x_col in x_colnames:
            scaler = MinMaxScaler()
            data_df[f'std_{x_col}'] = scaler.fit_transform(data_df.loc[:, [x_col]])
            updated_x_colnames.append(f'std_{x_col}')
    else:
        updated_x_colnames = x_colnames

    # dependent variable (Y)
    if transform_y =="std":
        print("standardizing independent variables (Y).")
        updated_y_colname = f'std_{y_colname}'
        data_df[updated_y_colname] = standardize(data_df, y_colname)
    elif transform_y == "norm":
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
    lr_df['prs_label'] = prs_column
    lr_df['model_pval'] =model_pval
    lr_df['nobs'] = nobs
    lr_df['independent_variable'] = y_colname
    lr_df['dependent_variable'] = ','.join(x_colnames)
    if logit:
        lr_df['odds_ratio'] = lr_df['coef'].apply(lambda x: np.exp(x))
        lr_df['or_CI_0.975'] = lr_df['CI_0.975'].apply(lambda x: np.exp(x))
        lr_df['or_CI_0.025'] = lr_df['CI_0.025'].apply(lambda x: np.exp(x))


    return lr_df

def prs_delivery_regress(analysis_df,  pca_col_names, y_colname, std_y=True):
    # pca_col_names -> no standardization is applied


    data_df = analysis_df.copy()
    standardize = lambda df, col_name: (df[col_name] - df[col_name].mean())/df[col_name].std()

    if std_y:
        updated_y_colname = f'std_{y_colname}'
        data_df[updated_y_colname] = standardize(data_df, y_colname)
    else:
        updated_y_colname = y_colname


    # set up indpeendent (X) and dependent (y) variables
    endo_x = data_df.loc[:, ['delivery'] + pca_col_names].values
    y=data_df[updated_y_colname]


    fml = f'{y_colname} ~ C(delivery) + 1 + ' + ' + '.join(pca_col_names)
    results = smf.ols(formula=fml, data=analysis_df).fit()
    model_pval = results.f_pvalue
    nobs = results.nobs

    # parse results
    lr_df = pd.read_html(results.summary(xname=['const']+['delivery']+pca_col_names).tables[1].as_html(), header=0,index_col=0)[0]
    lr_df.columns = ['coef','std_err','t','p-value', 'CI_0.025', 'CI_0.975']
    lr_df = lr_df.drop('const').reset_index()
    lr_df.rename({'index':'topic'},inplace=True, axis=1)
    lr_df['lower_err'] = np.array(lr_df['coef']  - lr_df['CI_0.025'])
    lr_df['upper_err'] = np.array(lr_df['CI_0.975']  - lr_df['coef'])
    lr_df['prs_label'] = prs_column
    lr_df['model_pval'] =model_pval
    lr_df['nobs'] = nobs

    return lr_df

def prep_for_heatmap(data_df,trait_details_df, dont_pivot=False):

    effect_label_df = pd.merge(data_df, trait_details_df.loc[:, ['Polygenic Score (PGS) ID', 'Mapped Trait(s) (Ontology)', 'big_category']], left_on='prs_label', right_on='Polygenic Score (PGS) ID', how='left')
    # modify names
    effect_label_df.loc[effect_label_df['Mapped Trait(s) (Ontology)']=='diastolic blood pressure', 'Mapped Trait(s) (Ontology)']  = 'blood pressure diastolic'
    effect_label_df.loc[effect_label_df['Mapped Trait(s) (Ontology)']=='systolic blood pressure', 'Mapped Trait(s) (Ontology)']  = 'blood pressure systolic'

    topic_no_pcs_names = set(effect_label_df['topic'].unique()).difference(pca_col_names)
    print(f"FDR adjusting for: {topic_no_pcs_names}")

    effect_label_df = effect_label_df.loc[effect_label_df['topic'].isin(topic_no_pcs_names)]


    # fdr adjust (alpha=0.05) within each latent factor ***(correct for ~64 traits)***
    effect_label_df['fdr_bh_adjusted_pvalue']=np.nan
    for lf in topic_no_pcs_names:

        fdr_bool,pval_corr, _, _ = smm.multipletests( effect_label_df.loc[effect_label_df['topic']==lf, 'p-value'],  method='fdr_bh')
        effect_label_df.loc[effect_label_df['topic']==lf, 'fdr_bh_adjusted_pvalue'] = pval_corr

    effect_label_df['id_label'] =  effect_label_df['Mapped Trait(s) (Ontology)'] + "_" + effect_label_df['prs_label']
    effect_label_df.sort_values(['id_label','fdr_bh_adjusted_pvalue'],inplace=True)

    if dont_pivot:
        return effect_label_df

    wide_df = effect_label_df.loc[effect_label_df['topic'].apply(lambda x: x.startswith("s"))].pivot(index='topic', columns="id_label", values='coef')
    pval_df = effect_label_df.loc[effect_label_df['topic'].apply(lambda x: x.startswith("s"))].pivot(index='topic', columns="id_label", values='fdr_bh_adjusted_pvalue')
    pval_df = pval_df.applymap(lambda x: '*' if x < 0.05 else '')



    return wide_df, pval_df

def prep_ptb_assoc_df_for_plot(prs_df, pca_col_names, trait_details_df, ):

    prs_df=all_ptb_prs_df
    trait_details_df=all_prs_details_df

    # remove pc columns
    prs_names = set(prs_df['topic'].unique()).difference(pca_col_names)
    no_pcs_df =  prs_df.loc[prs_df['topic'].isin(prs_names),:].copy()

    # merge with trait details
    effect_label_df = pd.merge(no_pcs_df, trait_details_df.loc[:, ['Trait_ID', "Trait", 'Trait_Category']], left_on='prs_label', right_on='Trait_ID', how='left')


    fdr_bool,pval_corr, _, _ = smm.multipletests(    effect_label_df.loc[:, 'p-value'],  method='fdr_bh')
    effect_label_df.loc[:, 'fdr_bh_adjusted_pvalue'] = pval_corr
    effect_label_df['pass_fdr'] = effect_label_df['fdr_bh_adjusted_pvalue']<0.05

    effect_label_df.sort_values(['Trait_ID', 'pass_fdr'],inplace=True)
    return effect_label_df

def map_traits_to_dict():
    return  {
        "coronary artery disease": "heart disease",
        "cardiovascular disease": "heart disease",
        "type II diabetes mellitus": "diabetes",
        "type I diabetes mellitus": "diabetes",
        "body mass index": "BMI",
        "high density lipoprotein cholesterol measurement": "heart disease",
        "low density lipoprotein cholesterol measurement": "heart disease",
        "triglyceride measurement": "heart disease",
        "bone density": "bone density",
        "HbA1c measurement": "diabetes",
        "schizophrenia": "mental health",
        "major depressive disorder": "mental health",
        "major depressive disorder\nrecurrent": "mental health",
        "self-reported trait\ndepressive symptom measurement": "mental health",
        "high density lipoprotein cholesterol measurement\nlow density lipoprotein cholesterol measurement": "heart disease",
        "BMI-adjusted waist-hip ratio": "BMI",
        "diastolic blood pressure": "heart disease",
        "systolic blood pressure": "heart disease"}

# %%
###
###    load prs and pcs
###

### -> load meta data for prs
trait_details_dict = pd.read_excel(TRAIT_DETAILS_FILE, sheet_name=None)
trait_details_df = trait_details_dict["selected_prs"]
mapped_traits_to_disease_dict = map_traits_to_dict()
trait_details_df['big_category'] = trait_details_df['Mapped Trait(s) (Ontology)'].map(mapped_traits_to_disease_dict)


short_label = {'BMI-adjusted waist-hip ratio': 'BMI-adjusted',
                'HbA1c measurement': 'A1c',
                'blood pressure diastolic': 'BP-diastolic',
                'blood pressure systolic': 'BP-systolic',
                'body mass index': 'BMI',
                'bone density': 'BMD',
                'cardiovascular disease': 'CVD',
                'coronary artery disease': 'CAD',
                'high density lipoprotein cholesterol measurement\nlow density lipoprotein cholesterol measurement': "HDL/LDL",
                'high density lipoprotein cholesterol measurement': 'HDL',
                'low density lipoprotein cholesterol measurement': 'LDL',
                'major depressive disorder\nrecurrent': "Depression",
                'major depressive disorder': "Depression",
                'schizophrenia': 'Schizophrenia',
                'self-reported trait\ndepressive symptom measurement': 'Depression',
                'triglyceride measurement': 'Triglycerides',
                'type I diabetes mellitus': 'T1DM',
                'type II diabetes mellitus': 'T2DM'
}
trait_details_df['short_label'] =  trait_details_df['Mapped Trait(s) (Ontology)'].map(short_label)
trait_details_df.sort_values(['big_category', 'Mapped Trait(s) (Ontology)'], inplace=True)


# load egg consortia
egg_consort_dict = pd.read_excel(EGG_CONSORTIA_FILE, sheet_name=None)
egg_trait_details_df = egg_consort_dict["Sheet1"]
egg_trait_details_df['trait_id'] = egg_trait_details_df['Filename'].apply(lambda x: x.split('.txt.gz')[0])
egg_trait_details_df['category'] = egg_trait_details_df['Short_name'].apply(lambda x: x.split("(")[0])
egg_trait_details_df.head(10)

# combine egg and nonegg consortia
all_prs_details_df  = pd.concat([egg_trait_details_df.loc[:, ['Short_name','trait_id', 'category']].rename(columns={'Short_name':'Trait', 'trait_id':'Trait_ID', 'category':"Trait_Category"}),
        trait_details_df.loc[:, ['Polygenic Score (PGS) ID', 'Reported Trait', 'big_category']].rename(columns={'Polygenic Score (PGS) ID':'Trait_ID', 'Reported Trait':"Trait", 'big_category':'Trait_Category'})],
        axis=0)

prs_id_to_name = dict(zip(all_prs_details_df['Trait_ID'], all_prs_details_df['Trait']))
prs_id_to_category = dict(zip(all_prs_details_df['Trait_ID'], all_prs_details_df['Trait']))

# -> load prs files
all_prs_df = load_prs(PRS_DIR)
all_egg_prs_df = load_prs_egg_consortia(EGG_PRS_DIR)
combined_prs_df =pd.merge(all_prs_df, all_egg_prs_df, on='FID', how='outer')
# remove outliers 
outliers_to_remove= ["PGS000117", "PGS000116"]
combined_prs_df.drop(columns=outliers_to_remove, inplace=True)
prs_col_names = combined_prs_df.columns.difference(['FID']).values.tolist()





###
###    load PCs
###
pca_df=pd.read_csv(PCA_FILE, sep="\t")
pca_df.drop(columns={'IID'}, inplace=True)
pca_col_names = [f'PC{num}' for num in np.arange(1,16)]


# %%
###
###    load white cohort
###

white_df = pd.read_csv( COHORT_FILE, sep="\t")
covar_df = pd.read_csv( COVAR_FILE, sep="\t")

label_colname = 'binary_delivery'
label_df = covar_df.loc[:, ['GRID','delivery']].copy()
label_df[label_colname]= (label_df['delivery']=='all_preterm').map(int)


# %% assoc analysis
# -----------
# association analyses with PTB/noPTB
# -----------
# three way merge
temp_merge_df = pd.merge(combined_prs_df, pca_df, on='FID', how='inner')
prs_covar_delivery_df = pd.merge(temp_merge_df, label_df, left_on='FID', right_on='GRID', how='inner')
prs_covar_delivery_df.drop(columns={'GRID'}, inplace=True)

any(prs_covar_delivery_df.columns.isin(outliers_to_remove))


# PTB/No-PTB ~ PRS (for each PRS )
all_ptb_prs_df = pd.DataFrame()
for ind, prs_column in enumerate(prs_col_names):
    print(f"On {ind} of {len(prs_col_names)}")
    analysis_df = prs_covar_delivery_df.loc[:, [label_colname] + pca_col_names + [prs_column]].copy()
    analysis_prs_ptb_df = prs_covar_delivery_df.loc[:, ['delivery'] + pca_col_names + [prs_column]].copy()

    # PTB/noPTB ~ PRS
    ptb_prs_df = run_regression(analysis_df, x_colnames=[prs_column], pca_col_names=pca_col_names, y_colname=label_colname, transform_x='std', transform_y=False, logit=True)
    all_ptb_prs_df = all_ptb_prs_df.append(ptb_prs_df)


ptb_prs_plot_df = prep_ptb_assoc_df_for_plot(all_ptb_prs_df, pca_col_names, all_prs_details_df, )
ptb_prs_plot_df['topic']=  ptb_prs_plot_df['topic'].apply(lambda x: x.split("std_")[-1])


# %%

# prepare for plotting
ptb_plot_df = ptb_prs_plot_df.loc[:, ['topic', 'coef','odds_ratio','Trait', 'Trait_ID', 'pass_fdr','or_CI_0.025', 'or_CI_0.975']].copy()
trait2short_label_dict = {'Head circumference': "Head circumference",
    'Birthweight (Fetal effect, 2019, European)': "Birthweight-Fetal",
    'Birthweight (Fetal effect, 2019, Trans-ancestry)': "Birthweight-Fetal",
    'Own Birthweight (Fetal effect, European)': "Birthweight-Fetal",
    'Early preterm birth (Fetal effect)': "Early preterm-Fetal",
    'Gestational duration (Fetal effect)': "Gestational duration",
    'Postterm birth (Fetal effect)': "Postterm birth-Fetal",
    'Preterm birth (Fetal effect)': "Preterm birth-Fetal",
    'Birthweight (Maternal effect, 2018)': "Birthweight-Maternal",
    'Birthweight (Maternal effect, 2019, European)': "Birthweight-Maternal",
    'Birthweight (Maternal effect, 2019, Trans-ancestry)': "Birthweight-Maternal",
    'Own Birthweight (Maternal effect, European)': "Birthweight-Maternal",
    'Coronary heart disease': "CVD",
    'Coronary artery disease': "CVD",
    'Type 2 diabetes': "T2DM",
    'Type 1 diabetes': "T1DM",
    'Body Mass Index': "BMI",
    'Type 2 diabetes (based on SNPs involved in β-cell function)': "T2DM",
    'Type 2 diabetes (based on SNPs involved in insulin resistance)': "T1DM",
    'Adult Body Mass Index': "BMI",
    'high-density lipoprotein (HDL) cholesterol':"HDL",
    'low-density lipoprotein (LDL) cholesterol': "LDL",
    'triglycerides': "TGs",
    'High density lipoprotein (HDL) cholesterol': "HDL",
    'Low density lipoprotein (HDL) cholesterol': "LDL",
    'Triglycerides (TG)': "TGs",
    'low density lipoprotein cholesterol': "LDL",
    'Coronary Artery Disease': "CVD",
    'Cardiovascular Disease': "CVD",
    'Bone mineral density (BMD)': "BMD",
    'Type 2 Diabetes': "T2DM",
    'Hemoglobin A1c': "A1C",
    'Schizophrenia': 'Schizophrenia',
    'Lifetime Major Depressive Disorder': "Depression",
    'Lifetime Major Depressive Disorder (with recurrence)': "Depression",
    'Self-reported depression or depression symptoms': "Depression",
    'Cholesterol': "Cholesterol",
    'Major depression': "Depression",
    'Body mass index': "BMI",
    'Waist-to-hip ratio (body mass index adjusted)': "WHR",
    'Systolic blood pressure': "SBP",
    'Diastolic blood pressure': "DBP",
    'HbA1c': "A1C",
    'High-density lipoprotein':"HDL",
    'Low-density lipoprotein': "LDL",
    'Triglycerides': "TGs",
    'Low-density lipoprotein cholesterol levels': "LDL"}

ptb_plot_df['Trait Label'] = ptb_plot_df['Trait'].map(trait2short_label_dict)
ptb_plot_df.sort_values('Trait Label', ascending=True, inplace=True)

ptb_plot_df = ptb_plot_df[ptb_plot_df['topic'].str.startswith("PGS")]


# %%
#
# ptb_plot_df['Trait'].nunique()
# ptb_plot_df.head()
# #
# # %%

# plotting grouping by trait

n_groups = ptb_plot_df['Trait Label'].nunique()
frows=1
fcols=int(np.ceil(n_groups/frows))
fig, axs = plt.subplots(nrows=frows, ncols=fcols, figsize=(15,4), sharey=True)
axs = axs.ravel()


# for enum, trait_label in ptb_plot_df['Trait Label'].unique():
for en, (trait_label, df) in enumerate(ptb_plot_df.groupby(['Trait Label'])):
    ax = axs[en]

    ax.scatter(x=df['topic'], y=df['odds_ratio'], color='gray')
    ax.scatter(x=df.loc[df['pass_fdr']==True, 'topic'], y=df.loc[df['pass_fdr']==True, 'odds_ratio'], color='indianred')
    sns.despine(ax=ax, top=True, right=True, left=True,trim=False)



    for _, row in df.iterrows():
        if row['pass_fdr'] == True:
            col = 'indianred'
        else:
            col = 'gray'
        ax.plot([row['topic'],row['topic']], [row['or_CI_0.025'], row['or_CI_0.975']], '-', color=col, zorder=-1, clip_on=False)

    ax.set_title(trait_label, fontsize='small')
    ax.set_xticks(df['topic'])

    xlables = [x[:10] for x in df['topic']]

    ax.set_xticklabels(xlables, rotation=90, fontsize='small')
    ax.axhline(1, color='black', linewidth=0.5, linestyle="--")
    ax.tick_params(axis='y', which='major', length=0)

plt.tight_layout()
# plt.savefig(OUTPUT_DIR.joinpath(f"{DATE}_ptb_assoc_with_prs_effect_size_grouped.pdf"), transparent=True)


# %%

# plot so each trait category plot is propr to the number of traits
#
# 

# manual order 

manual_order =['A1C', 'T1DM', 'T2DM',  'CVD', 'TGs', 'Cholesterol', 'HDL', 'LDL', 'BMI', 'WHR','SBP', 'DBP',   'Depression',  'Schizophrenia', 'BMD']

ptb_plot_df['Trait Label'] = pd.Categorical(ptb_plot_df['Trait Label'], categories=manual_order, ordered=True)
ptb_plot_df.sort_values(['Trait Label', 'coef'], ascending=True, inplace=True)

# Calculate the number of topics for each 'Trait Label'
group_widths = ptb_plot_df.groupby('Trait Label')['topic'].nunique().values
total_width = sum(group_widths)
relative_widths = group_widths / total_width  # Normalize to make them proportional

# Adjust figure size
fig_width = 18  # Total figure width
fig_height = 5
fig = plt.figure(figsize=(fig_width, fig_height))

# Create gridspec with proportional widths
from matplotlib.gridspec import GridSpec
gs = GridSpec(1, len(group_widths), width_ratios=relative_widths)

# Determine shared y-axis limits
y_min = ptb_plot_df['odds_ratio'].min() - 0.2  # Add a little padding
y_max = ptb_plot_df['odds_ratio'].max() + 0.2

# Create subplots
for en, (trait_label, df) in enumerate(ptb_plot_df.groupby(['Trait Label'])):
    ax = fig.add_subplot(gs[en])

    # Scatter plots
    ax.scatter(x=df['topic'], y=df['odds_ratio'], color='gray')
    ax.scatter(x=df.loc[df['pass_fdr'] == True, 'topic'], 
               y=df.loc[df['pass_fdr'] == True, 'odds_ratio'], 
               color='indianred')
    

    # Error bars
    for _, row in df.iterrows():
        col = 'indianred' if row['pass_fdr'] else 'gray'
        ax.plot([row['topic'], row['topic']], [row['or_CI_0.025'], row['or_CI_0.975']], '-', color=col, zorder=-1, clip_on=False)

    # Formatting
    # ax.set_title(trait_label, fontsize='small')
    ax.set_xticks(df['topic'])
    xlables = [x[:10] for x in df['topic']]
    ax.set_xticklabels(xlables, rotation=90, fontsize='small')
    ax.axhline(1, color='black', linewidth=0.5, linestyle="--")

    # add a text label with trait 
    x_min, x_max = ax.get_xlim()
    ax.text((x_min+x_max)/2 , y_min+0.025, f'{trait_label}', ha='center', va='center')


    # Set shared y-axis limits
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min-0.05, x_max+0.05)
    # Remove y-axis labels for all but the first plot
    if en != 0:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', left=False)
        sns.despine(ax=ax, top=True, right=True, left=True, trim=False)
    else:
        ax.set_ylabel('Odds Ratio')  # Add y-axis label only for the first plot
        sns.despine(ax=ax, top=True, right=True, left=False, trim=False)

plt.tight_layout()

# plt.savefig(OUTPUT_DIR.joinpath(f"{DATE}_ptb_assco_prs_grouped_for_manu_wide.pdf"), transparent=True, bbox_inches='tight')
plt.show()


# %%
# save this df 
ptb_plot_df.to_csv(OUTPUT_DIR.joinpath(f"{DATE}_ptb_assoc_with_prs_effect_size_grouped_for_plot.csv"), index=False)
