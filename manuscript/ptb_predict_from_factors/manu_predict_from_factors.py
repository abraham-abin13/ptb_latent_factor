#!/bin/python
# This script will will use lf factor weights ot predict preterm birth. No delivery type codes are used in the latent factor generation.
#
#
#
# Abin Abraham
# created on: 'now'


import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import pickle

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.metrics import PrecisionRecallDisplay
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib as mpl
import matplotlib.font_manager as font_manager
fpath='/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf'
font_dirs = ['/dors/capra_lab/users/abraha1/conda/envs/py36_r_ml/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf', ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)
mpl.rcParams['font.family'] = 'Arial'
bprop = font_manager.FontProperties(fname=fpath, size=10)

%matplotlib inline

DATE = datetime.now().strftime('%Y-%m-%d')


%load_ext autoreload
%autoreload 2
sys.path.append("/dors/capra_lab/users/abraha1/prelim_studies/2020_05_urinary_stones/scripts/run_models")
from hyperopt_xgb_func import run_xgb, run_xgb_mult

sys.path.append("/dors/capra_lab/users/abraha1/prelim_studies/2020_05_urinary_stones/scripts")
from helper_make_feat_func import clean_raw_ml_data, create_held_out, categorize_ua24h

sys.path.append("/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/manuscript/latent_factors_ptb_term")
from helper_plot_func import get_auroc_coords, get_pr_coord, plot_roc, plot_pr

sys.path.append('/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification')
from train_test_rf import  compute_metrics


# %%
# -----------
# Paths
# -----------
DIR="/dors/capra_lab/users/abraha1/projects/PTB_phewas/data/2020_07_30_longit_topic_modeling/r_parafac_outputs/preterm_term_all_race_icd9_no_preg"

FIG_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phewas/data/2020_07_30_longit_topic_modeling/r_parafac_outputs/preterm_term_all_race_icd9_no_preg/figures"
this_label="preterm_term_all_race_icd9_no_preg"
this_constraint="phe_ortho"


tensor_file ="/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/input_tensors/preterm_term_all_race_icd9_no_preg/tensor_preterm_term_all_race_icd9_no_preg_within_5yr.pickle"
phecode_axis_file = "/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/input_tensors/preterm_term_all_race_icd9_no_preg/phecode_axis_preterm_term_all_race_icd9_no_preg_within_5yr.pickle"
grid_axis_file="/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/input_tensors/preterm_term_all_race_icd9_no_preg/grids_axis_preterm_term_all_race_icd9_no_preg_within_5yr.pickle"
binned_years_axis_file="/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/input_tensors/preterm_term_all_race_icd9_no_preg/binned_years_axis_preterm_term_all_race_icd9_no_preg_within_5yr.pickle"

delivery_file = "/dors/capra_lab/users/abraha1/projects/PTB_phewas/data/2020_07_30_longit_topic_modeling/raw_data/est_delivery_date_at_least_one_icd_cpt_ega.tsv"

phecodes_label_file = "/dors/capra_lab/users/abraha1/data/phewas/phecode_descrip.tsv"
demo_file = "/dors/capra_lab/users/abraha1/projects/PTB_phewas/data/2020_07_30_longit_topic_modeling/raw_data/complete_demographics.tsv"


OUTPUT_DIR=Path("/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/manuscript/latent_factors_ptb_term/prediction_panel")

svfig = lambda x: plt.savefig(OUTPUT_DIR.joinpath(f'{DATE}_{x}.pdf'))
# %%
# -----------
# FUNCTIONs
# -----------
def make_performance_df(y_test, preds, proba):

    if (len(np.unique(y_test)) == 2):
        acu = metrics.accuracy_score(y_test, preds)
        bl_acu = metrics.balanced_accuracy_score(y_test, preds, adjusted=True)
        roc_auc_macro = metrics.roc_auc_score(y_test, proba[:,1])
        roc_auc_wt = np.nan


        pr_df = pd.DataFrame(metrics.precision_recall_fscore_support(y_test, preds), index=['Precision','Recall','F1','Count'])
        auc_df = pd.DataFrame({'accu':acu,'bl_accu':bl_acu,'roc_auc_macro':roc_auc_macro,'roc_auc_wt':roc_auc_wt}, index=[0])


    return auc_df, pr_df

# logistic regression
def run_logistic_regression(X_train, y_train, X_test, y_test,):


    lr_clf = LogisticRegression(random_state=0, max_iter=10000,solver='liblinear', class_weight='balanced')
    lr_clf.fit(X_train, y_train)
    lr_pred = lr_clf.predict(X_test)
    lr_proba = lr_clf.predict_proba(X_test)
    lr_auc_df, lr_pr_df = make_performance_df(y_test, lr_pred, lr_proba)
    lr_conf_matrix_df = pd.DataFrame(confusion_matrix(y_test, lr_pred))
    lr_metrics_results = compute_metrics(y_test, lr_pred, lr_proba[:, 1])

    return lr_clf, lr_auc_df, lr_metrics_results, lr_conf_matrix_df

def get_confusion_matrix(model, X_test, y_test ):


    y_pred = model.predict(X_test)


    # confusion matrix on test set
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=('not-preterm', 'preterm'))
    disp.plot(cmap='Blues')

    return cm, disp

# %%
# -----------
# load data
# -----------
NUM_LATENT_FACTORS = 33   # 18 was the best

# load factor matrices
phe_fmatrix = os.path.join(
    DIR,
    f"multiway_rank_{NUM_LATENT_FACTORS}",
    f"rank_{NUM_LATENT_FACTORS}_phecode_factors_{this_constraint}.csv",
)
time_fmatrix = os.path.join(
    DIR,
    f"multiway_rank_{NUM_LATENT_FACTORS}",
    f"rank_{NUM_LATENT_FACTORS}_time_since_delivery_factors_{this_constraint}.csv",
)
individ_fmatrix = os.path.join(
    DIR,
    f"multiway_rank_{NUM_LATENT_FACTORS}",
    f"rank_{NUM_LATENT_FACTORS}_individ_factors_{this_constraint}.csv",
)

# load input tensor axis labels
phe_labels = pickle.load(open(phecode_axis_file, "rb"))
grid_labels = pickle.load(open(grid_axis_file, "rb"))
binned_years_labels = pickle.load(open(binned_years_axis_file, "rb"))

len(phe_labels)
len(grid_labels)


# load tensor with phecodes
tensor  =   pickle.load(open(tensor_file, 'rb'))
tensor.shape

# collapse time dimension by summing
phe_df = pd.DataFrame(np.sum(tensor ,1 ), columns=grid_labels, index=phe_labels).T
phe_df.reset_index(inplace=True)
phe_df.rename(columns={'index':'GRID'}, inplace=True)


# load data
delivery_df = pd.read_csv(delivery_file, sep="\t")
ptb_grids = delivery_df.loc[delivery_df['consensus_label'] == 'preterm', 'GRID'].unique()
term_grids = delivery_df.loc[delivery_df['consensus_label'] == 'term', 'GRID'].unique()


# %%
models_shelf = dict()


# -----------
# train latent factor models
# -----------
###
###    latent factors dataset prediction
###

# format lf dataset
indivd_lf_df = pd.read_csv(individ_fmatrix)
indivd_lf_df['GRID']=grid_labels

indivd_lf_df['preg_type'] = np.nan
indivd_lf_df.loc[indivd_lf_df['GRID'].isin(ptb_grids), 'preg_type'] = 'preterm'
indivd_lf_df.loc[indivd_lf_df['GRID'].isin(term_grids), 'preg_type'] = 'term'

# create predictor and output labels
predictors =  [x for x in indivd_lf_df.columns if x.startswith("V")]
X = indivd_lf_df.loc[:,predictors].values
y = ((indivd_lf_df['preg_type']=='preterm')*1).values

# split into train test
X_train, y_train, X_test, y_test, annotated_df = create_held_out(X, y, indivd_lf_df, test_size=0.10)


# run xgboost model
max_evals=10
eval_metric='map'
best_xgb_rf, metrics_results, metrics_df, model_params, trial_df = run_xgb(X_train, y_train, X_test, y_test, max_evals, eval_metric)

# run logistic regression model
lr_clf, lr_auc_df, lr_metrics_results, lr_conf_matrix_df =  run_logistic_regression(X_train, y_train, X_test, y_test)


# save
models_shelf['lf_xgb'] = dict()
models_shelf['lf_xgb']['metrics_result']= metrics_results
models_shelf['lf_xgb']['y_test']= y_test
models_shelf['lf_logreg'] = dict()
models_shelf['lf_logreg']['metrics_result']= lr_metrics_results


###
###    phecodes based prediction
###
phe_df['preg_type'] = np.nan
phe_df.loc[phe_df['GRID'].isin(ptb_grids), 'preg_type'] = 'preterm'
phe_df.loc[phe_df['GRID'].isin(term_grids), 'preg_type'] = 'term'

# create predictor and output labels
X_phe = phe_df.loc[:,phe_labels].values
y_phe = ((phe_df['preg_type']=='preterm')*1).values

# split into train test
Xphe_train, yphe_train, Xphe_test, yphe_test, phe_annotated_df = create_held_out(X_phe, y_phe, phe_df, test_size=0.10)

# run xgboost model
phe_best_xgb_rf, phe_metrics_results, phe_metrics_df, phe_model_params, phe_trial_df = run_xgb(Xphe_train, yphe_train, Xphe_test, yphe_test,  1, eval_metric)

# logistic regression
lr_phe_clf, lr_phe_auc_df, lr_phe_metrics_results, lr_phe_conf_matrix_df =  run_logistic_regression(Xphe_train, yphe_train, Xphe_test, yphe_test)

# save
# models_shelf['phe_xgb'] = dict()
models_shelf['phe_xgb']['metrics_result']= phe_metrics_results
models_shelf['phe_logreg'] = dict()
models_shelf['phe_logreg']['metrics_result']= lr_phe_metrics_results

# pickle.dump(models_shelf, open(OUTPUT_DIR.joinpath('lf_phe_model_evals_dict.pickle'),'wb') )
# %%
# -----------
# model evaluation
# -----------

# %%
### ROC AUC
sns.set(style="ticks",  font_scale=1.0, rc={"figure.figsize": (4, 4)})
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)
plot_roc(ax, models_shelf['lf_xgb']['metrics_result'], 'ROC', '#EE8866', bprop, linestyle=":", label='LFs-XGB', no_chance=True)
plot_roc(ax, models_shelf['phe_xgb']['metrics_result'], 'ROC', '#EE8866', bprop, linestyle="-", label='PHE-XGB', no_chance=False)

# plot_roc(ax,  models_shelf['lf_logreg']['metrics_result'], 'ROC', '#EE8866', bprop, linestyle=":", label='LFs-LR', no_chance=True)
# plot_roc(ax, models_shelf['phe_logreg']['metrics_result'], 'ROC', '#BBCC33', bprop, linestyle="-", label='PHE-LR', no_chance=False)



# TODO: modify labels to include xgboost.
svfig('roc-auc_lf_and_phe_pred_ptb')


# %%
### PR AUC
sns.set(style="ticks",  font_scale=1.0, rc={"figure.figsize": (4, 4)})
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)
plot_pr(ax, models_shelf['lf_xgb']['metrics_result'], y_test, 'PR', '#EE8866', bprop,linestyle=":", label='LFs-XGB', no_chance=True)
plot_pr(ax, models_shelf['phe_xgb']['metrics_result'], y_test, 'PR', '#EE8866', bprop,linestyle="-", label='PHE-XGB', no_chance=False)



svfig('pr-auc_lf_and_phe_pred_ptb')


# # %%
# ### confusion matrix
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# # confusion matrix on test set
# cm = confusion_matrix(y_test, y_pred,)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=('not-preterm', 'preterm'))
# disp.plot(cmap='Blues')
#


# %%
## shap scores

import shap

### must use "X_test" because this is the standardized feature matrix
explainer = shap.TreeExplainer(best_xgb_rf)
test_shap = explainer.shap_values(X_test)
avg_importance = np.mean(np.abs(test_shap), axis=0)


shap_df = pd.DataFrame({'feature': predictors, 'Importance':avg_importance})


