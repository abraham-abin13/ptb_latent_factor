#!/bin/python
# This script will
#
#
#
# Abin Abraham
# created on: 2020-07-30 19:48:08

import os
import numpy as np
import pandas as pd
from datetime import datetime

DATE = datetime.now().strftime("%Y-%m-%d")
import pickle


###
#   paths
###

cohort_label='preterm_term_all_race_icd9_preg_rm'
phe_icd9_file = "/dors/capra_lab/users/abraha1/projects/PTB_phewas/data/2020_07_30_longit_topic_modeling/raw_data/phecodes_from_icd9_delivery_cohort.tsv"
# phe_icd10_file = "/dors/capra_lab/users/abraha1/projects/PTB_phewas/data/2020_07_30_longit_topic_modeling/raw_data/phecodes_from_icd10_delivery_cohort.tsv"
delivery_file = "/dors/capra_lab/users/abraha1/projects/PTB_phewas/data/2020_07_30_longit_topic_modeling/raw_data/est_delivery_date_at_least_one_icd_cpt_ega.tsv"
demo_file = "/dors/capra_lab/users/abraha1/projects/PTB_phewas/data/2020_07_30_longit_topic_modeling/raw_data/complete_demographics.tsv"
tensor_output_dir = "/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/input_tensors/preterm_term_all_race_icd9_no_preg"
icd9_to_phe_file ="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/phecodes/phecode_icd9_map_unrolled.csv"



# -----------
# functions
# -----------
# codes w. ICD-9, ICD-10
def load_delivery_codes():
    return {'preterm_code': ['O60.1 ', 'O60.10', 'O60.10X0', 'O60.10X1', 'O60.10X2', 'O60.10X3', 'O60.10X4', 'O60.10X5', 'O60.10X9', 'O60.12',   'O60.12X0', 'O60.12X1', 'O60.12X2', 'O60.12X3', 'O60.12X4', 'O60.12X5', 'O60.12X9', 'O60.13', 'O60.13X0',
                     'O60.13X1', 'O60.13X2', 'O60.13X3', 'O60.13X4', 'O60.13X5', 'O60.13X9', 'O60.14', 'O60.14X0', 'O60.14X1', 'O60.14X2', 'O60.14X3', 'O60.14X4', 'O60.14X5', 'O60.14X9', '644.2', '644.20', '644.21'],
            'term_code':['O60.20', 'O60.20X0', 'O60.20X1', 'O60.20X2', 'O60.20X3', 'O60.20X4', 'O60.20X5', 'O60.20X9', 'O60.22', 'O60.22X0', 'O60.22X1', 'O60.22X2', 'O60.22X3', 'O60.22X4', 'O60.22X5', 'O60.22X9', 'O60.23', 'O60.23X0', 'O60.23X1',
                  'O60.23X2', 'O60.23X3', 'O60.23X4', 'O60.23X5', 'O60.23X9', 'O80', 'O48.0', '650', '645.1', '645.10', '645.11', '645.13', '649.8', '649.81', '649.82'],
    'post_term':['O48.1', '645.2', '645.20', '645.21', '645.23', '645.00', '645.01', '645.03'],
    'less_20wk_codes':['Z3A.0', 'Z3A.00', 'Z3A.01', 'Z3A.08', 'Z3A.09', 'Z3A.1', 'Z3A.10', 'Z3A.11', 'Z3A.12', 'Z3A.13', 'Z3A.14', 'Z3A.15', 'Z3A.16', 'Z3A.17', 'Z3A.18', 'Z3A.19'],
    'bw_20_and_37wk_codes':['Z3A.2', 'Z3A.20', 'Z3A.21', 'Z3A.22', 'Z3A.23', 'Z3A.24', 'Z3A.25', 'Z3A.26', 'Z3A.27', 'Z3A.28', 'Z3A.29', 'Z3A.3', 'Z3A.30', 'Z3A.31', 'Z3A.32', 'Z3A.33', 'Z3A.34',
                            'Z3A.35', 'Z3A.36', 'Z3A.37'],
    'bw_37_and_42wk_codes':['Z3A.38', 'Z3A.39', 'Z3A.4', 'Z3A.40', 'Z3A.41'],
    "bw_42_and_higher_codes":['Z3A.42','Z3A.49']}

# %%
# -----------
# main
# -----------

# set up
phe_icd9_df = pd.read_csv(phe_icd9_file, sep="\t",dtype={"GRID": "str", "Date": "str", "phecode": "str", "from": "str"})
demo_df = pd.read_csv(demo_file, sep="\t")
delivery_df = pd.read_csv(delivery_file, sep="\t")


# pick the earliest delivery
delivery_df.consensus_delivery = pd.to_datetime(delivery_df.consensus_delivery)
delivery_df.sort_values(["GRID", "consensus_delivery"], ascending=True, inplace=True)
first_delivery_df = delivery_df[ ~delivery_df.duplicated(subset=["GRID"], keep="first")].copy()
delivery_date_dict = dict(zip(first_delivery_df.GRID, first_delivery_df.consensus_delivery))

ptb_grids = first_delivery_df.loc[first_delivery_df['consensus_label'] == 'preterm', 'GRID'].unique()
term_grids = first_delivery_df.loc[first_delivery_df['consensus_label'] == 'term', 'GRID'].unique()


all_phe_df = phe_icd9_df.copy()
all_phe_df = all_phe_df.loc[all_phe_df["GRID"].isin(first_delivery_df.GRID)].copy()
all_phe_df["delivery_date"] = all_phe_df.GRID.map(delivery_date_dict)


# remove phecode that are associated with deliveries
icd9_to_phe_df = pd.read_csv( icd9_to_phe_file, )
delivery_codes = load_delivery_codes() #

icd9_to_phe_df.phecode = icd9_to_phe_df.phecode.map(str)
icd9_to_phe_df.icd9 =icd9_to_phe_df.icd9.map(str)
phe_to_rm = []
for key in delivery_codes:
    phe_to_rm.extend(list(icd9_to_phe_df.loc[icd9_to_phe_df['icd9'].isin(delivery_codes[key]), 'phecode'].values))


phe_preg_rm_df = all_phe_df[~all_phe_df['phecode'].isin(phe_to_rm)].copy()
# calculate difference between deliery date and date of code
phe_preg_rm_df.Date = pd.to_datetime(phe_preg_rm_df.Date)
phe_preg_rm_df["time_from_delivery"] = (phe_preg_rm_df.Date - phe_preg_rm_df.delivery_date)
phe_preg_rm_df["binned_years_from_delivery"] = phe_preg_rm_df.time_from_delivery // np.timedelta64(1, "Y")

# negative years are before delivery, positive year bins are after delivery
# 5 years before and 5 years after
# binarize
within_5yrs_df = phe_preg_rm_df[(phe_preg_rm_df["binned_years_from_delivery"] < 5) & (phe_preg_rm_df["binned_years_from_delivery"] > -5)].copy()
binarized_within_5yrs_df = within_5yrs_df[~within_5yrs_df.duplicated(subset=["GRID", "phecode", "binned_years_from_delivery"], keep="first")].copy()

# %%

# remove phecode with low prevalence
#       - code prevalence = # of individuals with at least one occurence of a code / total number of individuals
#       - remove codes with prevalence less than 0.5%
grid_by_phe_df = (binarized_within_5yrs_df.groupby(["GRID", "phecode"]).size().reset_index())
grid_by_phe_df.rename(columns={0: "count"}, inplace=True)

phe_summary_df = grid_by_phe_df.groupby("phecode").size().reset_index()
phe_summary_df.rename(columns={0: "phe_count"}, inplace=True)
phe_summary_df["prevalence_percent"] = (phe_summary_df["phe_count"] / binarized_within_5yrs_df.GRID.nunique() * 100)
phe_summary_df["is_rare"] = phe_summary_df["prevalence_percent"] < 0.005

keep_phe_codes = phe_summary_df.loc[~phe_summary_df["is_rare"], "phecode"].values

# remove codes with low prevalence
common_binarized_within5yrs_df = binarized_within_5yrs_df.loc[binarized_within_5yrs_df.phecode.isin(keep_phe_codes)].copy()
common_binarized_within5yrs_df["binarized_count"] = 1

# %%

#
# Create Tensor
#

# a3d = np.array(list(pdf.groupby('a').apply(pd.DataFrame.as_matrix)))
to_tensor_df = common_binarized_within5yrs_df.loc[
    :, ["GRID", "phecode", "binned_years_from_delivery", "binarized_count"]
].copy()


# side analysis 
labeled_tensor_df = to_tensor_df.copy()
labeled_tensor_df['delivery'] = 'not-preterm'
labeled_tensor_df.loc[labeled_tensor_df['GRID'].isin(ptb_grids), 'delivery'] = 'preterm'
temp_save_dir = "/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/manuscript/latent_factors_ptb_term/collaborators"
labeled_tensor_df.to_csv(os.path.join(temp_save_dir, f"individual_phecode_duration_delivery_type.tsv"), index=False, sep="\t")

# resume analysis 


phecode_axis = np.sort(to_tensor_df.phecode.unique())
binned_years_axis = np.sort(to_tensor_df.binned_years_from_delivery.unique())
grids_axis = np.sort(to_tensor_df.GRID.unique())




# to_tensor_df.query("phecode=='636.2'") contains preterm birth phecode
pickle.dump(
    phecode_axis,
    open(
        os.path.join(
            tensor_output_dir,
            f"phecode_axis_{cohort_label}_within_5yr.pickle",
        ),
        "wb",
    ),
)
pickle.dump(
    binned_years_axis,
    open(
        os.path.join(
            tensor_output_dir,
            f"binned_years_axis_{cohort_label}_within_5yr.pickle",
        ),
        "wb",
    ),
)
pickle.dump(
    grids_axis,
    open(
        os.path.join(
            tensor_output_dir,
            f"grids_axis_{cohort_label}_within_5yr.pickle",
        ),
        "wb",
    ),
)

# %%
# -----------
# preallocate tensor
# -----------
phe_tensor = np.zeros((len(phecode_axis), len(binned_years_axis), len(grids_axis)))
# create tensor
for z_ind, this_grid in enumerate(grids_axis):
    # this_grid = "R212583382"
    print(f"{z_ind:,} out of {len(grids_axis)}")
    # wide df
    wide_this_grid_df = (
        to_tensor_df.loc[to_tensor_df["GRID"] == this_grid]
        .pivot(
            index="phecode",
            columns="binned_years_from_delivery",
            values="binarized_count",
        )
        .reset_index()
    )

    # fill missing columns
    this_cols = wide_this_grid_df.columns.tolist()
    this_cols.remove("phecode")
    missing_cols = set(binned_years_axis) - set(this_cols)

    for newcol in missing_cols:
        wide_this_grid_df[newcol] = np.nan

    # reorder columns
    wide_this_grid_df = wide_this_grid_df.loc[
        :, ["phecode"] + list(binned_years_axis)
    ].copy()

    # fill missing phecodes
    missing_phecodes = set(phecode_axis) - set(wide_this_grid_df.phecode.values)
    miss_df = pd.DataFrame({"phecode": list(missing_phecodes)})
    for newcol in binned_years_axis:
        miss_df[newcol] = np.nan
    wide_this_grid_df = wide_this_grid_df.append(miss_df, sort=True)

    # sort phecode according to order
    wide_this_grid_df.set_index("phecode", inplace=True)
    this_sorted_slice = wide_this_grid_df.loc[phecode_axis, binned_years_axis].copy()

    phe_tensor[:, :, z_ind] = this_sorted_slice.fillna(0).values



# %%
# -----------
# check tensor
# -----------


pickle.dump(
    phe_tensor,
    open(
        os.path.join(
            tensor_output_dir, f"tensor_{cohort_label}_within_5yr.pickle"
        ),
        "wb",
    ),
)
print("saved")
