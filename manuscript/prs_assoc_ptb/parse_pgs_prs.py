
#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on: 2020-12-21 11:30:54



import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
DATE = datetime.now().strftime('%Y-%m-%d')

import gzip

### PATHS
DIR = Path("/dors/capra_lab/users/abraha1/projects/PTB_phewas/data/2020_07_30_longit_topic_modeling/polygenic_risk_scores/downloaded_prs")

PLINK_BIM_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phewas/data/biovu_prs_cs/plink_cohort/ptb_and_all_controls_white.bim"


OUTPUT_DIR=Path("/dors/capra_lab/users/abraha1/projects/PTB_phewas/data/2020_07_30_longit_topic_modeling/polygenic_risk_scores/formatted_prs_for_plink")

# %%
# -----------
# main
# -----------


meta_data_df = pd.DataFrame()
formatted_status_df = pd.DataFrame()
for zpfile in DIR.glob(pattern="*.txt.gz"):



    pgsID = ''
    genome_build = ''
    reported_trait = ''

    # loop through each file
    with gzip.open(zpfile, 'rb') as fo:


        # -----------
        # check genome build
        # -----------
        line_num =0
        for line in fo:

            line = line.decode("utf-8")
            if line.startswith("# PGS ID"):
                pgsID = line.split("= ")[-1].splitlines()[0]


            if line.startswith("# Original Genome Build"):
                genome_build = line.split("= ")[-1].splitlines()[0]

            if line.startswith("# Reported Trait"):
                reported_trait = line.split("= ")[-1].splitlines()[0]


            line_num += 1
            if line_num == 10:
                add_row = pd.DataFrame({'pgsID':[pgsID], 'reported_trait':[reported_trait], 'genome_build':[genome_build]})
                meta_data_df = meta_data_df.append(add_row)
                break

    # -----------
    # format summary stats
    # -----------

    print(f"loading df --> {zpfile.name}")
    df = pd.read_csv(zpfile, sep="\t", comment='#')

    # keep format,
    # can have  header
    # var ID is column 1 (should be rsID)
    # effect allele is column 2
    # weight is column 3


    col_to_keep = ['rsID','effect_allele', 'effect_weight']
    if not np.all([x in df.columns for x in col_to_keep]):
        print(f"Missing columns --> {df.columns}")
        add_this_row = pd.DataFrame({'Name':[zpfile.name], 'cols_present':[False]})
        formatted_status_df = formatted_status_df.append(add_this_row)
        continue

    add_this_row = pd.DataFrame({'Name':[zpfile.name], 'cols_present':[True]})
    formatted_status_df = formatted_status_df.append(add_this_row)
    formatted_df = df.loc[:, col_to_keep].copy()


    # write
    output_file = OUTPUT_DIR.joinpath(f'{zpfile.name.split(".txt")[0]}_formatted_for_plink.tsv')
    if not output_file.exists():
        print("wrote file! ")
        formatted_df.to_csv(OUTPUT_DIR.joinpath(f'{zpfile.name.split(".txt")[0]}_formatted_for_plink.tsv'), sep="\t", index=False)

# %%
# -----------
# redo missing files
# -----------
bim_df = pd.read_csv( PLINK_BIM_FILE, sep="\t", names=['chr','rsid','b','pos','a1','a2'])
bim_df['chr_pos'] = bim_df['chr'].map(str) +"_"+bim_df['pos'].map(str)


# the prs that didn't format properly all have chrom and position that can be used to get the rsID
files_to_redo = formatted_status_df.loc[formatted_status_df['cols_present']==False, 'Name']


prs_root = zpfile.parent
for prs_file in files_to_redo:

    this_df = pd.read_csv( prs_root.joinpath(prs_file), sep="\t", comment='#')

    this_df['chr_pos'] = this_df['chr_name'].map(str) +"_"+this_df['chr_position'].map(str)


    overlap_bim_df = bim_df.loc[bim_df['chr_pos'].isin(this_df['chr_pos'])].copy()

    print("{} overlap --> {:.2f}%".format(prs_file, overlap_bim_df.shape[0]/this_df.shape[0]*100))
    map_dict = dict(zip(overlap_bim_df['chr_pos'], overlap_bim_df['rsid']))


    this_df['rsID']= this_df['chr_pos'].map(map_dict)

    col_to_keep = ['rsID','effect_allele', 'effect_weight']

    formatted_df = this_df.loc[~this_df['rsID'].isna(), col_to_keep].copy()

    formatted_df.to_csv(OUTPUT_DIR.joinpath(f'{prs_file.split(".txt")[0]}_formatted_for_plink.tsv'), sep="\t", index=False)

# %%


missing_files = []
for index, this_file in enumerate(DIR.glob(pattern="*.txt.gz")):

    check_file = OUTPUT_DIR.joinpath(f'{this_file.name.split(".txt")[0]}_formatted_for_plink.tsv')
    if not check_file.exists():
        missing_files.append(this_file)

missing_files


