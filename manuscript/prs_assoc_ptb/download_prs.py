#!/bin/python
# This script will ...
#
#
#
# Abin Abraham
# created on: 2020-12-21 10:26:41


import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

DATE = datetime.now().strftime('%Y-%m-%d')



### PATHS
PRS_MANIFEST_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phewas/data/2020_07_30_longit_topic_modeling/polygenic_risk_scores/pgs_scores_data.xlsx"

PRS_DOWNLOAD_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phewas/data/2020_07_30_longit_topic_modeling/polygenic_risk_scores/downloaded_prs"

# %%
# -----------
# main
# -----------


sheets = pd.read_excel(PRS_MANIFEST_FILE, sheet_name=None)
sheets.keys()

df = sheets['selected_prs']
df.shape

df['PGS Scoring File (FTP Link)']


for ind, row in df.iterrows():

    if row["PGS Scoring File (FTP Link)"].find('Check') != -1:
        print(f'Skipping {ind}, url: {row["PGS Scoring File (FTP Link)"]}')
        continue

    url_link = row["PGS Scoring File (FTP Link)"]


    print(f"on {ind}, url: {url_link}")
    stream = os.popen(f'wget -P {PRS_DOWNLOAD_DIR} {url_link}')
    print(stream.read())
    print("----")
