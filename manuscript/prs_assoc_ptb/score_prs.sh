#!/bin/bash
#SBATCH --mail-user=abraham.abin13@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem=10G
#SBATCH --output=prs_%A_%a.out
#SBATCH --array=1


bfile="/dors/capra_lab/users/abraha1/projects/PTB_phewas/data/biovu_prs_cs/plink_cohort/ptb_and_all_controls_white"

DataDir="/dors/capra_lab/users/abraha1/projects/PTB_phewas/data/2020_07_30_longit_topic_modeling/polygenic_risk_scores/formatted_prs_for_plink"
scorefile=$( ls $DataDir | grep tsv | awk -v line=${SLURM_ARRAY_TASK_ID} '{if (NR==line) print $0}')

output_file="/dors/capra_lab/users/abraha1/projects/PTB_phewas/data/2020_07_30_longit_topic_modeling/polygenic_risk_scores/plink_calculated_prs"



# scorefile columns order: varID, allele, weight (specify the column numbers)

scorefile="PGS000302_formatted_for_plink.tsv"
label=${scorefile%_formatted_for_plink.tsv}

echo "plink --bfile $bfile --score ${DataDir}"/"${scorefile} sum --out $output_file"
plink --bfile $bfile --score ${DataDir}"/"${scorefile} sum --out ${output_file}"/"${label}

