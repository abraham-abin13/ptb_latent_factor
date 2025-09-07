#!/bin/bash
#SBATCH --mail-user=abraham.abin13@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --time=1-08:00:00
#SBATCH --mem=450G
#SBATCH --output=slurm_out/ptb_term_no_preg_%a_%A.out
#SBATCH --array=45-50

# array=18-30



# term icd9
OUTPUT_DIR="/dors/capra_lab/users/abraha1/projects/PTB_phewas/data/2020_07_30_longit_topic_modeling/r_parafac_outputs/preterm_term_all_race_icd9_no_preg/multiway_rank_${SLURM_ARRAY_TASK_ID}"
INPUT_FILE="/dors/capra_lab/users/abraha1/projects/PTB_phewas/scripts/2020_07_30_longit_topic_modeling/input_tensors/preterm_term_all_race_icd9_no_preg/tensor_preterm_term_all_race_icd9_no_preg_within_5yr.pickle"

PHE_CONSTRAINT="ortnon"
TIME_CONSTRAINT="nonneg"

module load Intel/2019.1.144  IntelMPI/2018.4.274 R/3.6.0
module load Anaconda3/4.4.0
source activate py38_ml

echo "Rscript parafac_v1.R -n $SLURM_ARRAY_TASK_ID -i $INPUT_FILE -p $PHE_CONSTRAINT -t $TIME_CONSTRAINT -o $OUTPUT_DIR"
Rscript parafac_v1.R -n $SLURM_ARRAY_TASK_ID -i $INPUT_FILE -p $PHE_CONSTRAINT -t $TIME_CONSTRAINT -o $OUTPUT_DIR




