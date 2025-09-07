# Tests phecode or association with preterm birth vs. not-preterm birth.  
# Covariates include: age at delviery and length of EHR 
#
# Methods: 
#   - ICD9 and ICD10 codes are merged with phecode table 
#   - convert phecode into True/False if phecode count is > 3

rm(list=ls())
library(tidyverse)
library(dplyr)
library(knitr)
library(MatchIt)
library(lubridate)
library(PheWAS)

#----
#
# LOAD
#


output_dir="/Users/abraha1/GoogleDrive/1_MSTP/2_Research/1_CapraLab/projects/2020_ptb_phewas_zhang_mom_baby_vars/phecode_diff_by_race/results/phe_assoc_ptb_w_covars"

# load race and delivery date file 
root_path="/Users/abinabraham/GoogleDrive/1_MSTP/2_Research/1_CapraLab/projects/2020_ptb_phewas_zhang_mom_baby_vars/phecode_diff_by_race/convert_icd9_to_phe"
race_df = read_delim(file.path(root_path, "complete_demographics.tsv"), delim="\t", escape_double = F, trim_ws=F)
delivery_df = read_delim(file.path(root_path, "est_delivery_date_at_least_one_icd_cpt_ega.tsv"), delim="\t", escape_double=F, trim_ws=F)

# load icd9 and icd10
setwd(root_path)
icd9_file="full_ICD9_cohort.tsv"
icd10_file="full_ICD10_cohort.tsv"
icd9_df <- read_delim(icd9_file, "\t", escape_double = FALSE, trim_ws = TRUE) 
icd10_df <- read_delim(icd10_file, "\t", escape_double = FALSE, trim_ws = TRUE) 

#----
#
# FUNCTIONS
#

calc_ehr_length <- function(icd_df){
  
  temp = icd_df  %>% rename(code=ICD) 
  to_phe_df  = mapCodesToPhecodes(temp,  make.distinct = F)
  
  ehr_duration_df = to_phe_df%>% group_by(GRID) %>% summarize(ehr_length = max(Date) - min(Date))
  
  return (ehr_duration_df)
}


to_phecodes <- function(codes_df){
  id.sex_df = distinct(codes_df %>% select(GRID) %>% add_column(sex="F") %>% rename('id'="GRID"))
  phe_table = createPhenotypes(codes_df, 
                               translate=T,
                               add.phecode.exclusions = T,
                               id.sex=id.sex_df,
                               min.code.count=NA)
  return(phe_table)
  
}

run_phewas <- function(covar_df, cohort_phe_df, predictors){
  # assign cases ("all_preterm") as 1, others as 0 
  covar_df$binary_delivery = ifelse(covar_df$delivery == "all_preterm", 1, 0)
  analysis_df = inner_join(covar_df  %>%
                             select(GRID, binary_delivery, age_at_delivery_years, ehr_length),
                           cohort_phe_df, by="GRID")
  
  analysis_df$binary_delivery = as.logical(as.numeric(analysis_df$binary_delivery))
  
  
  
  phe_results = phewas(predictors=predictors, 
                       outcomes="binary_delivery", 
                       covariates=c("age_at_delivery_years", "ehr_length"),
                       data=analysis_df,
                       clean.phecode.predictors = T, additive.genotypes = F, return.models = F,
                       min.records=100)
  
  return(phe_results)}


plot_results <- function(results_df, title, label, output_dir){
  pl = phewasManhattan(results_df %>% rename('phenotype'='predictor'),
                       title=title,
                       OR.direction=T,
                       point.size=1,
                       annotate.phenotype.description=F,
                       annotate.size=3,
                       annotate.only.largest=F)
  
  ggsave(sprintf("%s_%s_manhattan_plot.pdf", today(), label),plot=pl,  path=output_dir, width=14, height=6)
  print('fig saved!')
}


#----
#
# MAIN 
#

# map one race and delivery type to each person 
black_df = race_df %>% filter(RACE_LIST == "AFRICAN_AMERICAN") %>% select(GRID, DOB, RACE_LIST) %>% add_column(race="black")
white_df = race_df %>% filter(RACE_LIST == "CAUCASIAN") %>% select(GRID, DOB, RACE_LIST) %>% add_column(race="white")
filtered_race_df = bind_rows(black_df, white_df)


#
# assign case (≥1 preterm) and controls (no preterm births)
#

# preterm is at least one preterm birth 
# not-ptb is never preterm birth 

delivery_df = delivery_df%>% filter(!consensus_label =="None")
ptb_df = delivery_df %>% filter(consensus_label == "preterm") %>% select(GRID, consensus_delivery, consensus_label) %>% add_column(delivery="all_preterm")
not_ptb_df = delivery_df %>% filter(!GRID %in% unique(ptb_df$GRID)) %>% filter(!consensus_label == "preterm")  %>% select(GRID, consensus_delivery, consensus_label) %>% add_column(delivery="no preterm")
rm(delivery_df)

# keep only one GRID; remove rows for multiple deliveries
case_control_df = bind_rows(ptb_df, not_ptb_df) %>% select(-consensus_label) %>% distinct(GRID, .keep_all = T) 
rm(ptb_df)
rm(not_ptb_df)

# merge race,age and delivery type 
demo_delivery_df = inner_join(filtered_race_df, case_control_df, by="GRID")
demo_delivery_df$DOB = as.Date(demo_delivery_df$DOB)
rm(filtered_race_df)
rm(case_control_df)

# cross tabs ..
demo_delivery_df %>% group_by(delivery, race) %>% summarize(n=n()) %>% spread(race, n) %>% kable() 

# calc age at delivery 
demo_delivery_df$age_at_delivery = demo_delivery_df$consensus_delivery - demo_delivery_df$DOB
demo_delivery_df$age_at_delivery_years = as.numeric(demo_delivery_df$age_at_delivery/365)
demo_delivery_df = demo_delivery_df %>% select(-age_at_delivery)


# calc ehr length
# keep only ICD codes that map to PheCodes
phemap=PheWAS::phecode_map
filt_icd9_df = icd9_df %>% filter(ICD %in% phemap$code)
filt_icd10_df = icd10_df %>% filter(ICD %in% phemap$code)
rm(icd9_df)
rm(icd10_df)

# combined icd9 and icd10 so we can calculate ehr length 
icd_df= rbind(filt_icd9_df %>% add_column(vocabulary_id="ICD9CM"), filt_icd10_df %>% add_column(vocabulary_id="ICD10CM"))
icd_ehr_length = calc_ehr_length(icd_df)
rm(icd_df)

# merge with phecodes, create covars 
covar_df = inner_join(icd_ehr_length, demo_delivery_df)


#
# create phecode wide df
#

# index will be summed when aggregating
clean_icd9_df = filt_icd9_df %>% add_column(vocabulary_id = "ICD9CM", index=1) %>% rename(code=ICD)  %>% select(-c(Date, concat_row)) %>% select(GRID, vocabulary_id, code, index)
clean_icd10_df = filt_icd10_df %>% add_column(vocabulary_id = "ICD10CM", index=1) %>% rename(code=ICD)  %>% select(-c(Date, concat_row)) %>% select(GRID, vocabulary_id, code, index)
codes_df = rbind(clean_icd9_df, clean_icd10_df)
rm(clean_icd9_df)
rm(clean_icd10_df)
rm(filt_icd9_df)
rm(filt_icd10_df) 

white_codes_df = codes_df %>% filter( GRID %in% white_df$GRID)
black_codes_df = codes_df %>% filter( GRID %in% black_df$GRID)
rm(codes_df)

# map to phecodes
white_phe_codes_df = left_join(white_codes_df, phemap, by=c('vocabulary_id', 'code')) 
black_phe_codes_df = left_join(black_codes_df, phemap, by=c('vocabulary_id', 'code')) 

white_wide_phe_table = white_phe_codes_df %>% count(GRID, phecode) %>% spread(phecode, n, fill=0)
black_wide_phe_table = black_phe_codes_df %>% count(GRID, phecode) %>% spread(phecode, n, fill=0)


# create table 1 
white_wide_covar_df = inner_join(covar_df, white_wide_phe_table)
black_wide_covar_df = inner_join(covar_df, black_wide_phe_table)

make.table(dat          = black_wide_covar_df,
           strat        = c("delivery"),
           # cat.varlist  = c("race"),
           cont.varlist = c("age_at_delivery_years"),
           cont.header  = c("Age at Delivery"),
           cat.rmstat= list(c("miss","col", "row")),
           cont.rmstat  = list(c( "meansd", "minmax","miss", "q1q3")),
           output="html", caption='Black Cohort', footer='', tspanner="",
           n.tspanner=4, cgroup="",n.cgroup=4)



make.table(dat          = white_wide_covar_df,
           strat        = c("delivery"),
           # cat.varlist  = c("race"),
           cont.varlist = c("age_at_delivery_years"),
           cont.header  = c("Age at Delivery"),
           cat.rmstat= list(c("miss","col", "row")),
           cont.rmstat  = list(c( "meansd", "minmax","miss", "q1q3")),
           output="html", caption='White Cohort', footer='', tspanner="",
           n.tspanner=4, cgroup="",n.cgroup=4)

# convert to True/False predictors 
convert_bool <- function(wide_phe_table, threshold){

  isBool <- function(x){ ifelse(x >threshold, TRUE, FALSE) }
  temp_for_phewas = wide_phe_table %>% select(-GRID) %>% mutate_each(isBool) %>% mutate_if(is.logical, as.factor)
  for_phewas = bind_cols(wide_phe_table[1] , temp_for_phewas)
  
  return(for_phewas)
}

white_bool_phe_df = convert_bool(white_wide_phe_table, 3)
black_bool_phe_df = convert_bool(black_wide_phe_table, 3)

#
# run phewas 
#

# run thresholded 
black_results = run_phewas(covar_df, black_bool_phe_df, colnames(black_bool_phe_df)[-1])
white_results = run_phewas(covar_df, white_bool_phe_df, colnames(white_bool_phe_df)[-1])

write_tsv(black_results, file.path(output_dir, "results_black_assoc_w_ptb.tsv"))
write_tsv(white_results, file.path(output_dir, "results_white_assoc_w_ptb.tsv"))


# TODO:  run continuous numbers 

#
# plotting
#
plot_results(black_results, 'black cohort - assoc with ptb', 'black_assoc_w_ptb', output_dir)
plot_results(white_results, 'white cohort - assoc with ptb', 'white_assoc_w_ptb', output_dir)

