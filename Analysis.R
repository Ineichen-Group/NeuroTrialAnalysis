# Load necessary libraries
library(readr)
library(mgcv)
library(tidyverse)
library(nnet)
library(ggplot2)

# Read the dataset
#clintrials_data <- read_csv("clintrials_data_to_analyse_for_trends_20250130.csv")
clintrials_data <- read_csv("clintrials_data_to_analyse_for_trends_20250219_v3.csv")

# Convert start_year to integer
clintrials_data <- clintrials_data %>%
  mutate(start_year = as.integer(start_year))

#########################
### Industry funding ####
#########################
# Chi-square test to check funding differences across years
# Create a contingency table of start_year vs. agency_class_refined (funding type)
funding_table <- table(clintrials_data$start_year, clintrials_data$agency_class_refined)

# Run chi-square test
chi_test <- chisq.test(funding_table)
print(chi_test)  # Display chi-square test results

# Create a binary variable for industry funding
clintrials_data <- clintrials_data %>%
  mutate(industry_funding = ifelse(agency_class_refined == "INDUSTRY", 1, 0))

# Fit a generalized additive model (GAM) to check trends in industry funding over time
gam_model <- gam(industry_funding ~ s(start_year), family = binomial, data = clintrials_data)

# Display GAM summary
summary(gam_model)

# Plot the GAM smooth function
plot(gam_model, main = "Industry Funding Trend Over Time")

plot(clintrials_data$start_year)

#########################
### University funding ####
#########################

# Create a binary variable for university funding
clintrials_data <- clintrials_data %>%
  mutate(university_funding = ifelse(agency_class_refined == "UNIVERSITY", 1, 0))

# Chi-square test for university funding over time
university_funding_table <- table(clintrials_data$start_year, clintrials_data$university_funding)
chi_test_university <- chisq.test(university_funding_table)
print(chi_test_university)

# Fit a GAM model for university funding over time
gam_university <- gam(university_funding ~ s(start_year), family = binomial, data = clintrials_data)

# Display GAM summary
summary(gam_university)

# Plot the GAM smooth function
plot(gam_university, main = "University Funding Trend Over Time")

#########################
### NIH Funding ####
#########################

# Create a binary variable for NIH funding
clintrials_data <- clintrials_data %>%
  mutate(NIH_funding = ifelse(agency_class_refined == "NIH", 1, 0))

# Chi-square test for NIH funding over time
NIH_funding_table <- table(clintrials_data$start_year, clintrials_data$NIH_funding)
chi_test_NIH <- chisq.test(NIH_funding_table)
print(chi_test_NIH)

# Fit a GAM model for NIH funding over time
gam_NIH <- gam(NIH_funding ~ s(start_year), family = binomial, data = clintrials_data)

# Display GAM summary
summary(gam_NIH)

# Plot the GAM smooth function
plot(gam_NIH, main = "NIH Funding Trend Over Time")

# Save the GAM plot
plot_gam(gam_NIH, "NIH Funding Trend Over Time", "gam_NIH_funding.png")

# Spearman test for NIH funding
NIH_corr <- spearman_test(clintrials_data, "start_year", "NIH_funding", "NIH Funding Trend")
results_summary <- append(results_summary, list(NIH_corr))

#########################
### Masking  ####
#########################
# Create a contingency table of start_year vs. masking type
masking_table <- table(clintrials_data$start_year, clintrials_data$masking)

# Run Chi-square test
chi_test_masking <- chisq.test(masking_table)

# Display test results
print(chi_test_masking)


# Convert masking to binary indicators
clintrials_data_masking <- clintrials_data %>%
  mutate(masking_open = ifelse(masking == "None (Open Label)", 1, 0),
         masking_quadruple = ifelse(masking == "Quadruple", 1, 0),
         masking_double = ifelse(masking == "Double", 1, 0))

# Fit separate GAM models for each type of masking
gam_masking_open <- gam(masking_open ~ s(start_year), family = binomial, data = clintrials_data_masking)
gam_masking_quadruple <- gam(masking_quadruple ~ s(start_year), family = binomial, data = clintrials_data_masking)
gam_masking_double <- gam(masking_double ~ s(start_year), family = binomial, data = clintrials_data_masking)

# Display summaries
summary(gam_masking_open)
summary(gam_masking_quadruple)
summary(gam_masking_double)

# Plot trends
plot(gam_masking_open, main = "Open Label Trend Over Time")
plot(gam_masking_quadruple, main = "Quadruple Masking Trend Over Time")
plot(gam_masking_double, main = "Double Masking Trend Over Time")

#########################
### Randomization   ####
#########################
# Create a binary variable for randomized allocation
clintrials_data <- clintrials_data %>%
  mutate(randomized = ifelse(allocation == "Randomized", 1, 0))

# Chi-square test for randomization over time
randomization_table <- table(clintrials_data$start_year, clintrials_data$randomized)
chi_test_randomization <- chisq.test(randomization_table)
print(chi_test_randomization)

# Fit a GAM model for randomization over time
gam_randomization <- gam(randomized ~ s(start_year), family = binomial, data = clintrials_data)

# Display GAM summary
summary(gam_randomization)

# Plot the GAM smooth function
plot(gam_randomization, main = "Randomization Trend Over Time")

#########################
### Non-Randomized Trials ####
#########################

# Create a binary variable for non-randomized trials
clintrials_data <- clintrials_data %>%
  mutate(non_randomized = ifelse(randomized == 0, 1, 0))

# Chi-square test for non-randomized trials over time
non_randomized_table <- table(clintrials_data$start_year, clintrials_data$non_randomized)
chi_test_non_randomized <- chisq.test(non_randomized_table)
print(chi_test_non_randomized)

# Fit a GAM model for non-randomized trials over time
gam_non_randomized <- gam(non_randomized ~ s(start_year), family = binomial, data = clintrials_data)

# Display GAM summary
summary(gam_non_randomized)

# Plot the GAM smooth function
plot(gam_non_randomized, main = "Non-Randomized Trials Trend Over Time")

# Save the GAM plot
plot_gam(gam_non_randomized, "Non-Randomized Trials Trend Over Time", "gam_non_randomized.png")

# Spearman test for non-randomized trials
non_randomized_corr <- spearman_test(clintrials_data, "start_year", "non_randomized", "Non-Randomized Trials Trend")
results_summary <- append(results_summary, list(non_randomized_corr))

#########################
### Reporting        ####
#########################
# Convert were_results_reported to binary (already a boolean)
clintrials_data <- clintrials_data %>%
  mutate(results_reported = as.integer(were_results_reported))

# Chi-square test for results reporting over time
results_table <- table(clintrials_data$start_year, clintrials_data$results_reported)
chi_test_results <- chisq.test(results_table)
print(chi_test_results)

# Fit a GAM model for reported results over time
gam_results <- gam(results_reported ~ s(start_year), family = binomial, data = clintrials_data)

# Display GAM summary
summary(gam_results)

# Plot the GAM smooth function
plot(gam_results, main = "Results Reporting Trend Over Time")


# Function to generate and save GAM plots
plot_gam <- function(gam_model, title, filename) {
  png(filename, width = 800, height = 600)  # Save as PNG
  plot(gam_model, main = title, shade = TRUE, rug = TRUE)
  dev.off()
}

# Plot GAM for industry funding
plot_gam(gam_model, "Industry Funding Trend Over Time", "gam_industry_funding.png")

# Plot GAM for university funding
plot_gam(gam_university, "University Funding Trend Over Time", "gam_university_funding.png")

# Plot GAM for results reporting
plot_gam(gam_results, "Results Reporting Trend Over Time", "gam_results_reporting.png")

# Plot GAM for randomization
plot_gam(gam_randomization, "Randomization Trend Over Time", "gam_randomization.png")

# Plot GAM for masking types
plot_gam(gam_masking_open, "Open Label Trend Over Time", "gam_masking_open.png")
plot_gam(gam_masking_quadruple, "Quadruple Masking Trend Over Time", "gam_masking_quadruple.png")
plot_gam(gam_masking_double, "Double Masking Trend Over Time", "gam_masking_double.png")

# Function to run Spearman correlation test and summarize results
spearman_test <- function(data, x_var, y_var, description) {
  # Run Spearman correlation
  cor_test <- cor.test(data[[x_var]], data[[y_var]], method = "spearman")
  list(
    description = description,
    spearman_rho = cor_test$estimate,
    p_value = cor_test$p.value,
    significance = ifelse(cor_test$p.value < 0.05, "Significant", "Not Significant")
  )
}

# Collect results in a list
results_summary <- list()

#########################
### Industry Funding ####
#########################
# Spearman test for industry funding
industry_corr <- spearman_test(clintrials_data, "start_year", "industry_funding", "Industry Funding Trend")
results_summary <- append(results_summary, list(industry_corr))

#########################
### University Funding ####
#########################
# Spearman test for university funding
university_corr <- spearman_test(clintrials_data, "start_year", "university_funding", "University Funding Trend")
results_summary <- append(results_summary, list(university_corr))

#########################
### Masking ####
#########################
# Spearman test for open label masking
masking_open_corr <- spearman_test(clintrials_data_masking, "start_year", "masking_open", "Open Label Masking Trend")
results_summary <- append(results_summary, list(masking_open_corr))

# Spearman test for quadruple masking
masking_quadruple_corr <- spearman_test(clintrials_data_masking, "start_year", "masking_quadruple", "Quadruple Masking Trend")
results_summary <- append(results_summary, list(masking_quadruple_corr))

# Spearman test for double masking
masking_double_corr <- spearman_test(clintrials_data_masking, "start_year", "masking_double", "Double Masking Trend")
results_summary <- append(results_summary, list(masking_double_corr))

#########################
### Randomization ####
#########################
# Spearman test for randomization
randomization_corr <- spearman_test(clintrials_data, "start_year", "randomized", "Randomization Trend")
results_summary <- append(results_summary, list(randomization_corr))

#########################
### Results Reporting ####
#########################
# Spearman test for results reporting
results_corr <- spearman_test(clintrials_data, "start_year", "results_reported", "Results Reporting Trend")
results_summary <- append(results_summary, list(results_corr))

#########################
### Print Summary ####
#########################
cat("Summary of GAM significance and Spearman correlations:\n\n")
for (result in results_summary) {
  cat(result$description, "\n")
  cat("  - Spearman rho:", round(result$spearman_rho, 3), "\n")
  cat("  - P-value:", format.pval(result$p_value), "\n")
  cat("  - Significance:", result$significance, "\n\n")
}


#### By phase
clintrials_data_phase <- read_csv("clintrials_data_to_analyse_for_trends_20250225_with_phase.csv")

# Create a binary variable for missing randomization (TRUE if missing, FALSE otherwise)
clintrials_data_phase$missing_randomization <- is.na(clintrials_data_phase$allocation)

# Summarize missing randomization counts for Phase 2 and Phase 3
phase_summary <- clintrials_data_phase %>%
  filter(phase %in% c("Phase 2", "Phase 3")) %>%
  group_by(phase) %>%
  summarise(
    missing = sum(missing_randomization),
    total = n()
  ) %>%
  mutate(non_missing = total - missing)

# Create a contingency table
contingency_table <- matrix(
  c(phase_summary$missing[1], phase_summary$non_missing[1], 
    phase_summary$missing[2], phase_summary$non_missing[2]),
  nrow = 2,
  byrow = TRUE
)

# Perform Chi-square test
chi_test <- chisq.test(contingency_table)

# Display results
chi_test

# Create binary variables for double-blind and quadruple masking
clintrials_data_phase$double_blind <- clintrials_data_phase$masking == "Double"
clintrials_data_phase$quadruple_blind <- clintrials_data_phase$masking == "Quadruple"

# Summarize counts for Phase 2 and Phase 3
masking_summary <- clintrials_data_phase %>%
  filter(phase %in% c("Phase 2", "Phase 3")) %>%
  group_by(phase) %>%
  summarise(
    double_blind = sum(double_blind, na.rm = TRUE),
    quadruple_blind = sum(quadruple_blind, na.rm = TRUE),
    total = n()
  ) %>%
  mutate(
    non_double_blind = total - double_blind,
    non_quadruple_blind = total - quadruple_blind
  )

# Create contingency tables
double_blind_table <- matrix(
  c(masking_summary$double_blind[1], masking_summary$non_double_blind[1],
    masking_summary$double_blind[2], masking_summary$non_double_blind[2]),
  nrow = 2, byrow = TRUE
)

quadruple_blind_table <- matrix(
  c(masking_summary$quadruple_blind[1], masking_summary$non_quadruple_blind[1],
    masking_summary$quadruple_blind[2], masking_summary$non_quadruple_blind[2]),
  nrow = 2, byrow = TRUE
)

# Perform Chi-square tests
chi_test_double <- chisq.test(double_blind_table)
chi_test_quadruple <- chisq.test(quadruple_blind_table)

# Display results
chi_test_double
chi_test_quadruple

# Create a binary variable for results reporting (TRUE if results reported, FALSE otherwise)
clintrials_data_phase$reported_results <- clintrials_data_phase$were_results_reported == TRUE

# Summarize results reporting counts for Phase 2 and Phase 3
results_summary <- clintrials_data_phase %>%
  filter(phase %in% c("Phase 2", "Phase 3")) %>%
  group_by(phase) %>%
  summarise(
    reported = sum(reported_results, na.rm = TRUE),
    total = n()
  ) %>%
  mutate(non_reported = total - reported)

# Create a contingency table
results_table <- matrix(
  c(results_summary$reported[1], results_summary$non_reported[1],
    results_summary$reported[2], results_summary$non_reported[2]),
  nrow = 2,
  byrow = TRUE
)

# Perform Chi-square test
chi_test_results <- chisq.test(results_table)

# Display results
chi_test_results