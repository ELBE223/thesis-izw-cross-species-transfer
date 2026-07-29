# =============================================================================
# 03_H3_LOSO_R2_summary_final.R
# =============================================================================
# Author: Lucas Beseler
#
# Purpose:
# - Refit the primary H3 model after excluding each species in turn.
# - Summarize leave-one-species-out (LOSO) coefficient stability.
# - Recalculate marginal and conditional R2 for the primary model.
# - Export manuscript-ready values, diagnostics, and a LOSO forest plot.
#
# Dependency:
# - Run 01_H3_mixed_effects_primary_final.R first.
# - This script reads the reconstructed H3 analysis table produced by script 01.
# =============================================================================


# ── 1. Setup ──────────────────────────────────────────────────────────────────

required_packages <- c(
  "readr",
  "dplyr",
  "purrr",
  "tibble",
  "ggplot2",
  "lme4",
  "lmerTest",
  "performance"
)

missing_packages <- required_packages[
  !vapply(
    required_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing_packages) > 0L) {
  stop(
    "Install the following packages before running the script: ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

invisible(
  lapply(
    required_packages,
    library,
    character.only = TRUE
  )
)

set.seed(42)

options(
  stringsAsFactors = FALSE,
  contrasts = c("contr.treatment", "contr.poly")
)


# ── 2. Configuration ──────────────────────────────────────────────────────────

# Supply the project root as the first command-line argument, set H3_BASE_DIR,
# or run the script from the project root.
command_args <- commandArgs(trailingOnly = TRUE)

base_dir <- if (length(command_args) >= 1L) {
  command_args[[1]]
} else {
  Sys.getenv("H3_BASE_DIR", unset = getwd())
}

base_dir <- normalizePath(
  base_dir,
  winslash = "/",
  mustWork = FALSE
)

# Replace this placeholder after assigning the final supplementary-table number.
table_reference <- Sys.getenv(
  "H3_LOSO_TABLE_REFERENCE",
  unset = "Table SXX"
)

confidence_level <- 0.95
singularity_tolerance <- 1e-4
plot_dpi <- 600

model_order <- c(
  "CNN",
  "ResNet",
  "HYDRA",
  "MultiRocket",
  "RF",
  "LGBM"
)

# TRUE preserves the original analysis: trait distance is re-standardized after
# each species exclusion. FALSE retains the full-data scaling in all LOSO fits,
# which makes coefficient magnitudes directly comparable to the primary model.
restandardize_loso_distance <- TRUE


# ── 3. Paths and output structure ─────────────────────────────────────────────

input_file <- file.path(
  base_dir,
  "Output_R",
  "H3_mixed_effects_primary",
  "csv",
  "04_h3_mixed_model_data.csv"
)

output_r_root <- file.path(base_dir, "Output_R")
out_dir <- file.path(output_r_root, "H3_LOSO_R2_summary")
csv_dir <- file.path(out_dir, "csv")
plots_dir <- file.path(out_dir, "plots")
txt_dir <- file.path(out_dir, "txt")

for (directory in c(
  output_r_root,
  out_dir,
  csv_dir,
  plots_dir,
  txt_dir
)) {
  dir.create(
    directory,
    recursive = TRUE,
    showWarnings = FALSE
  )
}

if (!file.exists(input_file)) {
  stop(
    "Input file not found:\n",
    input_file,
    "\nRun 01_H3_mixed_effects_primary_final.R first.",
    call. = FALSE
  )
}


# ── 4. General helpers ────────────────────────────────────────────────────────

print_section <- function(title) {
  cat(
    "\n",
    strrep("=", 78),
    "\n",
    title,
    "\n",
    strrep("=", 78),
    "\n",
    sep = ""
  )
}

fmt_num <- function(x, digits = 3) {
  ifelse(
    is.na(x),
    "NA",
    formatC(x, format = "f", digits = digits)
  )
}

fmt_p <- function(x) {
  ifelse(
    is.na(x),
    "NA",
    ifelse(
      x < 0.001,
      "< 0.001",
      sprintf("%.3f", x)
    )
  )
}

write_txt <- function(path, lines) {
  writeLines(
    enc2utf8(lines),
    con = path,
    useBytes = TRUE
  )
}


# ── 5. Statistical helpers ────────────────────────────────────────────────────

# Fit an lmer model while retaining errors and warnings for diagnostics.
safe_lmer <- function(
    formula_object,
    data,
    reml,
    label
) {
  warning_messages <- character()
  
  fit <- withCallingHandlers(
    tryCatch(
      lmerTest::lmer(
        formula = formula_object,
        data = data,
        REML = reml,
        control = lme4::lmerControl(
          optimizer = "bobyqa",
          optCtrl = list(maxfun = 200000),
          check.conv.singular = "ignore"
        )
      ),
      error = function(error_condition) {
        structure(
          list(message = conditionMessage(error_condition)),
          class = "h3_model_error"
        )
      }
    ),
    warning = function(warning_condition) {
      warning_messages <<- c(
        warning_messages,
        conditionMessage(warning_condition)
      )
      
      invokeRestart("muffleWarning")
    }
  )
  
  if (inherits(fit, "h3_model_error")) {
    return(
      list(
        label = label,
        fit = NULL,
        error = fit$message,
        warnings = unique(warning_messages)
      )
    )
  }
  
  list(
    label = label,
    fit = fit,
    error = NA_character_,
    warnings = unique(warning_messages)
  )
}

get_convergence_messages <- function(fit) {
  if (is.null(fit)) {
    return(NA_character_)
  }
  
  messages <- fit@optinfo$conv$lme4$messages
  
  if (is.null(messages)) {
    return("")
  }
  
  paste(messages, collapse = " | ")
}

get_optimizer_convergence_code <- function(fit) {
  if (is.null(fit)) {
    return(NA_integer_)
  }
  
  convergence_code <- fit@optinfo$conv$opt
  
  if (is.null(convergence_code) || length(convergence_code) == 0L) {
    return(NA_integer_)
  }
  
  as.integer(convergence_code[[1]])
}

extract_model_diagnostics <- function(
    model_object,
    model_name,
    omitted_species = NA_character_
) {
  fit <- model_object$fit
  
  if (is.null(fit)) {
    return(
      tibble(
        model_name = model_name,
        omitted_species = omitted_species,
        reml = NA,
        n_observations = NA_integer_,
        singular = NA,
        optimizer_convergence_code = NA_integer_,
        convergence_messages = NA_character_,
        warnings = paste(model_object$warnings, collapse = " | "),
        error = model_object$error,
        converged_without_warnings = FALSE
      )
    )
  }
  
  optimizer_code <- get_optimizer_convergence_code(fit)
  convergence_messages <- get_convergence_messages(fit)
  warning_messages <- paste(model_object$warnings, collapse = " | ")
  error_message <- model_object$error
  
  converged_without_warnings <-
    (is.na(optimizer_code) || optimizer_code == 0L) &&
    (is.na(convergence_messages) || convergence_messages == "") &&
    (is.na(warning_messages) || warning_messages == "") &&
    (is.na(error_message) || error_message == "")
  
  tibble(
    model_name = model_name,
    omitted_species = omitted_species,
    reml = lme4::isREML(fit),
    n_observations = nobs(fit),
    singular = lme4::isSingular(
      fit,
      tol = singularity_tolerance
    ),
    optimizer_convergence_code = optimizer_code,
    convergence_messages = convergence_messages,
    warnings = warning_messages,
    error = error_message,
    converged_without_warnings = converged_without_warnings
  )
}

extract_r2 <- function(fit) {
  empty_result <- c(
    marginal = NA_real_,
    conditional = NA_real_
  )
  
  if (is.null(fit)) {
    return(empty_result)
  }
  
  r2_result <- tryCatch(
    performance::r2_nakagawa(
      fit,
      tolerance = 1e-8
    ),
    error = function(error_condition) NULL
  )
  
  if (is.null(r2_result)) {
    return(empty_result)
  }
  
  c(
    marginal = as.numeric(r2_result$R2_marginal),
    conditional = as.numeric(r2_result$R2_conditional)
  )
}

extract_trait_effect <- function(
    fit,
    conf_level = confidence_level
) {
  empty_result <- tibble(
    beta = NA_real_,
    std_error = NA_real_,
    df = NA_real_,
    statistic = NA_real_,
    p_value = NA_real_,
    ci_low = NA_real_,
    ci_high = NA_real_
  )
  
  if (is.null(fit)) {
    return(empty_result)
  }
  
  coefficient_table <- as.data.frame(
    coef(summary(fit))
  )
  
  term <- "z_trait_distance_core"
  
  if (!term %in% rownames(coefficient_table)) {
    return(empty_result)
  }
  
  estimate <- coefficient_table[term, "Estimate"]
  standard_error <- coefficient_table[term, "Std. Error"]
  
  degrees_freedom <- if ("df" %in% colnames(coefficient_table)) {
    coefficient_table[term, "df"]
  } else {
    NA_real_
  }
  
  statistic <- if ("t value" %in% colnames(coefficient_table)) {
    coefficient_table[term, "t value"]
  } else {
    NA_real_
  }
  
  p_value <- if ("Pr(>|t|)" %in% colnames(coefficient_table)) {
    coefficient_table[term, "Pr(>|t|)"]
  } else {
    NA_real_
  }
  
  critical_value <- if (is.finite(degrees_freedom)) {
    qt(
      (1 + conf_level) / 2,
      df = degrees_freedom
    )
  } else {
    qnorm((1 + conf_level) / 2)
  }
  
  tibble(
    beta = estimate,
    std_error = standard_error,
    df = degrees_freedom,
    statistic = statistic,
    p_value = p_value,
    ci_low = estimate - critical_value * standard_error,
    ci_high = estimate + critical_value * standard_error
  )
}

compare_nested_models <- function(
    reduced_fit,
    full_fit
) {
  empty_result <- tibble(
    df_difference = NA_real_,
    likelihood_ratio_chisq = NA_real_,
    p_value = NA_real_,
    aic_reduced = NA_real_,
    aic_full = NA_real_
  )
  
  if (is.null(reduced_fit) || is.null(full_fit)) {
    return(empty_result)
  }
  
  anova_table <- suppressMessages(
    stats::anova(reduced_fit, full_fit)
  ) %>%
    as.data.frame()
  
  required_columns <- c("Df", "Chisq", "Pr(>Chisq)")
  missing_columns <- setdiff(required_columns, names(anova_table))
  
  if (length(missing_columns) > 0L) {
    stop(
      "Nested-model comparison is missing: ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }
  
  comparison_row <- nrow(anova_table)
  
  tibble(
    df_difference = as.numeric(
      anova_table[comparison_row, "Df"]
    ),
    likelihood_ratio_chisq = as.numeric(
      anova_table[comparison_row, "Chisq"]
    ),
    p_value = as.numeric(
      anova_table[comparison_row, "Pr(>Chisq)"]
    ),
    aic_reduced = AIC(reduced_fit),
    aic_full = AIC(full_fit)
  )
}

standardize_distance <- function(
    distance,
    reference_mean,
    reference_sd
) {
  if (!is.finite(reference_sd) || reference_sd <= 0) {
    stop(
      "Trait-distance standard deviation must be positive and finite.",
      call. = FALSE
    )
  }
  
  (distance - reference_mean) / reference_sd
}


# ── 6. Load and validate the primary H3 data ──────────────────────────────────

print_section("1. LOAD AND VALIDATE H3 DATA")

h3_raw <- readr::read_csv(
  input_file,
  show_col_types = FALSE
)

required_columns <- c(
  "model",
  "train_species",
  "test_species",
  "ordered_species_pair",
  "unordered_species_pair",
  "trait_distance_core",
  "mean_macro_f1"
)

missing_columns <- setdiff(
  required_columns,
  names(h3_raw)
)

if (length(missing_columns) > 0L) {
  stop(
    "Input data are missing: ",
    paste(missing_columns, collapse = ", "),
    call. = FALSE
  )
}

if (
  any(!is.finite(h3_raw$trait_distance_core)) ||
  any(!is.finite(h3_raw$mean_macro_f1))
) {
  stop(
    "Input data contain non-finite trait-distance or Macro-F1 values.",
    call. = FALSE
  )
}

full_distance_mean <- mean(h3_raw$trait_distance_core)
full_distance_sd <- sd(h3_raw$trait_distance_core)

h3 <- h3_raw %>%
  mutate(
    model = factor(model, levels = model_order),
    train_species = factor(train_species),
    test_species = factor(test_species),
    ordered_species_pair = factor(ordered_species_pair),
    unordered_species_pair = factor(unordered_species_pair),
    z_trait_distance_core = standardize_distance(
      trait_distance_core,
      reference_mean = full_distance_mean,
      reference_sd = full_distance_sd
    )
  )

study_species <- sort(
  unique(
    c(
      as.character(h3$train_species),
      as.character(h3$test_species)
    )
  )
)

structure_summary <- tibble(
  quantity = c(
    "model_specific_observations",
    "classifiers",
    "species",
    "directed_species_pairs",
    "unordered_species_pairs",
    "full_distance_mean",
    "full_distance_sd"
  ),
  value = c(
    nrow(h3),
    n_distinct(h3$model),
    length(study_species),
    n_distinct(h3$ordered_species_pair),
    n_distinct(h3$unordered_species_pair),
    full_distance_mean,
    full_distance_sd
  )
)

expected_structure <- c(
  model_specific_observations = 432,
  classifiers = 6,
  species = 9,
  directed_species_pairs = 72,
  unordered_species_pairs = 36
)

observed_structure <- setNames(
  structure_summary$value[
    structure_summary$quantity %in% names(expected_structure)
  ],
  structure_summary$quantity[
    structure_summary$quantity %in% names(expected_structure)
  ]
)

if (
  !all(
    observed_structure[names(expected_structure)] == expected_structure
  )
) {
  write_csv(
    structure_summary,
    file.path(csv_dir, "00_input_structure_summary.csv")
  )
  
  stop(
    "Unexpected H3 data structure. See 00_input_structure_summary.csv.",
    call. = FALSE
  )
}

write_csv(
  structure_summary,
  file.path(csv_dir, "00_input_structure_summary.csv")
)


# ── 7. Define the primary models ──────────────────────────────────────────────

# The random-effects structure matches the predefined primary H3 analysis.
formula_null <- mean_macro_f1 ~ model +
  (1 | unordered_species_pair) +
  (1 | ordered_species_pair) +
  (1 | train_species) +
  (1 | test_species)

formula_additive <- mean_macro_f1 ~ z_trait_distance_core + model +
  (1 | unordered_species_pair) +
  (1 | ordered_species_pair) +
  (1 | train_species) +
  (1 | test_species)


# ── 8. Primary-model R2 and trait effect ──────────────────────────────────────

print_section("2. PRIMARY-MODEL R2 AND TRAIT EFFECT")

primary_null_ml <- safe_lmer(
  formula_object = formula_null,
  data = h3,
  reml = FALSE,
  label = "primary_null_ml"
)

primary_additive_ml <- safe_lmer(
  formula_object = formula_additive,
  data = h3,
  reml = FALSE,
  label = "primary_additive_ml"
)

primary_additive_reml <- safe_lmer(
  formula_object = formula_additive,
  data = h3,
  reml = TRUE,
  label = "primary_additive_reml"
)

primary_model_store <- list(
  primary_null_ml = primary_null_ml,
  primary_additive_ml = primary_additive_ml,
  primary_additive_reml = primary_additive_reml
)

primary_diagnostics <- purrr::imap_dfr(
  primary_model_store,
  ~ extract_model_diagnostics(
    model_object = .x,
    model_name = .y
  )
)

write_csv(
  primary_diagnostics,
  file.path(csv_dir, "01_primary_model_diagnostics.csv")
)

r2_null <- extract_r2(primary_null_ml$fit)
r2_additive <- extract_r2(primary_additive_ml$fit)

delta_marginal_r2 <-
  r2_additive[["marginal"]] - r2_null[["marginal"]]

primary_r2 <- tibble(
  model_name = c(
    "primary_null_ml",
    "primary_additive_ml"
  ),
  marginal_r2 = c(
    r2_null[["marginal"]],
    r2_additive[["marginal"]]
  ),
  conditional_r2 = c(
    r2_null[["conditional"]],
    r2_additive[["conditional"]]
  ),
  delta_marginal_r2 = c(
    NA_real_,
    delta_marginal_r2
  )
)

primary_effect <- extract_trait_effect(
  primary_additive_reml$fit
)

primary_lrt <- compare_nested_models(
  reduced_fit = primary_null_ml$fit,
  full_fit = primary_additive_ml$fit
)

write_csv(
  primary_r2,
  file.path(csv_dir, "02_primary_model_r2.csv")
)

write_csv(
  primary_effect,
  file.path(csv_dir, "03_primary_trait_effect.csv")
)

write_csv(
  primary_lrt,
  file.path(csv_dir, "04_primary_likelihood_ratio_test.csv")
)


# ── 9. Leave-one-species-out analysis ─────────────────────────────────────────

print_section("3. LEAVE-ONE-SPECIES-OUT ANALYSIS")

# Each fit excludes all directed pairs in which the omitted species appears as
# either source or target. The additive model is then refitted with the same
# random-effects structure used in the primary analysis.
fit_loso <- function(omitted_species) {
  loso_data <- h3 %>%
    filter(
      as.character(train_species) != omitted_species,
      as.character(test_species) != omitted_species
    ) %>%
    droplevels()
  
  if (restandardize_loso_distance) {
    loso_distance_mean <- mean(loso_data$trait_distance_core)
    loso_distance_sd <- sd(loso_data$trait_distance_core)
  } else {
    loso_distance_mean <- full_distance_mean
    loso_distance_sd <- full_distance_sd
  }
  
  loso_data <- loso_data %>%
    mutate(
      z_trait_distance_core = standardize_distance(
        trait_distance_core,
        reference_mean = loso_distance_mean,
        reference_sd = loso_distance_sd
      )
    )
  
  null_ml <- safe_lmer(
    formula_object = formula_null,
    data = loso_data,
    reml = FALSE,
    label = "loso_null_ml"
  )
  
  additive_ml <- safe_lmer(
    formula_object = formula_additive,
    data = loso_data,
    reml = FALSE,
    label = "loso_additive_ml"
  )
  
  additive_reml <- safe_lmer(
    formula_object = formula_additive,
    data = loso_data,
    reml = TRUE,
    label = "loso_additive_reml"
  )
  
  effect <- extract_trait_effect(additive_reml$fit)
  
  likelihood_ratio_test <- compare_nested_models(
    reduced_fit = null_ml$fit,
    full_fit = additive_ml$fit
  )
  
  diagnostic <- extract_model_diagnostics(
    model_object = additive_reml,
    model_name = "loso_additive_reml",
    omitted_species = omitted_species
  )
  
  tibble(
    omitted_species = omitted_species,
    n_observations = nrow(loso_data),
    n_directed_pairs = n_distinct(loso_data$ordered_species_pair),
    n_unordered_pairs = n_distinct(loso_data$unordered_species_pair),
    distance_mean_used = loso_distance_mean,
    distance_sd_used = loso_distance_sd,
    beta = effect$beta,
    std_error = effect$std_error,
    df = effect$df,
    statistic = effect$statistic,
    ci_low = effect$ci_low,
    ci_high = effect$ci_high,
    ci_includes_zero = effect$ci_low <= 0 & effect$ci_high >= 0,
    p_value_coefficient = effect$p_value,
    likelihood_ratio_chisq = likelihood_ratio_test$likelihood_ratio_chisq,
    df_difference = likelihood_ratio_test$df_difference,
    p_value_lrt = likelihood_ratio_test$p_value,
    singular = diagnostic$singular,
    optimizer_convergence_code = diagnostic$optimizer_convergence_code,
    convergence_messages = diagnostic$convergence_messages,
    warnings = diagnostic$warnings,
    error = diagnostic$error,
    converged_without_warnings = diagnostic$converged_without_warnings,
    valid_model = diagnostic$converged_without_warnings &
      !diagnostic$singular
  )
}

loso_results <- map_dfr(
  study_species,
  fit_loso
)

write_csv(
  loso_results,
  file.path(csv_dir, "05_leave_one_species_out_estimates.csv")
)

valid_loso <- loso_results %>%
  filter(
    valid_model,
    is.finite(beta),
    is.finite(ci_low),
    is.finite(ci_high)
  )

if (nrow(valid_loso) == 0L) {
  stop(
    "No valid LOSO estimates. Inspect 05_leave_one_species_out_estimates.csv.",
    call. = FALSE
  )
}

if (nrow(valid_loso) < length(study_species)) {
  warning(
    length(study_species) - nrow(valid_loso),
    " LOSO model(s) were excluded from the summary because of fitting ",
    "errors, warnings, or singularity.",
    call. = FALSE
  )
}

loso_summary <- tibble(
  n_species_exclusions_expected = length(study_species),
  n_species_exclusions_valid = nrow(valid_loso),
  beta_min = min(valid_loso$beta),
  beta_min_omitted_species = valid_loso$omitted_species[
    which.min(valid_loso$beta)
  ],
  beta_max = max(valid_loso$beta),
  beta_max_omitted_species = valid_loso$omitted_species[
    which.max(valid_loso$beta)
  ],
  all_estimates_negative = all(valid_loso$beta < 0),
  all_cis_include_zero = all(valid_loso$ci_includes_zero),
  n_cis_including_zero = sum(valid_loso$ci_includes_zero),
  n_cis_excluding_zero = sum(!valid_loso$ci_includes_zero),
  restandardized_within_each_loso_dataset = restandardize_loso_distance
)

write_csv(
  loso_summary,
  file.path(csv_dir, "06_leave_one_species_out_summary.csv")
)


# ── 10. Manuscript-ready text ─────────────────────────────────────────────────

print_section("4. MANUSCRIPT-READY TEXT")

r2_sentence <- paste0(
  "The primary additive model had a marginal R² of ",
  fmt_num(r2_additive[["marginal"]]),
  " and a conditional R² of ",
  fmt_num(r2_additive[["conditional"]]),
  ". Adding trait distance increased marginal R² by ΔR² = ",
  fmt_num(delta_marginal_r2),
  "."
)

if (
  loso_summary$all_estimates_negative &&
  loso_summary$all_cis_include_zero
) {
  loso_sentence <- paste0(
    "Leave-one-species-out estimates ranged from β = ",
    fmt_num(loso_summary$beta_min),
    " to β = ",
    fmt_num(loso_summary$beta_max),
    ". The estimated association remained negative across all species ",
    "exclusions, although every confidence interval included zero (",
    table_reference,
    ")."
  )
} else if (loso_summary$all_estimates_negative) {
  loso_sentence <- paste0(
    "Leave-one-species-out estimates ranged from β = ",
    fmt_num(loso_summary$beta_min),
    " to β = ",
    fmt_num(loso_summary$beta_max),
    ". The estimated association remained negative across all species ",
    "exclusions; ",
    loso_summary$n_cis_excluding_zero,
    " of ",
    loso_summary$n_species_exclusions_valid,
    " confidence intervals excluded zero (",
    table_reference,
    ")."
  )
} else {
  loso_sentence <- paste0(
    "Leave-one-species-out estimates ranged from β = ",
    fmt_num(loso_summary$beta_min),
    " to β = ",
    fmt_num(loso_summary$beta_max),
    ". The estimated association did not retain a consistent negative ",
    "direction across species exclusions (",
    table_reference,
    ")."
  )
}

write_txt(
  file.path(txt_dir, "01_manuscript_ready_values.txt"),
  c(
    "R2 SENTENCE",
    r2_sentence,
    "",
    "LOSO SENTENCE",
    loso_sentence,
    "",
    "DETAILS",
    paste0(
      "Minimum LOSO estimate: β = ",
      fmt_num(loso_summary$beta_min),
      " after omitting ",
      loso_summary$beta_min_omitted_species,
      "."
    ),
    paste0(
      "Maximum LOSO estimate: β = ",
      fmt_num(loso_summary$beta_max),
      " after omitting ",
      loso_summary$beta_max_omitted_species,
      "."
    ),
    paste0(
      "Confidence intervals including zero: ",
      loso_summary$n_cis_including_zero,
      " of ",
      loso_summary$n_species_exclusions_valid,
      "."
    ),
    paste0(
      "Primary REML coefficient: β = ",
      fmt_num(primary_effect$beta),
      ", 95% CI [",
      fmt_num(primary_effect$ci_low),
      ", ",
      fmt_num(primary_effect$ci_high),
      "], p = ",
      fmt_p(primary_effect$p_value),
      "."
    ),
    paste0(
      "Primary likelihood-ratio test: χ²(",
      fmt_num(primary_lrt$df_difference, 0),
      ") = ",
      fmt_num(primary_lrt$likelihood_ratio_chisq, 2),
      ", p = ",
      fmt_p(primary_lrt$p_value),
      "."
    ),
    paste0(
      "LOSO distance re-standardized within each reduced dataset: ",
      restandardize_loso_distance,
      "."
    )
  )
)


# ── 11. LOSO forest plot ──────────────────────────────────────────────────────

print_section("5. LOSO FOREST PLOT")

plot_data <- valid_loso %>%
  arrange(beta) %>%
  mutate(
    omitted_species = factor(
      omitted_species,
      levels = omitted_species
    )
  )

plot_loso <- ggplot(
  plot_data,
  aes(
    x = beta,
    y = omitted_species
  )
) +
  geom_vline(
    xintercept = 0,
    linetype = 2,
    colour = "grey40",
    linewidth = 0.4
  ) +
  geom_segment(
    aes(
      x = ci_low,
      xend = ci_high,
      yend = omitted_species
    ),
    colour = "grey25",
    linewidth = 0.6
  ) +
  geom_point(
    shape = 21,
    fill = "grey55",
    colour = "black",
    stroke = 0.35,
    size = 2.4
  ) +
  labs(
    x = "Trait-distance coefficient per SD increase",
    y = "Omitted species"
  ) +
  theme_bw(base_size = 11) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text = element_text(colour = "black"),
    axis.title = element_text(colour = "black")
  )

ggsave(
  filename = file.path(
    plots_dir,
    "01_leave_one_species_out_forest_plot.png"
  ),
  plot = plot_loso,
  width = 7.2,
  height = 5.2,
  units = "in",
  dpi = plot_dpi,
  bg = "white"
)


# ── 12. Reproducibility exports ───────────────────────────────────────────────

print_section("6. REPRODUCIBILITY EXPORTS")

formula_registry <- tibble(
  analysis = c(
    "null_model",
    "additive_model"
  ),
  formula = c(
    paste(deparse(formula_null), collapse = " "),
    paste(deparse(formula_additive), collapse = " ")
  )
)

run_manifest <- tibble(
  item = c(
    "script",
    "run_time",
    "base_dir",
    "input_file",
    "output_dir",
    "random_seed",
    "confidence_level",
    "singularity_tolerance",
    "table_reference",
    "restandardize_loso_distance",
    "full_distance_mean",
    "full_distance_sd"
  ),
  value = c(
    "03_H3_LOSO_R2_summary_final.R",
    as.character(Sys.time()),
    base_dir,
    input_file,
    out_dir,
    "42",
    as.character(confidence_level),
    as.character(singularity_tolerance),
    table_reference,
    as.character(restandardize_loso_distance),
    as.character(full_distance_mean),
    as.character(full_distance_sd)
  )
)

write_csv(
  formula_registry,
  file.path(csv_dir, "07_model_formula_registry.csv")
)

write_csv(
  run_manifest,
  file.path(csv_dir, "08_run_manifest.csv")
)

write_txt(
  file.path(txt_dir, "02_primary_model_summaries.txt"),
  c(
    "PRIMARY NULL ML MODEL",
    capture.output(summary(primary_null_ml$fit)),
    "",
    "PRIMARY ADDITIVE ML MODEL",
    capture.output(summary(primary_additive_ml$fit)),
    "",
    "PRIMARY ADDITIVE REML MODEL",
    capture.output(summary(primary_additive_reml$fit))
  )
)

write_txt(
  file.path(txt_dir, "03_session_info.txt"),
  capture.output(sessionInfo())
)


# ── 13. Console summary ───────────────────────────────────────────────────────

print_section("7. FINAL CONSOLE SUMMARY")

cat(r2_sentence, "\n\n")
cat(loso_sentence, "\n\n")
cat(
  "Valid LOSO models: ",
  loso_summary$n_species_exclusions_valid,
  " of ",
  loso_summary$n_species_exclusions_expected,
  "\n",
  sep = ""
)
cat(
  "LOSO distance re-standardized within each reduced dataset: ",
  restandardize_loso_distance,
  "\n\n",
  sep = ""
)
cat(
  "Outputs written to:\n",
  "- ", csv_dir, "\n",
  "- ", plots_dir, "\n",
  "- ", txt_dir, "\n",
  sep = ""
)

# End of script
