# =============================================================================
# 04_H3_sensitivity_paper.R
# =============================================================================
#
# Purpose:
# - Pre-specified H3 sensitivity analyses S1-S3 (see ANALYSIS_PLAN.md).
# - The primary model is refitted first as a consistency check against the
#   thesis (standardized beta = -0.031, LR chi2 = 2.54, p = 0.111).
# - S1: within-dataset Macro-F1 (LOIO) of source and target as covariates.
# - S2: target-normalized retention = pairwise Macro-F1 / within Macro-F1 of
#   the target dataset (log retention as additional check).
# - S3: Gower distance with gawdis-balanced trait weights.
#
# Inputs:
# - Models/<model>/Pairwise_<model>/statistics/metrics_all.csv
# - Output_checks/Within_LOIO/<model>/statistics/metrics_all_incl_partial.csv
#   (from Python/within_loio.py)
# - species_traits_core_h3_final.csv (Traits_expert_coded/)
#
# Outputs: Output_checks/H3_sensitivity/{csv,txt}
# =============================================================================


# ── 1. Setup ──────────────────────────────────────────────────────────────────

required_packages <- c(
  "readr", "dplyr", "tidyr", "purrr", "tibble", "stringr",
  "cluster", "lme4", "lmerTest", "performance", "gawdis"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
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
    setdiff(required_packages, "gawdis"),
    library,
    character.only = TRUE
  )
)

options(
  stringsAsFactors = FALSE,
  contrasts = c("contr.treatment", "contr.poly")
)


# ── 2. Settings ───────────────────────────────────────────────────────────────

base_dir <- "/Volumes/Z Slim/11_05_2026_Data_Analysis"

models_dir <- file.path(base_dir, "Models")
loio_dir   <- file.path(base_dir, "Output_checks", "Within_LOIO")
out_dir    <- file.path(base_dir, "Output_checks", "H3_sensitivity")
csv_dir    <- file.path(out_dir, "csv")
txt_dir    <- file.path(out_dir, "txt")

# The trait file is searched in these folders, in this order.
trait_filename <- "species_traits_core_h3_final.csv"
trait_dir_candidates <- c(
  getwd(),
  file.path(base_dir, "R_checks"),
  file.path(base_dir, "Traits_expert_coded"),
  file.path(base_dir, "data", "traits")
)

model_order <- c("CNN", "ResNet", "HYDRA", "MultiRocket", "RF", "LGBM")
confidence_level <- 0.95
singularity_tolerance <- 1e-4

# S3 (ANALYSIS_PLAN.md, amendment of 2026-10-13).
gawdis_seed <- 42
gawdis_maxiter <- 300

# Thesis values of the primary model, used for the consistency check.
thesis_primary <- c(beta = -0.031, lr_chisq = 2.54, p_value = 0.111)

for (dir_path in c(out_dir, csv_dir, txt_dir)) {
  dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
}


# ── 3. Helpers ────────────────────────────────────────────────────────────────

print_section <- function(title) {
  cat("\n", strrep("=", 78), "\n", title, "\n", strrep("=", 78), "\n", sep = "")
}

standardize <- function(x) {
  (x - mean(x)) / sd(x)
}

lmer_control <- lmerControl(
  optimizer = "bobyqa",
  optCtrl = list(maxfun = 200000),
  check.conv.singular = "ignore"
)

random_effects <- paste(
  "(1 | unordered_species_pair)",
  "(1 | ordered_species_pair)",
  "(1 | train_species)",
  "(1 | test_species)",
  sep = " + "
)

fit_lmer <- function(response, fixed, data, reml) {
  warnings_seen <- character()
  fit <- withCallingHandlers(
    lmerTest::lmer(
      as.formula(paste(response, "~", fixed, "+", random_effects)),
      data = data,
      REML = reml,
      control = lmer_control
    ),
    warning = function(w) {
      warnings_seen <<- c(warnings_seen, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  attr(fit, "fit_warnings") <- unique(warnings_seen)
  fit
}

convergence_note <- function(fit) {
  messages <- c(fit@optinfo$conv$lme4$messages, attr(fit, "fit_warnings"))
  if (length(messages) == 0L) "" else paste(unique(messages), collapse = " | ")
}

# ML fits for the likelihood-ratio test, REML fit for the coefficient
# (Satterthwaite df), as in 01_H3_mixed_effects.R.
run_analysis <- function(analysis, response, fixed_null, fixed_full, term, data) {
  null_ml <- fit_lmer(response, fixed_null, data, reml = FALSE)
  full_ml <- fit_lmer(response, fixed_full, data, reml = FALSE)
  full_reml <- fit_lmer(response, fixed_full, data, reml = TRUE)

  lr <- suppressMessages(anova(null_ml, full_ml))
  coefficients <- coef(summary(full_reml))
  critical <- qt((1 + confidence_level) / 2, df = coefficients[term, "df"])

  result <- tibble(
    analysis = analysis,
    response = response,
    term = term,
    n_observations = nobs(full_reml),
    estimate = coefficients[term, "Estimate"],
    std_error = coefficients[term, "Std. Error"],
    df = coefficients[term, "df"],
    ci_low = estimate - critical * std_error,
    ci_high = estimate + critical * std_error,
    lr_chisq = lr$Chisq[2],
    lr_df = lr$Df[2],
    lr_p_value = lr$`Pr(>Chisq)`[2],
    singular_reml = isSingular(full_reml, tol = singularity_tolerance),
    convergence_notes = paste(
      c(convergence_note(null_ml), convergence_note(full_ml), convergence_note(full_reml)),
      collapse = " || "
    )
  )

  list(result = result, full_reml = full_reml)
}

fmt <- function(x, digits = 3) formatC(x, format = "f", digits = digits)


# ── 4. Load pairwise transfer results ─────────────────────────────────────────

print_section("1. PAIRWISE TRANSFER RESULTS")

dataset_to_species <- c(
  "Bison_dataset_1" = "Bison bonasus",
  "Cattle_dataset_1" = "Bos taurus",
  "Dog_dataset_1" = "Canis lupus familiaris",
  "Dog_dataset_2" = "Canis lupus familiaris",
  "Fox_dataset_1" = "Vulpes vulpes",
  "Fox_dataset_2" = "Vulpes vulpes",
  "Giraffe_dataset_1" = "Giraffa camelopardalis",
  "Hedgehog_dataset_1" = "Erinaceus europaeus",
  "Horse_dataset_1" = "Equus ferus przewalskii",
  "Horse_dataset_2" = "Equus caballus",
  "Raccoon_dataset_1" = "Procyon lotor",
  "Raccoon_dataset_2" = "Procyon lotor"
)

pairwise_file <- function(model) {
  file.path(models_dir, model, paste0("Pairwise_", model), "statistics", "metrics_all.csv")
}

missing_pairwise <- model_order[!file.exists(vapply(model_order, pairwise_file, character(1)))]
if (length(missing_pairwise) > 0L) {
  stop("Missing pairwise metrics for: ", paste(missing_pairwise, collapse = ", "), call. = FALSE)
}

pairwise <- map_dfr(
  model_order,
  ~ read_csv(pairwise_file(.x), show_col_types = FALSE) %>%
    transmute(
      model = .x,
      train_dataset = str_trim(train_dataset),
      test_dataset = str_trim(test_dataset),
      macro_f1 = as.numeric(macro_f1)
    )
) %>%
  mutate(
    train_species = unname(dataset_to_species[train_dataset]),
    test_species = unname(dataset_to_species[test_dataset])
  )

if (any(is.na(pairwise$train_species) | is.na(pairwise$test_species))) {
  stop("Unmapped dataset names in the pairwise files.", call. = FALSE)
}

pairwise <- pairwise %>% filter(train_species != test_species)
cat("Cross-species dataset pairs (all models):", nrow(pairwise), "\n")


# ── 5. Load LOIO within-dataset results ───────────────────────────────────────

print_section("2. LOIO WITHIN-DATASET RESULTS")

# Datasets without complete LOIO coverage use their partial value, so all 432
# species-pair records are retained (ANALYSIS_PLAN.md, section 3).
loio_file <- function(model) {
  file.path(loio_dir, model, "statistics", "metrics_all_incl_partial.csv")
}

missing_loio <- model_order[!file.exists(vapply(model_order, loio_file, character(1)))]
if (length(missing_loio) > 0L) {
  stop(
    "Missing LOIO results for: ", paste(missing_loio, collapse = ", "),
    ". Run Python/within_loio.py first.",
    call. = FALSE
  )
}

within_loio <- map_dfr(
  model_order,
  ~ read_csv(loio_file(.x), show_col_types = FALSE) %>%
    transmute(
      model = .x,
      dataset = analysis_dataset,
      within_macro_f1 = as.numeric(macro_f1),
      coverage = if ("coverage" %in% names(.)) as.numeric(coverage) else NA_real_
    )
)

missing_within <- pairwise %>%
  select(model, dataset = train_dataset) %>%
  bind_rows(pairwise %>% select(model, dataset = test_dataset)) %>%
  distinct() %>%
  anti_join(within_loio, by = c("model", "dataset"))

if (nrow(missing_within) > 0L) {
  write_csv(missing_within, file.path(csv_dir, "error_missing_within_values.csv"))
  stop("Within-dataset values missing for some datasets. See error_missing_within_values.csv.", call. = FALSE)
}

if (any(within_loio$within_macro_f1 <= 0)) {
  stop("Within-dataset Macro-F1 of 0 found; retention cannot be computed.", call. = FALSE)
}

print(within_loio %>% filter(coverage < 1), n = Inf)


# ── 6. Trait distances: primary Gower and gawdis ──────────────────────────────

print_section("3. TRAIT DISTANCES")

trait_file <- file.path(trait_dir_candidates, trait_filename)
trait_file <- trait_file[file.exists(trait_file)][1]
if (is.na(trait_file)) {
  stop("Trait file not found: ", trait_filename, call. = FALSE)
}
cat("Trait file:", trait_file, "\n")

ordinal_levels <- c("low", "moderate", "high", "very_high")

# Same trait typing as 01_H3_mixed_effects.R.
traits_raw <- read_csv(trait_file, show_col_types = FALSE) %>%
  mutate(across(where(is.character), str_trim))

traits <- traits_raw %>%
  transmute(
    log_body_mass_g = as.numeric(log_body_mass_g),
    dominant_locomotor_mode = as.factor(dominant_locomotor_mode),
    body_configuration_type = as.factor(body_configuration_type),
    expected_trunk_motion_amplitude =
      ordered(str_to_lower(expected_trunk_motion_amplitude), levels = ordinal_levels),
    expected_directional_change_tendency =
      ordered(str_to_lower(expected_directional_change_tendency), levels = ordinal_levels),
    expected_head_neck_contribution_to_acceleration_relevant_motion =
      ordered(str_to_lower(expected_head_neck_contribution_to_acceleration_relevant_motion), levels = ordinal_levels),
    dominant_feeding_foraging_mode = as.factor(dominant_feeding_foraging_mode)
  ) %>%
  as.data.frame()
rownames(traits) <- traits_raw$species

if (any(!complete.cases(traits))) {
  stop("Missing or invalid trait values.", call. = FALSE)
}

primary_gower <- as.matrix(cluster::daisy(traits, metric = "gower"))

# Check: equal weights must reproduce the primary distance.
gawdis_equal <- gawdis::gawdis(traits, w.type = "equal", ord = "classic", silent = TRUE)
equal_check <- max(abs(as.matrix(gawdis_equal) - primary_gower))
cat("gawdis (equal weights) vs primary Gower, max abs difference:", format(equal_check, digits = 3), "\n")
if (equal_check > 1e-8) {
  stop("gawdis with equal weights does not reproduce the primary Gower distance.", call. = FALSE)
}

gawdis_analytic <- suppressWarnings(
  gawdis::gawdis(traits, w.type = "analytic", ord = "classic", silent = TRUE)
)

set.seed(gawdis_seed)
gawdis_optimized <- gawdis::gawdis(
  traits,
  w.type = "optimized",
  ord = "classic",
  opti.maxiter = gawdis_maxiter,
  silent = TRUE
)
gawdis_matrix <- as.matrix(gawdis_optimized)
dimnames(gawdis_matrix) <- dimnames(primary_gower)

trait_weights <- tibble(
  trait = names(traits),
  weight_equal = as.numeric(attr(gawdis_equal, "weights")),
  correlation_equal = as.numeric(attr(gawdis_equal, "correls")),
  weight_analytic = as.numeric(attr(gawdis_analytic, "weights")),
  weight_optimized = as.numeric(attr(gawdis_optimized, "weights")),
  correlation_optimized = as.numeric(attr(gawdis_optimized, "correls"))
)

print(trait_weights, n = Inf, width = Inf)
cat(
  "Correlation of gawdis and primary Gower distance (36 species pairs):",
  fmt(cor(as.numeric(as.dist(gawdis_matrix)), as.numeric(as.dist(primary_gower)))),
  "\n"
)


# ── 7. Build the 432-row sensitivity table ────────────────────────────────────

print_section("4. SENSITIVITY TABLE")

dataset_pairs <- pairwise %>%
  left_join(
    within_loio %>% select(model, train_dataset = dataset, source_within = within_macro_f1),
    by = c("model", "train_dataset")
  ) %>%
  left_join(
    within_loio %>% select(model, test_dataset = dataset, target_within = within_macro_f1),
    by = c("model", "test_dataset")
  ) %>%
  mutate(retention = macro_f1 / target_within)

# Averaged over dataset pairs within each classifier and species pair, as the
# primary response in 01_H3_mixed_effects.R.
h3 <- dataset_pairs %>%
  group_by(model, train_species, test_species) %>%
  summarise(
    mean_macro_f1 = mean(macro_f1),
    source_within = mean(source_within),
    target_within = mean(target_within),
    retention = mean(retention),
    n_dataset_pairs = n(),
    .groups = "drop"
  ) %>%
  mutate(
    log_retention = log(retention),
    trait_distance_core = map2_dbl(train_species, test_species, ~ primary_gower[.x, .y]),
    trait_distance_gawdis = map2_dbl(train_species, test_species, ~ gawdis_matrix[.x, .y]),
    ordered_species_pair = factor(paste(train_species, test_species, sep = "__")),
    unordered_species_pair = factor(map2_chr(
      train_species, test_species, ~ paste(sort(c(.x, .y)), collapse = "__")
    )),
    z_trait_distance_core = standardize(trait_distance_core),
    z_trait_distance_gawdis = standardize(trait_distance_gawdis),
    z_source_within = standardize(source_within),
    z_target_within = standardize(target_within),
    model = factor(model, levels = model_order),
    train_species = factor(train_species),
    test_species = factor(test_species)
  )

structure_check <- c(
  rows = nrow(h3),
  directed_pairs = n_distinct(h3$ordered_species_pair),
  unordered_pairs = n_distinct(h3$unordered_species_pair)
)
print(structure_check)
if (!all(structure_check == c(432, 72, 36))) {
  stop("Unexpected H3 structure (expected 432 rows, 72 directed and 36 unordered pairs).", call. = FALSE)
}

write_csv(h3, file.path(csv_dir, "01_h3_sensitivity_data.csv"))
write_csv(trait_weights, file.path(csv_dir, "02_gawdis_trait_weights.csv"))
write_csv(
  as_tibble(gawdis_matrix, rownames = "species"),
  file.path(csv_dir, "03_gawdis_distance_matrix.csv")
)


# ── 8. Fit primary model and S1-S3 ────────────────────────────────────────────

print_section("5. MIXED MODELS")

covariates <- "z_source_within + z_target_within"

analyses <- list(
  run_analysis(
    "primary (consistency check)", "mean_macro_f1",
    "model", "z_trait_distance_core + model",
    "z_trait_distance_core", h3
  ),
  run_analysis(
    "S1 within covariates", "mean_macro_f1",
    paste("model +", covariates), paste("z_trait_distance_core + model +", covariates),
    "z_trait_distance_core", h3
  ),
  run_analysis(
    "S2 retention", "retention",
    "model", "z_trait_distance_core + model",
    "z_trait_distance_core", h3
  ),
  run_analysis(
    "S2 log retention", "log_retention",
    "model", "z_trait_distance_core + model",
    "z_trait_distance_core", h3
  ),
  run_analysis(
    "S3 gawdis distance", "mean_macro_f1",
    "model", "z_trait_distance_gawdis + model",
    "z_trait_distance_gawdis", h3
  )
)

results <- map_dfr(analyses, "result")
print(results %>% select(analysis, estimate, std_error, ci_low, ci_high, lr_chisq, lr_p_value, singular_reml), width = Inf)
write_csv(results, file.path(csv_dir, "04_sensitivity_model_results.csv"))

# S1: all fixed effects, collinearity, and correlations with trait distance.
s1_fit <- analyses[[2]]$full_reml
s1_fixed <- as_tibble(coef(summary(s1_fit)), rownames = "term")
s1_collinearity <- as_tibble(performance::check_collinearity(s1_fit))
s1_correlations <- tibble(
  variable = c("source_within", "target_within"),
  r_with_trait_distance = c(
    cor(h3$source_within, h3$trait_distance_core),
    cor(h3$target_within, h3$trait_distance_core)
  )
)

write_csv(s1_fixed, file.path(csv_dir, "05_s1_fixed_effects.csv"))
write_csv(s1_collinearity, file.path(csv_dir, "06_s1_collinearity.csv"))
write_csv(s1_correlations, file.path(csv_dir, "07_s1_covariate_correlations.csv"))


# ── 9. Report ─────────────────────────────────────────────────────────────────

print_section("6. REPORT")

primary <- results %>% filter(analysis == "primary (consistency check)")
primary_matches <- abs(primary$estimate - thesis_primary[["beta"]]) < 0.0005 &&
  abs(primary$lr_chisq - thesis_primary[["lr_chisq"]]) < 0.005 &&
  abs(primary$lr_p_value - thesis_primary[["p_value"]]) < 0.0005

result_lines <- sprintf(
  "%-28s beta = %s [%s, %s], SE = %s, LR chi2(%d) = %s, p = %s%s",
  results$analysis,
  fmt(results$estimate), fmt(results$ci_low), fmt(results$ci_high), fmt(results$std_error),
  as.integer(results$lr_df), fmt(results$lr_chisq, 2), fmt(results$lr_p_value),
  ifelse(results$singular_reml, "  (singular fit)", "")
)

report <- c(
  "H3 sensitivity analyses (ANALYSIS_PLAN.md)",
  paste("Created on:", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  paste("Trait file:", trait_file),
  "",
  paste(
    "Consistency check of the primary model against the thesis:",
    if (primary_matches) "OK" else "MISMATCH - check inputs before interpreting S1-S3"
  ),
  "",
  "Trait-distance coefficients (standardized; S2 on the retention scale):",
  result_lines,
  "",
  "gawdis trait weights (S3):",
  capture.output(print(trait_weights, n = Inf, width = Inf)),
  "",
  "S1 fixed effects:",
  capture.output(print(s1_fixed, n = Inf, width = Inf)),
  "",
  "S1 collinearity:",
  capture.output(print(s1_collinearity, n = Inf, width = Inf)),
  "",
  "S1 correlations with core trait distance (432 rows):",
  capture.output(print(s1_correlations, width = Inf)),
  "",
  "Datasets with partial LOIO coverage (within values from the valid folds):",
  capture.output(print(within_loio %>% filter(coverage < 1), n = Inf, width = Inf))
)

writeLines(enc2utf8(report), file.path(txt_dir, "h3_sensitivity_report.txt"), useBytes = TRUE)
cat(report, sep = "\n")
cat("\nOutputs written to:", out_dir, "\n")
