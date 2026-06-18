# =============================================================================
# H3 sensitivity analysis: trait-distance robustness
# =============================================================================
# Author : Lucas Beseler
# Date   : 2026-06-17
# Cleaned: 2026-06-18
#
# Purpose:
# - Test whether the H3 trait-distance results are robust to alternative
#   functional-biomechanical trait specifications and aggregation levels.
# - Use the final split trait CSV structure with continuous log_body_mass_g.
# - Treat locomotor mode and feeding/foraging mode as separate traits.
# - Run leave-one-trait-out, leave-one-species-out, bootstrap CI, and
#   pseudoreplication checks for cross-species transfer performance.
# - Export publication-ready sensitivity tables, plots, and text reports for H3.
#
# Hypothesis:
# - H3: Cross-species transfer performance declines with increasing
#   functional-biomechanical trait distance between source and target species.
#
# =============================================================================

# ── 1. Setup ──────────────────────────────────────────────────────────────────
if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman", repos = "https://cloud.r-project.org")
}

pacman::p_load(
  readr, dplyr, tidyr, tibble, purrr, stringr,
  cluster, ggplot2, patchwork
)


# ── 2. User settings ──────────────────────────────────────────────────────────
set.seed(42)
n_boot <- 1000L


# ── 3. Paths ──────────────────────────────────────────────────────────────────
base_dir      <- "/Volumes/Z Slim/11_05_2026_Data_Analysis"
models_dir    <- file.path(base_dir, "Models")
output_r_root <- file.path(base_dir, "Output_R")

# H3 sensitivity uses the final manually curated trait CSVs.
# The script searches the working directory and project H3 folders first
# and stops if the final CSVs are missing.
traits_dir_candidates <- c(
  getwd(),
  file.path(base_dir, "R_analysis", "H3"),
  file.path(base_dir, "R_scripts", "H3"),
  file.path("/Volumes/Z Slim/07_04_2026_Data_Analysis", "R_scripts", "H3")
)
trait_core_filename <- "species_traits_core_h3_final.csv"
trait_extended_filename <- "species_traits_extended_sensitivity_full_final.csv"

traits_dir_hits <- traits_dir_candidates[
  dir.exists(traits_dir_candidates) &
    file.exists(file.path(traits_dir_candidates, trait_core_filename)) &
    file.exists(file.path(traits_dir_candidates, trait_extended_filename))
]

if (length(traits_dir_hits) > 0) {
  traits_dir <- traits_dir_hits[[1]]
} else {
  traits_dir_existing <- traits_dir_candidates[dir.exists(traits_dir_candidates)]
  traits_dir <- ifelse(length(traits_dir_existing) > 0, traits_dir_existing[[1]], traits_dir_candidates[[1]])
}

manual_traits_core_file <- file.path(traits_dir, trait_core_filename)
manual_traits_extended_file <- file.path(traits_dir, trait_extended_filename)

out_dir  <- file.path(output_r_root, "H3_sensitivity")
csv_dir  <- file.path(out_dir, "csv")
plot_dir <- file.path(out_dir, "plots")
txt_dir  <- file.path(out_dir, "txt")

if (!dir.exists(models_dir)) {
  stop("Models folder not found: ", models_dir)
}

for (d in c(output_r_root, out_dir, csv_dir, plot_dir, txt_dir)) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}


# ── 4. Constants ──────────────────────────────────────────────────────────────
model_order <- c("CNN", "ResNet", "HYDRA", "MultiRocket", "RF", "LGBM")
metrics     <- c("accuracy", "macro_recall", "macro_precision", "macro_f1")

required_pairwise_cols <- c("pair_id", "train_dataset", "test_dataset", metrics)
required_within_cols   <- c("analysis_dataset", metrics)

required_core_trait_cols <- c(
  "species",
  "log_body_mass_g",
  "dominant_locomotor_mode",
  "body_configuration_type",
  "expected_trunk_motion_amplitude",
  "expected_directional_change_tendency",
  "expected_head_neck_contribution_to_acceleration_relevant_motion",
  "dominant_feeding_foraging_mode"
)

required_extended_trait_cols <- c(
  "species",
  "log_body_mass_g",
  "dominant_locomotor_mode",
  "body_configuration_type",
  "expected_trunk_motion_amplitude",
  "expected_directional_change_tendency",
  "expected_head_neck_contribution_to_acceleration_relevant_motion",
  "dominant_feeding_foraging_mode",
  "activity_pattern",
  "vertical_use_motion_scope",
  "postural_compactness_body_profile_reduction",
  "defensive_posture_specialization"
)

core_trait_cols <- setdiff(required_core_trait_cols, "species")

dataset_to_species <- c(
  # Legacy/simple dataset labels
  "Bison" = "Bison bonasus",
  "Cattle" = "Bos taurus",
  "Dog" = "Canis lupus familiaris",
  "Giraffe" = "Giraffa camelopardalis",
  "Hedgehog" = "Erinaceus europaeus",
  
  # New final-dataset labels from the 11_05_2026 model outputs
  "Bison_dataset_1" = "Bison bonasus",
  "Cattle_dataset_1" = "Bos taurus",
  "Dog_dataset_1" = "Canis lupus familiaris",
  "Dog_dataset_2" = "Canis lupus familiaris",
  "Giraffe_dataset_1" = "Giraffa camelopardalis",
  "Hedgehog_dataset_1" = "Erinaceus europaeus",
  
  # Dataset labels already used by inter/pairwise designs
  "Fox_dataset_1" = "Vulpes vulpes",
  "Fox_dataset_2" = "Vulpes vulpes",
  "Horse_dataset_1" = "Equus ferus przewalskii",
  "Horse_dataset_2" = "Equus caballus",
  "Raccoon_dataset_1" = "Procyon lotor",
  "Raccoon_dataset_2" = "Procyon lotor"
)


# ── 5. Input file registry ────────────────────────────────────────────────────
model_files <- tribble(
  ~model,           ~pairwise_dir,                                                                    ~within_dir,
  "RF",            file.path(models_dir, "RF",          "Pairwise_RF",          "statistics"), file.path(models_dir, "RF",          "Within_RF",          "statistics"),
  "LGBM",          file.path(models_dir, "LGBM",        "Pairwise_LGBM",        "statistics"), file.path(models_dir, "LGBM",        "Within_LGBM",        "statistics"),
  "CNN",           file.path(models_dir, "CNN",         "Pairwise_CNN",         "statistics"), file.path(models_dir, "CNN",         "Within_CNN",         "statistics"),
  "ResNet",        file.path(models_dir, "ResNet",      "Pairwise_ResNet",      "statistics"), file.path(models_dir, "ResNet",      "Within_ResNet",      "statistics"),
  "HYDRA",         file.path(models_dir, "HYDRA",       "Pairwise_HYDRA",       "statistics"), file.path(models_dir, "HYDRA",       "Within_HYDRA",       "statistics"),
  "MultiRocket",   file.path(models_dir, "MultiRocket", "Pairwise_MultiRocket", "statistics"), file.path(models_dir, "MultiRocket", "Within_MultiRocket", "statistics")
)


# ── 6. Helpers ────────────────────────────────────────────────────────────────

# ---- 6.1 Formatting and utilities -------------------------------------------
# Small utilities for console headers, file checks, formatting, and plot labels.

print_section <- function(x) {
  cat("\n", strrep("=", 18), x, strrep("=", 18), "\n", sep = " ")
}

must_exist <- function(x) {
  missing <- x[!file.exists(x)]
  if (length(missing) > 0) stop("Missing files:\n", paste(missing, collapse = "\n"))
}
fmt_num <- function(x, digits = 3) {
  ifelse(is.na(x), "NA", formatC(x, format = "f", digits = digits))
}

fmt_p <- function(x) {
  ifelse(is.na(x), "NA", ifelse(x < 0.001, "< 0.001", sprintf("%.3f", round(x, 3))))
}

short_species <- function(x) {
  parts <- str_split(x, " ")
  vapply(parts, function(p) {
    if (length(p) >= 2) paste0(substr(p[1], 1, 1), ". ", paste(p[-1], collapse = " ")) else p[1]
  }, character(1))
}


# ---- 6.2 File resolution and data loading -----------------------------------
# Resolve the first available metrics file from a list of candidates, then
# load pairwise / within tables with required-column checks.

resolve_existing_file <- function(dir_path, candidates) {
  hits <- file.path(dir_path, candidates)
  hit <- hits[file.exists(hits)][1]
  
  if (length(hit) == 0 || is.na(hit)) {
    return(hits[1])
  }
  
  hit
}

model_files <- model_files %>%
  mutate(
    pairwise_file = map_chr(
      pairwise_dir,
      ~ resolve_existing_file(.x, c("metrics_all.csv", "pairwise_metrics_all.csv", "pairwise_summary_metrics.csv"))
    ),
    within_file = map_chr(
      within_dir,
      ~ resolve_existing_file(.x, c("metrics_all.csv", "within_metrics_all.csv", "within_summary_metrics.csv"))
    )
  ) %>%
  select(model, pairwise_file, within_file)

model_files_registry <- model_files %>%
  mutate(
    pairwise_exists = file.exists(pairwise_file),
    within_exists = file.exists(within_file),
    is_complete = pairwise_exists & within_exists,
    missing_reason = case_when(
      is_complete ~ "complete",
      !pairwise_exists & !within_exists ~ "missing pairwise and within metrics",
      !pairwise_exists ~ "missing pairwise metrics",
      !within_exists ~ "missing within metrics",
      TRUE ~ "unknown"
    )
  )

missing_model_files <- model_files_registry %>%
  filter(!is_complete)

model_files <- model_files_registry %>%
  filter(is_complete) %>%
  select(model, pairwise_file, within_file)

model_order_requested <- model_order
model_order <- model_order[model_order %in% model_files$model]

if (nrow(model_files) == 0) {
  stop("No complete H3 sensitivity model inputs found. Need pairwise and within metrics for at least one model.")
}

read_pairwise_model <- function(model, pairwise_file) {
  dt <- read_csv(pairwise_file, show_col_types = FALSE)
  missing_cols <- setdiff(required_pairwise_cols, names(dt))
  if (length(missing_cols) > 0) {
    stop(model, " pairwise missing cols: ", paste(missing_cols, collapse = ", "))
  }
  
  dt %>%
    select(any_of(required_pairwise_cols)) %>%
    mutate(model = model, .before = 1)
}

read_within_model <- function(model, within_file) {
  dt <- read_csv(within_file, show_col_types = FALSE)
  missing_cols <- setdiff(required_within_cols, names(dt))
  if (length(missing_cols) > 0) {
    stop(model, " within missing cols: ", paste(missing_cols, collapse = ", "))
  }
  
  dt %>%
    select(any_of(required_within_cols)) %>%
    transmute(
      model = model,
      analysis_dataset = analysis_dataset,
      within_accuracy = accuracy,
      within_macro_recall = macro_recall,
      within_macro_precision = macro_precision,
      within_macro_f1 = macro_f1
    )
}

read_trait_file <- function(file, required_cols) {
  dt <- read_csv(file, show_col_types = FALSE) %>%
    mutate(across(where(is.character), str_trim)) %>%
    filter(!if_all(everything(), ~ is.na(.x) | (.x == "")))
  
  missing_cols <- setdiff(required_cols, names(dt))
  if (length(missing_cols) > 0) {
    stop(basename(file), " missing cols: ", paste(missing_cols, collapse = ", "))
  }
  
  dt <- dt %>% select(all_of(required_cols))
  
  if (anyDuplicated(dt$species) > 0) {
    dup_species <- dt %>% dplyr::count(species) %>% filter(n > 1)
    stop("Duplicate species in ", basename(file), ":\n", paste(dup_species$species, collapse = "\n"))
  }
  
  if (any(!complete.cases(dt))) {
    stop(basename(file), " contains missing values.")
  }
  
  dt
}


# ---- 6.3 Trait handling and Gower distance ----------------------------------
# Convert trait columns to ordered/factor classes and compute the long-form
# Gower distance tables used in the functional-biomechanical sensitivity analyses.

validate_no_na_after_trait_conversion <- function(dt, label) {
  bad_cols <- names(dt)[vapply(dt, function(x) any(is.na(x)), logical(1))]
  if (length(bad_cols) > 0) {
    stop(
      label,
      " contains invalid or missing values after trait type conversion in: ",
      paste(bad_cols, collapse = ", ")
    )
  }
  dt
}

make_core_traits_ordered <- function(dt) {
  dt %>%
    mutate(
      log_body_mass_g = as.numeric(log_body_mass_g),
      expected_trunk_motion_amplitude = ordered(expected_trunk_motion_amplitude, levels = c("low", "moderate", "high", "very_high")),
      expected_directional_change_tendency = ordered(expected_directional_change_tendency, levels = c("low", "moderate", "high", "very_high")),
      expected_head_neck_contribution_to_acceleration_relevant_motion = ordered(expected_head_neck_contribution_to_acceleration_relevant_motion, levels = c("low", "moderate", "high", "very_high")),
      across(c(dominant_locomotor_mode, body_configuration_type, dominant_feeding_foraging_mode), as.factor)
    ) %>%
    validate_no_na_after_trait_conversion("Core trait table")
}

make_extended_traits_ordered <- function(dt) {
  dt %>%
    mutate(
      log_body_mass_g = as.numeric(log_body_mass_g),
      expected_trunk_motion_amplitude = ordered(expected_trunk_motion_amplitude, levels = c("low", "moderate", "high", "very_high")),
      expected_directional_change_tendency = ordered(expected_directional_change_tendency, levels = c("low", "moderate", "high", "very_high")),
      expected_head_neck_contribution_to_acceleration_relevant_motion = ordered(expected_head_neck_contribution_to_acceleration_relevant_motion, levels = c("low", "moderate", "high", "very_high")),
      defensive_posture_specialization = ordered(defensive_posture_specialization, levels = c("low", "moderate", "high", "very_high")),
      postural_compactness_body_profile_reduction = ordered(postural_compactness_body_profile_reduction, levels = c("low", "moderate", "high", "very_high")),
      across(c(dominant_locomotor_mode, body_configuration_type, dominant_feeding_foraging_mode, activity_pattern, vertical_use_motion_scope), as.factor)
    ) %>%
    validate_no_na_after_trait_conversion("Extended trait table")
}

compute_gower_long <- function(trait_df, value_name = "trait_distance_core") {
  mat <- as.matrix(daisy(trait_df %>% select(-species), metric = "gower"))
  rownames(mat) <- trait_df$species
  colnames(mat) <- trait_df$species
  
  as.data.frame(mat) %>%
    rownames_to_column("train_species") %>%
    pivot_longer(-train_species, names_to = "test_species", values_to = value_name)
}

compute_gower_without_trait <- function(drop_trait, trait_df, value_name = "trait_distance_core") {
  reduced <- trait_df %>% select(-all_of(drop_trait), -species)
  mat <- as.matrix(daisy(reduced, metric = "gower"))
  rownames(mat) <- trait_df$species
  colnames(mat) <- trait_df$species
  
  as.data.frame(mat) %>%
    rownames_to_column("train_species") %>%
    pivot_longer(-train_species, names_to = "test_species", values_to = value_name)
}


# ---- 6.4 Statistics ----------------------------------------------------------
# Spearman correlation helpers for the reference analyses and bootstrap CIs.

safe_spearman <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]
  y <- y[ok]
  if (length(x) < 3) {
    return(tibble(n = length(x), rho = NA_real_, p_value = NA_real_))
  }
  tmp <- suppressWarnings(cor.test(x, y, method = "spearman", exact = FALSE))
  tibble(n = length(x), rho = unname(tmp$estimate), p_value = tmp$p.value)
}

boot_spearman <- function(x, y, n_iter = n_boot) {
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]
  y <- y[ok]
  n <- length(x)
  
  if (n < 5) {
    return(tibble(rho_obs = NA_real_, ci_lo = NA_real_, ci_hi = NA_real_, n = n, n_boot = 0L))
  }
  
  rho_obs <- suppressWarnings(cor(x, y, method = "spearman"))
  rho_boot <- replicate(n_iter, {
    idx <- sample.int(n, replace = TRUE)
    suppressWarnings(cor(x[idx], y[idx], method = "spearman"))
  })
  rho_boot <- rho_boot[is.finite(rho_boot)]
  
  ci <- quantile(rho_boot, probs = c(0.025, 0.975))
  tibble(
    rho_obs = rho_obs,
    ci_lo = unname(ci[1]),
    ci_hi = unname(ci[2]),
    n = n,
    n_boot = length(rho_boot)
  )
}


# ---- 6.5 Plotting ------------------------------------------------------------
# Shared ggplot theme and save wrapper used across all exported figures.

theme_explore <- function(base_size = 11) {
  theme_classic(base_size = base_size) %+replace%
    theme(
      plot.title       = element_text(hjust = 0.5, face = "bold", size = 13),
      plot.subtitle    = element_text(hjust = 0.5, size = 10, color = "grey40"),
      strip.text       = element_text(face = "bold", size = 11),
      strip.background = element_blank(),
      axis.text        = element_text(size = 9.5, color = "black"),
      axis.title       = element_text(size = 11),
      axis.ticks       = element_line(linewidth = 0.35, color = "black"),
      axis.line        = element_line(linewidth = 0.35, color = "black"),
      legend.position  = "right"
    )
}

save_plot <- function(plot_obj, filename, width = 9, height = 6, dpi = 300) {
  ggsave(
    filename = file.path(plot_dir, filename),
    plot = plot_obj,
    width = width,
    height = height,
    dpi = dpi,
    bg = "white"
  )
}


# ── 7. Load data and prepare ──────────────────────────────────────────────────
print_section("LOADING DATA")

cat("Available complete models: ", paste(model_files$model, collapse = ", "), "\n", sep = "")
if (nrow(missing_model_files) > 0) {
  cat("Skipped incomplete models:\n")
  print(missing_model_files %>% select(model, missing_reason, pairwise_file, within_file), n = Inf)
}

study_species_expected <- sort(unique(unname(dataset_to_species)))

must_exist(c(
  manual_traits_core_file,
  manual_traits_extended_file
))

if (basename(manual_traits_core_file) != trait_core_filename ||
    basename(manual_traits_extended_file) != trait_extended_filename) {
  stop("Unexpected H3 trait file names. Use the final core and full extended CSVs only.")
}

trait_input_source <- "final_core_extended_trait_csvs"
core_traits_label <- basename(manual_traits_core_file)
extended_traits_label <- basename(manual_traits_extended_file)
extended_available <- TRUE

manual_traits_core_raw <- read_trait_file(manual_traits_core_file, required_core_trait_cols)
manual_traits_extended_raw <- read_trait_file(manual_traits_extended_file, required_extended_trait_cols)

manual_traits_core <- make_core_traits_ordered(manual_traits_core_raw)
manual_traits_extended <- make_extended_traits_ordered(manual_traits_extended_raw)

cat("Core final trait file detected: ", core_traits_label, "\n", sep = "")
cat("Extended final trait file detected: ", extended_traits_label, "\n", sep = "")
cat("Trait input source: ", trait_input_source, "\n", sep = "")

pairwise <- pmap_dfr(model_files[, c("model", "pairwise_file")], read_pairwise_model) %>%
  mutate(
    train_species = recode(train_dataset, !!!dataset_to_species, .default = NA_character_),
    test_species = recode(test_dataset, !!!dataset_to_species, .default = NA_character_),
    same_species = train_species == test_species,
    species_pair_id = paste(train_species, test_species, sep = "__"),
    model = factor(model, levels = model_order)
  )

# Kept for consistency with the final H3 structure
within_ref <- pmap_dfr(model_files[, c("model", "within_file")], read_within_model)

study_species <- pairwise %>%
  select(train_species, test_species) %>%
  pivot_longer(everything(), values_to = "species") %>%
  filter(!is.na(species)) %>%
  distinct(species)

unmapped_train <- pairwise %>% filter(is.na(train_species)) %>% distinct(train_dataset)
unmapped_test  <- pairwise %>% filter(is.na(test_species)) %>% distinct(test_dataset)
missing_traits_core <- study_species %>% anti_join(manual_traits_core, by = "species")
missing_traits_extended <- study_species %>% anti_join(manual_traits_extended, by = "species")

cat("Unmapped train datasets: ", nrow(unmapped_train), "\n", sep = "")
if (nrow(unmapped_train) > 0) print(unmapped_train)
cat("Unmapped test datasets: ", nrow(unmapped_test), "\n", sep = "")
if (nrow(unmapped_test) > 0) print(unmapped_test)
cat("Missing in core trait table: ", nrow(missing_traits_core), "\n", sep = "")
if (nrow(missing_traits_core) > 0) print(missing_traits_core)
cat("Missing in extended trait table: ", nrow(missing_traits_extended), "\n", sep = "")
if (nrow(missing_traits_extended) > 0) print(missing_traits_extended)

if (nrow(unmapped_train) > 0 || nrow(unmapped_test) > 0 ||
    nrow(missing_traits_core) > 0 || nrow(missing_traits_extended) > 0) {
  stop("Coverage check failed. Fix dataset_to_species or the final H3 trait CSVs first.")
}

gower_core_long <- compute_gower_long(manual_traits_core, "trait_distance_core")

analysis_dt <- pairwise %>%
  left_join(gower_core_long, by = c("train_species", "test_species")) %>%
  filter(is.finite(trait_distance_core), is.finite(macro_f1))

species_level_dt <- analysis_dt %>%
  group_by(model, train_species, test_species, trait_distance_core, same_species) %>%
  summarise(
    mean_macro_f1 = mean(macro_f1, na.rm = TRUE),
    sd_macro_f1 = sd(macro_f1, na.rm = TRUE),
    n_rows = n(),
    .groups = "drop"
  )

species_diff_dt <- species_level_dt %>% filter(!same_species)
row_diff_dt <- analysis_dt %>% filter(!same_species)


# ── 8. Reference correlations ────────────────────────────────────────────────
print_section("REFERENCE CORRELATIONS (core functional-biomechanical Gower, species-pair, diff-species)")

ref_overall_sp <- safe_spearman(species_diff_dt$trait_distance_core, species_diff_dt$mean_macro_f1)
ref_per_model_sp <- species_diff_dt %>%
  group_by(model) %>%
  group_modify(~ safe_spearman(.x$trait_distance_core, .x$mean_macro_f1)) %>%
  ungroup()

cat(
  "Overall: rho =", fmt_num(ref_overall_sp$rho),
  ", p =", fmt_p(ref_overall_sp$p_value),
  ", n =", ref_overall_sp$n, "\n"
)
print(ref_per_model_sp)


# ── 9. Leave-one-trait-out (LOTO) ────────────────────────────────────────────
print_section("1. LEAVE-ONE-TRAIT-OUT")

loto_results <- map_dfr(core_trait_cols, function(dropped) {
  cat("  Dropping:", dropped, "\n")
  
  gower_reduced <- compute_gower_without_trait(dropped, manual_traits_core, "trait_distance_core")
  
  sp_dt <- analysis_dt %>%
    select(-trait_distance_core) %>%
    left_join(gower_reduced, by = c("train_species", "test_species")) %>%
    filter(!same_species, is.finite(trait_distance_core), is.finite(macro_f1)) %>%
    group_by(model, train_species, test_species, trait_distance_core) %>%
    summarise(mean_macro_f1 = mean(macro_f1, na.rm = TRUE), .groups = "drop")
  
  overall <- safe_spearman(sp_dt$trait_distance_core, sp_dt$mean_macro_f1) %>%
    mutate(model = factor("overall", levels = c("overall", model_order)))
  
  per_model <- sp_dt %>%
    group_by(model) %>%
    group_modify(~ safe_spearman(.x$trait_distance_core, .x$mean_macro_f1)) %>%
    ungroup()
  
  bind_rows(overall, per_model) %>%
    mutate(dropped_trait = dropped, .before = 1)
})

loto_ref <- bind_rows(
  ref_overall_sp %>% mutate(model = factor("overall", levels = c("overall", model_order))),
  ref_per_model_sp
) %>%
  mutate(dropped_trait = "none (full model)", .before = 1)

loto_all <- bind_rows(loto_ref, loto_results) %>%
  mutate(model = factor(as.character(model), levels = c("overall", model_order))) %>%
  arrange(model, dropped_trait)

loto_delta <- loto_results %>%
  left_join(loto_ref %>% select(model, rho_full = rho), by = "model") %>%
  mutate(delta_rho = rho - rho_full)

print_section("LOTO: largest rho changes (overall)")
loto_delta %>%
  filter(model == "overall") %>%
  arrange(delta_rho) %>%
  print()

loto_plot_dt <- loto_all %>%
  mutate(
    dropped_trait = str_replace_all(dropped_trait, "_", " "),
    is_ref = dropped_trait == "none (full model)"
  )

p_loto <- ggplot(loto_plot_dt, aes(x = reorder(dropped_trait, -rho), y = rho)) +
  geom_hline(
    data = loto_ref %>% rename(ref_rho = rho),
    aes(yintercept = ref_rho),
    linetype = "dashed",
    color = "grey50",
    linewidth = 0.4
  ) +
  geom_point(
    aes(fill = ifelse(is_ref, "#E69F00", "grey35")),
    shape = 21,
    size = 3,
    color = "white",
    stroke = 0.4
  ) +
  scale_fill_identity() +
  facet_wrap(~ model, ncol = 3, scales = "free_y") +
  coord_flip() +
  labs(
    title = "Leave-one-trait-out: Spearman rho",
    subtitle = "Core functional-biomechanical trait distance, species-pair level, different species only",
    x = NULL,
    y = "Spearman rho"
  ) +
  theme_explore() +
  theme(axis.text.y = element_text(size = 8))

save_plot(p_loto, "01_loto_rho_by_model.png", width = 14, height = 8)


# ── 10. Leave-one-species-out (LOSO) ──────────────────────────────────────────
print_section("2. LEAVE-ONE-SPECIES-OUT")

all_species <- sort(unique(c(species_diff_dt$train_species, species_diff_dt$test_species)))

loso_results <- map_dfr(all_species, function(sp) {
  cat("  Dropping:", sp, "\n")
  
  sp_dt <- species_diff_dt %>% filter(train_species != sp, test_species != sp)
  
  if (nrow(sp_dt) < 5) {
    return(tibble(
      dropped_species = sp,
      model = factor("overall", levels = c("overall", model_order)),
      n = nrow(sp_dt),
      rho = NA_real_,
      p_value = NA_real_
    ))
  }
  
  overall <- safe_spearman(sp_dt$trait_distance_core, sp_dt$mean_macro_f1) %>%
    mutate(model = factor("overall", levels = c("overall", model_order)))
  
  per_model <- sp_dt %>%
    group_by(model) %>%
    group_modify(~ safe_spearman(.x$trait_distance_core, .x$mean_macro_f1)) %>%
    ungroup()
  
  bind_rows(overall, per_model) %>%
    mutate(dropped_species = sp, .before = 1)
})

loso_ref <- bind_rows(
  ref_overall_sp %>% mutate(model = factor("overall", levels = c("overall", model_order))),
  ref_per_model_sp
) %>%
  mutate(dropped_species = "none (full)", .before = 1)

loso_all <- bind_rows(loso_ref, loso_results) %>%
  mutate(model = factor(as.character(model), levels = c("overall", model_order))) %>%
  arrange(model, dropped_species)

loso_plot_dt <- loso_all %>%
  mutate(
    short_name = ifelse(dropped_species == "none (full)", "none (full)", short_species(dropped_species)),
    is_ref = dropped_species == "none (full)"
  )

p_loso <- ggplot(loso_plot_dt, aes(x = reorder(short_name, -rho), y = rho)) +
  geom_hline(
    data = loso_ref %>% rename(ref_rho = rho),
    aes(yintercept = ref_rho),
    linetype = "dashed",
    color = "grey50",
    linewidth = 0.4
  ) +
  geom_point(
    aes(fill = ifelse(is_ref, "#E69F00", "grey35")),
    shape = 21,
    size = 3.5,
    color = "white",
    stroke = 0.4
  ) +
  scale_fill_identity() +
  facet_wrap(~ model, ncol = 3, scales = "free_y") +
  coord_flip() +
  labs(
    title = "Leave-one-species-out: Spearman rho",
    subtitle = "Core functional-biomechanical trait distance, species-pair level, different species only",
    x = NULL,
    y = "Spearman rho"
  ) +
  theme_explore() +
  theme(axis.text.y = element_text(size = 9))

save_plot(p_loso, "02_loso_rho_by_model.png", width = 14, height = 7)

print_section("LOSO: most influential species (overall)")
loso_results %>%
  filter(model == "overall") %>%
  mutate(delta_rho = rho - ref_overall_sp$rho) %>%
  arrange(delta_rho) %>%
  print()


# ── 11. Bootstrap confidence intervals ────────────────────────────────────────
print_section("3. BOOTSTRAP CIs")

cat("  Bootstrapping overall (species-pair) ...\n")
boot_overall <- boot_spearman(species_diff_dt$trait_distance_core, species_diff_dt$mean_macro_f1) %>%
  mutate(model = factor("overall", levels = c("overall", model_order)), scope = "species_pair_diff")

cat("  Bootstrapping per model (species-pair) ...\n")
boot_per_model <- species_diff_dt %>%
  group_by(model) %>%
  group_modify(~ boot_spearman(.x$trait_distance_core, .x$mean_macro_f1)) %>%
  ungroup() %>%
  mutate(scope = "species_pair_diff")

cat("  Bootstrapping overall (row level) ...\n")
boot_row_overall <- boot_spearman(row_diff_dt$trait_distance_core, row_diff_dt$macro_f1) %>%
  mutate(model = factor("overall", levels = c("overall", model_order)), scope = "row_diff")

cat("  Bootstrapping per model (row level) ...\n")
boot_row_per_model <- row_diff_dt %>%
  group_by(model) %>%
  group_modify(~ boot_spearman(.x$trait_distance_core, .x$macro_f1)) %>%
  ungroup() %>%
  mutate(scope = "row_diff")

boot_all <- bind_rows(boot_overall, boot_per_model, boot_row_overall, boot_row_per_model) %>%
  mutate(model = factor(as.character(model), levels = c("overall", model_order))) %>%
  select(scope, model, n, rho_obs, ci_lo, ci_hi, n_boot) %>%
  arrange(scope, model)

print(boot_all)

boot_plot_dt <- boot_all %>%
  mutate(
    scope_label = recode(scope, "species_pair_diff" = "species-pair", "row_diff" = "row-level"),
    label = paste0(model, " (", scope_label, ")"),
    label = factor(label, levels = rev(unique(label)))
  )

p_boot <- ggplot(boot_plot_dt, aes(x = rho_obs, y = label)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey60") +
  geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi), height = 0.25, linewidth = 0.5, color = "grey40") +
  geom_point(shape = 21, size = 3.5, fill = "grey35", color = "white", stroke = 0.4) +
  labs(
    title = paste0("Bootstrap 95% CI for Spearman rho (", format(n_boot, big.mark = ","), " resamples)"),
    subtitle = "Core functional-biomechanical trait distance, different species only",
    x = "Spearman rho",
    y = NULL
  ) +
  theme_explore() +
  theme(axis.text.y = element_text(size = 9.5))

save_plot(p_boot, "03_bootstrap_ci_forest.png", width = 10, height = 6)


# ── 12. Pseudoreplication check ───────────────────────────────────────────────
print_section("4. PSEUDOREPLICATION CHECK")

pseudo_row <- row_diff_dt %>%
  group_by(model) %>%
  group_modify(~ safe_spearman(.x$trait_distance_core, .x$macro_f1)) %>%
  ungroup() %>%
  mutate(level = "A_row")
pseudo_row_overall <- safe_spearman(row_diff_dt$trait_distance_core, row_diff_dt$macro_f1) %>%
  mutate(model = factor("overall", levels = c("overall", model_order)), level = "A_row")

dataset_pair_dt <- row_diff_dt %>%
  group_by(model, train_dataset, test_dataset, trait_distance_core) %>%
  summarise(mean_macro_f1 = mean(macro_f1, na.rm = TRUE), .groups = "drop")

pseudo_dataset <- dataset_pair_dt %>%
  group_by(model) %>%
  group_modify(~ safe_spearman(.x$trait_distance_core, .x$mean_macro_f1)) %>%
  ungroup() %>%
  mutate(level = "B_dataset_pair")
pseudo_dataset_overall <- safe_spearman(dataset_pair_dt$trait_distance_core, dataset_pair_dt$mean_macro_f1) %>%
  mutate(model = factor("overall", levels = c("overall", model_order)), level = "B_dataset_pair")

pseudo_species <- species_diff_dt %>%
  group_by(model) %>%
  group_modify(~ safe_spearman(.x$trait_distance_core, .x$mean_macro_f1)) %>%
  ungroup() %>%
  mutate(level = "C_species_pair")
pseudo_species_overall <- safe_spearman(species_diff_dt$trait_distance_core, species_diff_dt$mean_macro_f1) %>%
  mutate(model = factor("overall", levels = c("overall", model_order)), level = "C_species_pair")

pseudo_all <- bind_rows(
  pseudo_row, pseudo_row_overall,
  pseudo_dataset, pseudo_dataset_overall,
  pseudo_species, pseudo_species_overall
) %>%
  mutate(model = factor(as.character(model), levels = c("overall", model_order))) %>%
  select(level, model, n, rho, p_value) %>%
  arrange(level, model)

print(pseudo_all)

pseudo_plot_dt <- pseudo_all %>%
  mutate(level_label = recode(
    level,
    "A_row" = "A: Raw rows",
    "B_dataset_pair" = "B: Dataset pairs",
    "C_species_pair" = "C: Species pairs"
  ))

p_pseudo <- ggplot(pseudo_plot_dt, aes(x = model, y = rho, fill = level_label)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey60") +
  geom_col(position = position_dodge(width = 0.75), width = 0.65, alpha = 0.85) +
  geom_text(
    aes(label = sprintf("%.3f", rho), y = rho + sign(rho) * 0.015),
    position = position_dodge(width = 0.75),
    size = 2.8,
    vjust = 0
  ) +
  scale_fill_manual(values = c(
    "A: Raw rows" = "#bdbdbd",
    "B: Dataset pairs" = "#737373",
    "C: Species pairs" = "#252525"
  )) +
  labs(
    title = "Pseudoreplication check: Spearman rho by aggregation level",
    subtitle = "Core functional-biomechanical trait distance, different species only",
    x = NULL,
    y = "Spearman rho",
    fill = "Level"
  ) +
  theme_explore() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1), legend.position = "top")

save_plot(p_pseudo, "04_pseudoreplication_comparison.png", width = 10, height = 6.5)


# ── 13. Paper-ready combined panel ────────────────────────────────────────────
print_section("PAPER-READY COMBINED PANEL")

loto_overall <- loto_all %>%
  filter(model == "overall") %>%
  mutate(
    trait_label = str_replace_all(dropped_trait, "_", " "),
    is_ref = dropped_trait == "none (full model)"
  )

p_panel_loto <- ggplot(loto_overall, aes(x = reorder(trait_label, -rho), y = rho)) +
  geom_hline(yintercept = ref_overall_sp$rho, linetype = "dashed", color = "grey50", linewidth = 0.4) +
  geom_point(aes(fill = ifelse(is_ref, "#E69F00", "grey35")), shape = 21, size = 3, color = "white", stroke = 0.4) +
  scale_fill_identity() +
  coord_flip() +
  labs(title = "A  Leave-one-trait-out", x = NULL, y = "rho") +
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_text(hjust = 0, face = "bold", size = 11),
    axis.text = element_text(size = 8.5, color = "black"),
    axis.line = element_line(linewidth = 0.3)
  )

loso_overall <- loso_all %>%
  filter(model == "overall") %>%
  mutate(
    short_name = ifelse(dropped_species == "none (full)", "none (full)", short_species(dropped_species)),
    is_ref = dropped_species == "none (full)"
  )

p_panel_loso <- ggplot(loso_overall, aes(x = reorder(short_name, -rho), y = rho)) +
  geom_hline(yintercept = ref_overall_sp$rho, linetype = "dashed", color = "grey50", linewidth = 0.4) +
  geom_point(aes(fill = ifelse(is_ref, "#E69F00", "grey35")), shape = 21, size = 3, color = "white", stroke = 0.4) +
  scale_fill_identity() +
  coord_flip() +
  labs(title = "B  Leave-one-species-out", x = NULL, y = "rho") +
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_text(hjust = 0, face = "bold", size = 11),
    axis.text = element_text(size = 8.5, color = "black"),
    axis.line = element_line(linewidth = 0.3)
  )

boot_sp <- boot_all %>%
  filter(scope == "species_pair_diff") %>%
  mutate(model = factor(as.character(model), levels = c("overall", model_order)))

p_panel_boot <- ggplot(boot_sp, aes(x = rho_obs, y = model)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey60") +
  geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi), height = 0.3, linewidth = 0.5, color = "grey40") +
  geom_point(shape = 21, size = 3.5, fill = "grey35", color = "white", stroke = 0.4) +
  labs(title = "C  Bootstrap 95% CI", x = "rho", y = NULL) +
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_text(hjust = 0, face = "bold", size = 11),
    axis.text = element_text(size = 8.5, color = "black"),
    axis.line = element_line(linewidth = 0.3)
  )

pseudo_overall_plot <- pseudo_all %>%
  filter(model == "overall") %>%
  mutate(level_label = recode(
    level,
    "A_row" = "raw rows",
    "B_dataset_pair" = "dataset pairs",
    "C_species_pair" = "species pairs"
  ))

p_panel_pseudo <- ggplot(pseudo_overall_plot, aes(x = level_label, y = rho)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey60") +
  geom_col(fill = "grey35", width = 0.6, alpha = 0.85) +
  geom_text(aes(label = sprintf("%.3f", rho)), vjust = -0.5, size = 3.2) +
  labs(title = "D  Pseudoreplication", x = NULL, y = "rho") +
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_text(hjust = 0, face = "bold", size = 11),
    axis.text = element_text(size = 8.5, color = "black"),
    axis.line = element_line(linewidth = 0.3)
  )

p_combined <- (p_panel_loto | p_panel_loso) / (p_panel_boot | p_panel_pseudo) +
  plot_annotation(
    title = "Sensitivity analyses",
    subtitle = "Core functional-biomechanical trait distance, species-pair level, different species only",
    theme = theme(
      plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5, color = "grey40")
    )
  )

save_plot(p_combined, "05_sensitivity_combined.png", width = 10.24, height = 8.66, dpi = 600)

# ---- 13b. Single panels, exported individually in high resolution ------------
# Each of the four panels (A-D) is also saved as a standalone 600 dpi figure.
single_panels <- list(
  "05a_panel_leave_one_trait_out.png"   = p_panel_loto,
  "05b_panel_leave_one_species_out.png" = p_panel_loso,
  "05c_panel_bootstrap_ci.png"          = p_panel_boot,
  "05d_panel_pseudoreplication.png"     = p_panel_pseudo
)

for (panel_file in names(single_panels)) {
  save_plot(single_panels[[panel_file]], panel_file, width = 5.5, height = 4.5, dpi = 600)
}


# ── 14. Export CSV ────────────────────────────────────────────────────────────
print_section("EXPORTING CSV")

csv_exports <- list(
  "00_model_file_registry.csv"          = model_files_registry,
  "01_loto_results.csv"                 = loto_all,
  "02_loto_delta_rho.csv"               = loto_delta,
  "03_loso_results.csv"                 = loso_all,
  "04_bootstrap_ci.csv"                 = boot_all,
  "05_pseudoreplication_comparison.csv" = pseudo_all
)

walk2(csv_exports, names(csv_exports), ~ write_csv(.x, file.path(csv_dir, .y)))

plot_index <- tibble(
  file = c(
    "01_loto_rho_by_model.png",
    "02_loso_rho_by_model.png",
    "03_bootstrap_ci_forest.png",
    "04_pseudoreplication_comparison.png",
    "05_sensitivity_combined.png",
    "05a_panel_leave_one_trait_out.png",
    "05b_panel_leave_one_species_out.png",
    "05c_panel_bootstrap_ci.png",
    "05d_panel_pseudoreplication.png"
  ),
  description = c(
    "LOTO rho by model",
    "LOSO rho by model",
    "Bootstrap CI forest plot",
    "Pseudoreplication comparison",
    "Combined sensitivity figure",
    "Single panel A: leave-one-trait-out (overall)",
    "Single panel B: leave-one-species-out (overall)",
    "Single panel C: bootstrap 95% CI",
    "Single panel D: pseudoreplication"
  )
)

write_csv(plot_index, file.path(csv_dir, "06_plot_index.csv"))

input_summary <- tibble(
  item = c(
    "base_dir",
    "models_dir",
    "traits_dir",
    "output_dir",
    "models_requested",
    "models_complete_used",
    "models_skipped_incomplete",
    "core_traits_file",
    "extended_traits_file",
    "trait_input_source",
    "extended_traits_file_found",
    "n_pairwise_rows",
    "n_species_pairs_diff",
    "n_row_pairs_diff",
    "n_within_rows"
  ),
  value = c(
    base_dir,
    models_dir,
    traits_dir,
    out_dir,
    paste(model_order_requested, collapse = ", "),
    paste(model_files$model, collapse = ", "),
    paste(missing_model_files$model, collapse = ", "),
    core_traits_label,
    extended_traits_label,
    trait_input_source,
    as.character(extended_available),
    as.character(nrow(pairwise)),
    as.character(nrow(species_diff_dt)),
    as.character(nrow(row_diff_dt)),
    as.character(nrow(within_ref))
  )
)

write_csv(input_summary, file.path(csv_dir, "07_input_summary.csv"))


# ── 15. Export TXT ────────────────────────────────────────────────────────────
print_section("EXPORTING TXT")

loto_ov <- loto_delta %>% filter(model == "overall")
loto_max_delta <- max(abs(loto_ov$delta_rho), na.rm = TRUE)
loto_min_trait <- loto_results %>% filter(model == "overall") %>% slice_min(rho, n = 1)
loto_max_trait <- loto_results %>% filter(model == "overall") %>% slice_max(rho, n = 1)

loso_ov <- loso_results %>% filter(model == "overall")
loso_sign_changes <- loso_ov %>% filter(sign(rho) != sign(ref_overall_sp$rho))
loso_sig_changes <- loso_ov %>% filter((p_value >= 0.05) != (ref_overall_sp$p_value >= 0.05))
loso_min_sp <- loso_ov %>% slice_min(rho, n = 1)
loso_max_sp <- loso_ov %>% slice_max(rho, n = 1)

boot_crosses_zero <- boot_all %>% filter(scope == "species_pair_diff", ci_lo <= 0 & ci_hi >= 0)

pseudo_ov <- pseudo_all %>% filter(model == "overall")
pseudo_rho_range <- range(pseudo_ov$rho, na.rm = TRUE)

loto_stable <- is.finite(loto_max_delta) && loto_max_delta < 0.05
loso_stable <- all(loso_ov$rho < 0, na.rm = TRUE)
boot_stable <- is.finite(boot_overall$ci_hi) && boot_overall$ci_hi < 0
pseudo_stable <- all(pseudo_ov$rho < 0, na.rm = TRUE)

summary_lines <- c(
  "# ===================================================",
  "# Sensitivity Analysis Summary - H3_sensitivity_final_traits.R",
  "# ===================================================",
  "",
  paste0("Date:              ", Sys.time()),
  paste0("Base dir:          ", base_dir),
  paste0("Models dir:        ", models_dir),
  paste0("Traits dir:        ", traits_dir),
  paste0("Bootstrap samples: ", n_boot),
  paste0("Random seed:       42"),
  paste0("Primary traits:    ", core_traits_label),
  paste0("Trait input source: ", trait_input_source),
  paste0("Extended traits:   ", extended_traits_label),
  paste0("Models used:       ", paste(model_files$model, collapse = ", ")),
  "",
  "# ----- Reference (full model, core functional-biomechanical Gower, species-pair, diff-species) -----",
  paste0("Overall rho = ", fmt_num(ref_overall_sp$rho), ", p = ", fmt_p(ref_overall_sp$p_value), ", n = ", ref_overall_sp$n),
  "",
  "Per model:",
  paste0(
    "  ", ref_per_model_sp$model,
    ": rho = ", fmt_num(ref_per_model_sp$rho),
    ", p = ", fmt_p(ref_per_model_sp$p_value),
    ", n = ", ref_per_model_sp$n
  ),
  "",
  "# ----- 1. Leave-one-trait-out (LOTO) -----",
  "Overall rho range when dropping one core trait:",
  paste0("  min rho = ", fmt_num(loto_min_trait$rho), " (dropped: ", loto_min_trait$dropped_trait, ")"),
  paste0("  max rho = ", fmt_num(loto_max_trait$rho), " (dropped: ", loto_max_trait$dropped_trait, ")"),
  paste0("  full model rho = ", fmt_num(ref_overall_sp$rho)),
  paste0("  max |delta rho| = ", fmt_num(loto_max_delta, 4)),
  "",
  "# ----- 2. Leave-one-species-out (LOSO) -----",
  "Overall rho range when dropping one species:",
  paste0("  min rho = ", fmt_num(loso_min_sp$rho), " (dropped: ", loso_min_sp$dropped_species, ")"),
  paste0("  max rho = ", fmt_num(loso_max_sp$rho), " (dropped: ", loso_max_sp$dropped_species, ")"),
  ifelse(
    nrow(loso_sign_changes) > 0,
    paste0("  WARNING: sign flip when dropping: ", paste(loso_sign_changes$dropped_species, collapse = ", ")),
    "  No sign flips - correlation direction is stable."
  ),
  ifelse(
    nrow(loso_sig_changes) > 0,
    paste0("  Significance changes when dropping: ", paste(loso_sig_changes$dropped_species, collapse = ", ")),
    "  No significance changes (all remain p < 0.05 or all remain n.s.)."
  ),
  "",
  "# ----- 3. Bootstrap CIs -----",
  "Species-pair level, different species only:",
  paste0("  Overall: rho = ", fmt_num(boot_overall$rho_obs), " [", fmt_num(boot_overall$ci_lo), ", ", fmt_num(boot_overall$ci_hi), "]"),
  paste0(
    "  ", boot_per_model$model,
    ": rho = ", fmt_num(boot_per_model$rho_obs),
    " [", fmt_num(boot_per_model$ci_lo), ", ", fmt_num(boot_per_model$ci_hi), "]"
  ),
  ifelse(
    nrow(boot_crosses_zero) > 0,
    paste0("  CI crosses zero for: ", paste(boot_crosses_zero$model, collapse = ", ")),
    "  No CI crosses zero - all correlations are robust."
  ),
  "",
  "# ----- 4. Pseudoreplication check -----",
  "Overall rho at each aggregation level (diff-species, core functional-biomechanical Gower):",
  paste0(
    "  ", c("raw rows", "dataset pairs", "species pairs"),
    ": rho = ", fmt_num(pseudo_ov$rho),
    ", p = ", fmt_p(pseudo_ov$p_value),
    ", n = ", pseudo_ov$n
  ),
  paste0("  rho range across levels: [", fmt_num(pseudo_rho_range[1]), ", ", fmt_num(pseudo_rho_range[2]), "]"),
  "",
  "# ----- Overall verdict -----",
  paste0("LOTO:    ", ifelse(loto_stable,
                             paste0("STABLE (max |delta rho| = ", fmt_num(loto_max_delta, 4), " < 0.05)"),
                             paste0("SENSITIVE (max |delta rho| = ", fmt_num(loto_max_delta, 4), " >= 0.05)"))),
  paste0("LOSO:    ", ifelse(loso_stable, "STABLE (sign always negative)", "SENSITIVE (sign flip detected)")),
  paste0("Boot CI: ", ifelse(boot_stable, "STABLE (CI does not cross zero)", "UNCERTAIN (CI crosses zero)")),
  paste0("Pseudo:  ", ifelse(pseudo_stable, "STABLE (negative at all levels)", "SENSITIVE"))
)

writeLines(summary_lines, con = file.path(txt_dir, "01_h3_sensitivity_summary.txt"))

files_lines <- c(
  paste0("Output folder: ", out_dir),
  paste0("CSV folder: ", csv_dir),
  paste0("Plots folder: ", plot_dir),
  paste0("TXT folder: ", txt_dir),
  "",
  "Written plot files:",
  paste0("- ", plot_index$file),
  "",
  "Written CSV files:",
  paste0("- ", c(names(csv_exports), "06_plot_index.csv", "07_input_summary.csv"))
)

writeLines(files_lines, con = file.path(txt_dir, "02_h3_sensitivity_files_written.txt"))


# ── 16. Console output ────────────────────────────────────────────────────────
cat("\n")
cat(paste(summary_lines, collapse = "\n"))
cat("\n")

print_section("DONE")
cat("Output directory: ", out_dir, "\n", sep = "")
cat("CSVs:             ", csv_dir, "\n", sep = "")
cat("Plots:            ", plot_dir, "\n", sep = "")
cat("TXT:              ", txt_dir, "\n", sep = "")