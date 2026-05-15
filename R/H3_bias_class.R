# =============================================================================
# H3 bias analysis: class-distribution and dataset-size controls
# =============================================================================
# Author : Lucas Beseler
# Date   : 2026-05-15
#
# Purpose:
# - Test whether the H3 trait-distance effect remains after controlling for
#   class-distribution shift and dataset-size asymmetry.
# - Recalculate functional-biomechanical Gower distances using log body mass
#   instead of body-size classes.
# - Export publication-ready bias-control tables, plots, and text reports for H3.
#
# Hypothesis:
# - H3: Cross-species transfer performance declines with increasing
#   functional-biomechanical trait distance between source and target species.
#
# Notes:
# - This script is a bias-control analysis for H3 and does not replace the main
#   trait-distance analysis.
#
# =============================================================================

# ── 1. Setup ──────────────────────────────────────────────────────────────────
if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman", repos = "https://cloud.r-project.org")
}

pacman::p_load(
  readr, dplyr, tidyr, tibble, purrr, stringr,
  cluster, ggplot2, scales
)


# ── 2. User settings ──────────────────────────────────────────────────────────
base_dir      <- "/Volumes/Z Slim/11_05_2026_Data_Analysis"
models_dir    <- file.path(base_dir, "Models")
output_r_root <- file.path(base_dir, "Output_R")

# Keep old project only as fallback for manually curated H3 trait files.
old_base_dir <- "/Volumes/Z Slim/07_04_2026_Data_Analysis"


# ── 3. Paths ──────────────────────────────────────────────────────────────────
out_dir   <- file.path(output_r_root, "H3_bias_class_shift")
csv_dir   <- file.path(out_dir, "csv")
plots_dir <- file.path(out_dir, "plots")
txt_dir   <- file.path(out_dir, "txt")

traits_dir_candidates <- c(
  file.path(base_dir, "R_analysis", "H3"),
  file.path(base_dir, "R_scripts", "H3"),
  file.path("/Volumes/Z Slim/07_04_2026_Data_Analysis", "R_scripts", "H3")
)
traits_dir <- traits_dir_candidates[dir.exists(traits_dir_candidates)][1]
if (length(traits_dir) == 0 || is.na(traits_dir)) {
  traits_dir <- traits_dir_candidates[[1]]
}
elton_bodymass_dir <- file.path(traits_dir, "elton_bodymass_check")

manual_traits_core_candidates <- c(
  file.path(elton_bodymass_dir, "species_traits_core_logbodymass_gower.csv"),
  file.path(traits_dir, "species_traits_core_logbodymass_gower.csv"),
  file.path(elton_bodymass_dir, "species_traits_core_with_elton_bodymass.csv"),
  file.path(traits_dir, "species_traits_core.csv")
)
manual_traits_core_file <- manual_traits_core_candidates[file.exists(manual_traits_core_candidates)][1]
if (length(manual_traits_core_file) == 0 || is.na(manual_traits_core_file)) {
  manual_traits_core_file <- manual_traits_core_candidates[[1]]
}

# Preferred source in the new structure: model-level diagnostic files written by
# the Python model scripts. Old Supplement files are kept only as fallback.
model_summary_roots <- file.path(models_dir, c("CNN", "ResNet", "HYDRA", "MultiRocket", "RF", "LGBM"))

supp_behavior_file_candidates <- c(
  file.path(output_r_root, "Supp_Material", "csv", "dataset_behavior_distribution.csv"),
  file.path(base_dir, "Output", "Supp_Material", "csv", "dataset_behavior_distribution.csv"),
  file.path(old_base_dir, "Output", "Supp_Material", "csv", "dataset_behavior_distribution.csv")
)
supp_overview_file_candidates <- c(
  file.path(output_r_root, "Supp_Material", "csv", "dataset_overview.csv"),
  file.path(base_dir, "Output", "Supp_Material", "csv", "dataset_overview.csv"),
  file.path(old_base_dir, "Output", "Supp_Material", "csv", "dataset_overview.csv")
)

for (d in c(output_r_root, out_dir, csv_dir, plots_dir, txt_dir)) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

if (!dir.exists(models_dir)) {
  stop("Models folder not found: ", models_dir)
}


# ── 4. Constants ──────────────────────────────────────────────────────────────
model_order <- c("CNN", "ResNet", "HYDRA", "MultiRocket", "RF", "LGBM")
required_pairwise_cols <- c("pair_id", "train_dataset", "test_dataset", "accuracy", "macro_recall", "macro_precision", "macro_f1")
required_core_trait_cols <- c(
  "species",
  "log_bodymass_g",
  "dominant_locomotor_mode",
  "body_configuration_type",
  "expected_trunk_motion_amplitude",
  "directional_change_frequency",
  "head_neck_contribution_to_whole_body_motion",
  "postural_compactness_body_profile_reduction"
)
target_behaviors <- c("Foraging", "Locomotion", "Resting")


# ── 5. Input file registry ────────────────────────────────────────────────────
resolve_existing_file <- function(dir_path, candidates) {
  hits <- file.path(dir_path, candidates)
  hit <- hits[file.exists(hits)][1]
  
  if (length(hit) == 0 || is.na(hit)) {
    return(hits[1])
  }
  
  hit
}

model_files <- tribble(
  ~model,         ~pairwise_dir,
  "RF",          file.path(models_dir, "RF",          "Pairwise_RF",          "statistics"),
  "LGBM",        file.path(models_dir, "LGBM",        "Pairwise_LGBM",        "statistics"),
  "CNN",         file.path(models_dir, "CNN",         "Pairwise_CNN",         "statistics"),
  "ResNet",      file.path(models_dir, "ResNet",      "Pairwise_ResNet",      "statistics"),
  "HYDRA",       file.path(models_dir, "HYDRA",       "Pairwise_HYDRA",       "statistics"),
  "MultiRocket", file.path(models_dir, "MultiRocket", "Pairwise_MultiRocket", "statistics")
)

model_files <- model_files %>%
  mutate(
    pairwise_file = map_chr(
      pairwise_dir,
      ~ resolve_existing_file(.x, c("metrics_all.csv", "pairwise_metrics_all.csv", "pairwise_summary_metrics.csv"))
    ),
    pairwise_exists = file.exists(pairwise_file),
    status = if_else(pairwise_exists, "used", "missing_pairwise_metrics")
  ) %>%
  select(model, pairwise_file, pairwise_exists, status)

write_csv(model_files, file.path(csv_dir, "00_model_file_registry.csv"))
model_files_used <- model_files %>% filter(pairwise_exists) %>% select(model, pairwise_file)

if (nrow(model_files_used) == 0) {
  stop("No pairwise model metric files found in: ", models_dir)
}


# ── 6. Helpers ────────────────────────────────────────────────────────────────

# ---- 6.1 Formatting and reporting --------------------------------------------
must_exist <- function(x) {
  missing <- x[!file.exists(x)]
  if (length(missing) > 0) stop("Missing files:\n", paste(missing, collapse = "\n"))
}

print_section <- function(x) {
  cat("\n", strrep("=", 18), x, strrep("=", 18), "\n", sep = " ")
}

append_section <- function(lines, title, obj = NULL) {
  section_header <- c(paste0(strrep("=", 18), " ", title, " ", strrep("=", 18)))
  if (is.null(obj)) return(c(lines, "", section_header, ""))
  body <- capture.output(print(obj))
  c(lines, "", section_header, body)
}

write_txt_report <- function(file, lines) {
  writeLines(enc2utf8(lines), con = file, useBytes = TRUE)
}

fmt_num <- function(x, digits = 3) {
  ifelse(is.na(x), "NA", formatC(x, format = "f", digits = digits))
}

fmt_p <- function(x) {
  ifelse(is.na(x), "NA", format.pval(x, digits = 3, eps = 0.001))
}


# ---- 6.2 Data loading and parsing --------------------------------------------
find_first_col <- function(nms, candidates) {
  hit <- intersect(candidates, nms)
  if (length(hit) == 0) return(NA_character_)
  hit[[1]]
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

normalize_behavior <- function(x) {
  x_low <- str_to_lower(str_trim(as.character(x)))
  case_when(
    str_detect(x_low, "forag") ~ "Foraging",
    str_detect(x_low, "locomot|walk|run|move|travel") ~ "Locomotion",
    str_detect(x_low, "rest|sleep|inactive|lie|lying|sit|stationary") ~ "Resting",
    TRUE ~ NA_character_
  )
}

parse_prop_to_fraction <- function(x) {
  if (is.numeric(x)) {
    out <- as.numeric(x)
  } else {
    x_chr <- str_trim(as.character(x))
    x_chr[x_chr %in% c("", "NA", "NaN", "NULL", "null")] <- NA_character_
    out <- suppressWarnings(parse_number(x_chr))
    has_pct <- !is.na(x_chr) & str_detect(x_chr, "%")
    if (any(has_pct, na.rm = TRUE)) {
      out[has_pct] <- out[has_pct] / 100
    }
  }
  gt_one <- is.finite(out) & out > 1
  if (any(gt_one)) {
    out[gt_one] <- out[gt_one] / 100
  }
  out
}

read_behavior_distribution <- function(file, dataset_overview = NULL) {
  dt <- read_csv(file, show_col_types = FALSE)
  nms <- names(dt)
  
  dataset_col <- find_first_col(nms, c("analysis_dataset", "dataset", "dataset_id"))
  behavior_col <- find_first_col(nms, c("behavior", "behaviour", "target_behavior", "target_behaviour", "class", "label"))
  behavior_count_col <- find_first_col(nms, c("n_rows_behavior", "behavior_n_rows", "behavior_rows", "n_behavior", "count_behavior", "class_n", "class_count", "freq", "frequency"))
  prop_col <- find_first_col(nms, c("prop", "proportion", "share", "fraction", "percent", "pct"))
  total_col <- find_first_col(nms, c("total_rows", "dataset_rows", "dataset_n_rows", "n_rows_total", "total_n_rows", "n_rows"))
  
  if (any(is.na(c(dataset_col, behavior_col)))) {
    stop("Could not detect required columns in ", basename(file))
  }
  
  out <- dt %>%
    transmute(
      analysis_dataset = as.character(.data[[dataset_col]]),
      behavior_raw = as.character(.data[[behavior_col]]),
      behavior_n_rows = if (!is.na(behavior_count_col)) suppressWarnings(as.numeric(.data[[behavior_count_col]])) else NA_real_,
      prop_raw = if (!is.na(prop_col)) .data[[prop_col]] else NA,
      file_total_rows = if (!is.na(total_col)) suppressWarnings(as.numeric(.data[[total_col]])) else NA_real_
    ) %>%
    mutate(
      behavior = normalize_behavior(behavior_raw),
      prop_from_file = parse_prop_to_fraction(prop_raw)
    ) %>%
    filter(!is.na(analysis_dataset), !is.na(behavior))
  
  if (!is.null(dataset_overview)) {
    out <- out %>%
      left_join(dataset_overview %>% select(analysis_dataset, overview_n_rows), by = "analysis_dataset")
  } else {
    out <- out %>% mutate(overview_n_rows = NA_real_)
  }
  
  out <- out %>%
    mutate(total_rows_ref = coalesce(overview_n_rows, file_total_rows))
  
  # Prefer explicit behaviour counts. Otherwise reconstruct from prop * total rows.
  out <- out %>%
    mutate(
      n_rows = case_when(
        is.finite(behavior_n_rows) ~ behavior_n_rows,
        is.finite(prop_from_file) & is.finite(total_rows_ref) ~ prop_from_file * total_rows_ref,
        TRUE ~ NA_real_
      )
    ) %>%
    filter(is.finite(n_rows)) %>%
    group_by(analysis_dataset, behavior) %>%
    summarise(n_rows = sum(n_rows), .groups = "drop") %>%
    complete(analysis_dataset, behavior = target_behaviors, fill = list(n_rows = 0))
  
  out
}

read_dataset_overview <- function(file) {
  dt <- read_csv(file, show_col_types = FALSE)
  nms <- names(dt)
  
  dataset_col <- find_first_col(nms, c("analysis_dataset", "dataset", "dataset_id"))
  n_rows_col <- find_first_col(nms, c("n_rows", "n", "rows", "count"))
  species_col <- find_first_col(nms, c("species", "species_name"))
  individuals_col <- find_first_col(nms, c("n_individuals", "individuals", "n_ids"))
  files_col <- find_first_col(nms, c("n_source_files", "source_files", "n_files"))
  
  if (any(is.na(c(dataset_col, n_rows_col)))) {
    stop("Could not detect required columns in ", basename(file))
  }
  
  out <- dt %>%
    transmute(
      analysis_dataset = as.character(.data[[dataset_col]]),
      overview_n_rows = as.numeric(.data[[n_rows_col]]),
      species = if (!is.na(species_col)) as.character(.data[[species_col]]) else NA_character_,
      n_individuals = if (!is.na(individuals_col)) as.numeric(.data[[individuals_col]]) else NA_real_,
      n_source_files = if (!is.na(files_col)) as.numeric(.data[[files_col]]) else NA_real_
    ) %>%
    distinct()
  
  out
}

read_model_output_distribution <- function(model_roots) {
  for (root in model_roots) {
    counts_file <- file.path(root, "counts_by_dataset_species_behavior.csv")
    lookup_file <- file.path(root, "analysis_dataset_lookup.csv")
    
    if (!file.exists(counts_file)) {
      next
    }
    
    counts_raw <- read_csv(counts_file, show_col_types = FALSE)
    nms <- names(counts_raw)
    
    dataset_col <- find_first_col(nms, c("analysis_dataset", "dataset_name", "dataset", "dataset_id"))
    behavior_col <- find_first_col(nms, c("behavior", "behaviour", "behavior_en", "main_class", "class", "label"))
    n_col <- find_first_col(nms, c("N", "n", "rows", "n_rows", "count"))
    species_col <- find_first_col(nms, c("species", "species_name"))
    
    if (any(is.na(c(dataset_col, behavior_col, n_col)))) {
      next
    }
    
    behavior_long <- counts_raw %>%
      transmute(
        analysis_dataset = as.character(.data[[dataset_col]]),
        species = if (!is.na(species_col)) as.character(.data[[species_col]]) else NA_character_,
        behavior = normalize_behavior(.data[[behavior_col]]),
        n_rows = suppressWarnings(as.numeric(.data[[n_col]]))
      ) %>%
      filter(!is.na(analysis_dataset), !is.na(behavior), is.finite(n_rows)) %>%
      group_by(analysis_dataset, species, behavior) %>%
      summarise(n_rows = sum(n_rows), .groups = "drop") %>%
      group_by(analysis_dataset, behavior) %>%
      summarise(
        species = dplyr::first(stats::na.omit(species)),
        n_rows = sum(n_rows),
        .groups = "drop"
      ) %>%
      complete(analysis_dataset, behavior = target_behaviors, fill = list(n_rows = 0))
    
    dataset_overview <- behavior_long %>%
      group_by(analysis_dataset) %>%
      summarise(
        overview_n_rows = sum(n_rows, na.rm = TRUE),
        species = dplyr::first(stats::na.omit(species)),
        n_individuals = NA_real_,
        n_source_files = NA_real_,
        .groups = "drop"
      )
    
    if (file.exists(lookup_file)) {
      lookup_raw <- read_csv(lookup_file, show_col_types = FALSE)
      lookup_nms <- names(lookup_raw)
      lookup_dataset_col <- find_first_col(lookup_nms, c("analysis_dataset", "dataset_name", "dataset", "dataset_id"))
      lookup_rows_col <- find_first_col(lookup_nms, c("rows", "N", "n", "n_rows", "count"))
      lookup_species_col <- find_first_col(lookup_nms, c("species", "species_name"))
      lookup_names_col <- find_first_col(lookup_nms, c("n_names", "n_individuals", "individuals", "n_ids"))
      lookup_subject_col <- find_first_col(lookup_nms, c("n_subject_keys", "n_subjects"))
      lookup_files_col <- find_first_col(lookup_nms, c("n_source_files", "source_files", "n_files"))
      
      if (!is.na(lookup_dataset_col)) {
        lookup_overview <- lookup_raw %>%
          transmute(
            analysis_dataset = as.character(.data[[lookup_dataset_col]]),
            rows = if (!is.na(lookup_rows_col)) suppressWarnings(as.numeric(.data[[lookup_rows_col]])) else NA_real_,
            species = if (!is.na(lookup_species_col)) as.character(.data[[lookup_species_col]]) else NA_character_,
            n_names = if (!is.na(lookup_names_col)) suppressWarnings(as.numeric(.data[[lookup_names_col]])) else NA_real_,
            n_subject_keys = if (!is.na(lookup_subject_col)) suppressWarnings(as.numeric(.data[[lookup_subject_col]])) else NA_real_,
            n_source_files = if (!is.na(lookup_files_col)) suppressWarnings(as.numeric(.data[[lookup_files_col]])) else NA_real_
          ) %>%
          group_by(analysis_dataset) %>%
          summarise(
            lookup_rows = sum(rows, na.rm = TRUE),
            lookup_species = dplyr::first(stats::na.omit(species)),
            lookup_n_individuals = suppressWarnings(max(c(n_names, n_subject_keys), na.rm = TRUE)),
            lookup_n_source_files = suppressWarnings(max(n_source_files, na.rm = TRUE)),
            .groups = "drop"
          ) %>%
          mutate(
            lookup_rows = if_else(is.finite(lookup_rows) & lookup_rows > 0, lookup_rows, NA_real_),
            lookup_n_individuals = if_else(is.finite(lookup_n_individuals), lookup_n_individuals, NA_real_),
            lookup_n_source_files = if_else(is.finite(lookup_n_source_files), lookup_n_source_files, NA_real_)
          )
        
        dataset_overview <- dataset_overview %>%
          left_join(lookup_overview, by = "analysis_dataset") %>%
          mutate(
            overview_n_rows = coalesce(lookup_rows, overview_n_rows),
            species = coalesce(lookup_species, species),
            n_individuals = lookup_n_individuals,
            n_source_files = lookup_n_source_files
          ) %>%
          select(analysis_dataset, overview_n_rows, species, n_individuals, n_source_files)
      }
    }
    
    source_info <- tibble(
      source_type = "model_output_counts",
      source_root = root,
      counts_file = counts_file,
      lookup_file = ifelse(file.exists(lookup_file), lookup_file, NA_character_)
    )
    
    return(list(
      behavior_long = behavior_long %>% select(analysis_dataset, behavior, n_rows),
      dataset_overview = dataset_overview,
      source_info = source_info
    ))
  }
  
  NULL
}

read_supplement_distribution <- function(behavior_candidates, overview_candidates) {
  behavior_file <- behavior_candidates[file.exists(behavior_candidates)][1]
  overview_file <- overview_candidates[file.exists(overview_candidates)][1]
  
  if (length(behavior_file) == 0 || is.na(behavior_file) ||
      length(overview_file) == 0 || is.na(overview_file)) {
    return(NULL)
  }
  
  dataset_overview <- read_dataset_overview(overview_file)
  behavior_long <- read_behavior_distribution(behavior_file, dataset_overview = dataset_overview)
  
  list(
    behavior_long = behavior_long,
    dataset_overview = dataset_overview,
    source_info = tibble(
      source_type = "supplement_csv",
      source_root = dirname(dirname(behavior_file)),
      counts_file = behavior_file,
      lookup_file = overview_file
    )
  )
}

load_dataset_distribution_inputs <- function() {
  from_model_outputs <- read_model_output_distribution(model_summary_roots)
  if (!is.null(from_model_outputs)) {
    return(from_model_outputs)
  }
  
  from_supplement <- read_supplement_distribution(
    supp_behavior_file_candidates,
    supp_overview_file_candidates
  )
  if (!is.null(from_supplement)) {
    return(from_supplement)
  }
  
  stop(
    "Could not find dataset behaviour distributions. Expected either:
",
    "- Models/<Model>/counts_by_dataset_species_behavior.csv in the new structure, or
",
    "- Supplement CSV files dataset_behavior_distribution.csv and dataset_overview.csv."
  )
}


# ---- 6.3 Functional-biomechanical distance and statistics ---------------------------------------
make_core_traits_ordered <- function(dt) {
  dt %>%
    mutate(
      log_bodymass_g = as.numeric(log_bodymass_g),
      expected_trunk_motion_amplitude = ordered(expected_trunk_motion_amplitude, levels = c("low", "moderate", "high", "very_high")),
      directional_change_frequency = ordered(directional_change_frequency, levels = c("low", "moderate", "high", "very_high")),
      head_neck_contribution_to_whole_body_motion = ordered(head_neck_contribution_to_whole_body_motion, levels = c("low", "moderate", "high", "very_high")),
      postural_compactness_body_profile_reduction = ordered(postural_compactness_body_profile_reduction, levels = c("low", "moderate", "high", "very_high")),
      across(c(dominant_locomotor_mode, body_configuration_type), as.factor)
    )
}

compute_gower_long <- function(trait_df, value_name = "trait_distance_core") {
  mat <- as.matrix(daisy(trait_df %>% select(-species), metric = "gower"))
  rownames(mat) <- trait_df$species
  colnames(mat) <- trait_df$species
  
  as.data.frame(mat) %>%
    rownames_to_column("train_species") %>%
    pivot_longer(-train_species, names_to = "test_species", values_to = value_name)
}

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

safe_entropy_norm <- function(p) {
  p <- as.numeric(p)
  p <- p[is.finite(p)]
  if (length(p) == 0 || sum(p) <= 0) return(NA_real_)
  p <- p / sum(p)
  p <- p[p > 0]
  if (length(p) == 0) return(NA_real_)
  -sum(p * log(p)) / log(length(target_behaviors))
}

safe_kl <- function(p, q, log_base = 2) {
  p <- as.numeric(p)
  q <- as.numeric(q)
  ok <- p > 0 & q > 0 & is.finite(p) & is.finite(q)
  if (!any(ok)) return(0)
  sum(p[ok] * (log(p[ok] / q[ok]) / log(log_base)))
}

safe_jsd <- function(p, q, log_base = 2) {
  p <- as.numeric(p)
  q <- as.numeric(q)
  if (any(!is.finite(c(p, q)))) return(NA_real_)
  if (sum(p) <= 0 || sum(q) <= 0) return(NA_real_)
  p <- p / sum(p)
  q <- q / sum(q)
  m <- (p + q) / 2
  0.5 * safe_kl(p, m, log_base = log_base) + 0.5 * safe_kl(q, m, log_base = log_base)
}

safe_cor_matrix <- function(df, cols) {
  keep <- df %>% select(all_of(cols))
  if (ncol(keep) < 2) return(tibble())
  cor_mat <- suppressWarnings(cor(keep, use = "pairwise.complete.obs", method = "spearman"))
  as.data.frame(cor_mat) %>%
    rownames_to_column("var_1") %>%
    pivot_longer(-var_1, names_to = "var_2", values_to = "rho")
}


# ---- 6.4 Modeling ------------------------------------------------------------
safe_z <- function(x) {
  x <- as.numeric(x)
  mu <- mean(x, na.rm = TRUE)
  sd_x <- stats::sd(x, na.rm = TRUE)
  if (!is.finite(sd_x) || sd_x == 0) {
    return(rep(0, length(x)))
  }
  (x - mu) / sd_x
}

safe_lm <- function(df, formula_obj) {
  fit <- lm(formula_obj, data = df)
  sm <- summary(fit)
  coef_tbl <- as.data.frame(sm$coefficients)
  coef_tbl$term <- rownames(coef_tbl)
  rownames(coef_tbl) <- NULL
  names(coef_tbl) <- c("estimate", "std_error", "statistic", "p_value", "term")
  
  ci <- suppressMessages(confint(fit)) %>%
    as.data.frame() %>%
    rownames_to_column("term") %>%
    setNames(c("term", "conf_low", "conf_high"))
  
  coef_tbl %>%
    left_join(ci, by = "term") %>%
    mutate(
      n = nobs(fit),
      r_squared = sm$r.squared,
      adj_r_squared = sm$adj.r.squared,
      aic = AIC(fit),
      formula = paste(deparse(formula_obj), collapse = " ")
    )
}


# ---- 6.5 Plotting ------------------------------------------------------------
theme_clean <- function(base_size = 11) {
  theme_classic(base_size = base_size) %+replace%
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
      plot.subtitle = element_text(hjust = 0.5, size = 10, color = "grey40"),
      axis.text = element_text(size = 9.5, color = "black"),
      axis.title = element_text(size = 11),
      axis.ticks = element_line(linewidth = 0.35, color = "black"),
      axis.line = element_line(linewidth = 0.35, color = "black"),
      strip.text = element_text(face = "bold", size = 10.5),
      strip.background = element_blank(),
      legend.position = "right"
    )
}

save_plot <- function(plot_obj, filename, width = 9, height = 6, dpi = 300) {
  ggsave(
    filename = file.path(plots_dir, filename),
    plot = plot_obj,
    width = width,
    height = height,
    dpi = dpi,
    bg = "white"
  )
}


# ── 7. Input checks ───────────────────────────────────────────────────────────
must_exist(c(manual_traits_core_file))


# ── 8. Dataset -> species mapping ─────────────────────────────────────────────
dataset_to_species <- c(
  "Bison" = "Bison bonasus",
  "Bison_dataset_1" = "Bison bonasus",
  "Cattle" = "Bos taurus",
  "Cattle_dataset_1" = "Bos taurus",
  "Dog" = "Canis lupus familiaris",
  "Dog_dataset_1" = "Canis lupus familiaris",
  "Dog_dataset_2" = "Canis lupus familiaris",
  "Fox_dataset_1" = "Vulpes vulpes",
  "Fox_dataset_2" = "Vulpes vulpes",
  "Giraffe" = "Giraffa camelopardalis",
  "Giraffe_dataset_1" = "Giraffa camelopardalis",
  "Hedgehog" = "Erinaceus europaeus",
  "Hedgehog_dataset_1" = "Erinaceus europaeus",
  "Horse_dataset_1" = "Equus ferus przewalskii",
  "Horse_dataset_2" = "Equus caballus",
  "Raccoon_dataset_1" = "Procyon lotor",
  "Raccoon_dataset_2" = "Procyon lotor"
)


# ── 9. Read inputs ────────────────────────────────────────────────────────────
print_section("READ INPUTS")

manual_traits_core <- read_trait_file(manual_traits_core_file, required_core_trait_cols) %>%
  make_core_traits_ordered()

gower_core_long <- compute_gower_long(manual_traits_core, "trait_distance_core")

pairwise <- pmap_dfr(model_files_used, read_pairwise_model) %>%
  mutate(
    train_species = recode(train_dataset, !!!dataset_to_species, .default = NA_character_),
    test_species = recode(test_dataset, !!!dataset_to_species, .default = NA_character_),
    same_species = train_species == test_species,
    model = factor(model, levels = model_order)
  )

dataset_inputs <- load_dataset_distribution_inputs()
dataset_overview <- dataset_inputs$dataset_overview
behavior_long <- dataset_inputs$behavior_long
source_info <- dataset_inputs$source_info
write_csv(source_info, file.path(csv_dir, "00_dataset_distribution_source.csv"))

input_overview <- pairwise %>%
  dplyr::count(model, name = "pairwise_rows") %>%
  left_join(pairwise %>% distinct(model, train_dataset) %>% dplyr::count(model, name = "n_train_datasets"), by = "model") %>%
  left_join(pairwise %>% distinct(model, test_dataset) %>% dplyr::count(model, name = "n_test_datasets"), by = "model")

print(input_overview)


# ── 10. Coverage checks ───────────────────────────────────────────────────────
print_section("COVERAGE CHECKS")

unmapped_train <- pairwise %>% filter(is.na(train_species)) %>% distinct(train_dataset)
unmapped_test  <- pairwise %>% filter(is.na(test_species)) %>% distinct(test_dataset)
missing_behavior_datasets <- pairwise %>%
  distinct(train_dataset) %>%
  rename(analysis_dataset = train_dataset) %>%
  bind_rows(pairwise %>% distinct(test_dataset) %>% rename(analysis_dataset = test_dataset)) %>%
  distinct() %>%
  anti_join(behavior_long %>% distinct(analysis_dataset), by = "analysis_dataset")

cat("Unmapped train datasets:", nrow(unmapped_train), "\n")
if (nrow(unmapped_train) > 0) print(unmapped_train)
cat("Unmapped test datasets:", nrow(unmapped_test), "\n")
if (nrow(unmapped_test) > 0) print(unmapped_test)
cat("Datasets missing behavior distribution:", nrow(missing_behavior_datasets), "\n")
if (nrow(missing_behavior_datasets) > 0) print(missing_behavior_datasets)

if (nrow(unmapped_train) > 0 || nrow(unmapped_test) > 0 || nrow(missing_behavior_datasets) > 0) {
  stop("Coverage check failed. Fix dataset mapping or supplementary dataset tables first.")
}


# ── 11. Dataset profiles ──────────────────────────────────────────────────────
print_section("DATASET PROFILES")

dataset_profiles <- behavior_long %>%
  rename(n_rows_behavior = n_rows) %>%
  group_by(analysis_dataset) %>%
  mutate(total_from_behavior = sum(n_rows_behavior, na.rm = TRUE)) %>%
  ungroup() %>%
  left_join(
    dataset_overview %>% select(analysis_dataset, overview_n_rows, species, n_individuals, n_source_files),
    by = "analysis_dataset"
  ) %>%
  mutate(
    total_rows = if_else(is.finite(overview_n_rows), overview_n_rows, total_from_behavior),
    prop = if_else(total_rows > 0, n_rows_behavior / total_rows, NA_real_)
  ) %>%
  select(analysis_dataset, species, n_individuals, n_source_files, behavior, n_rows_behavior, total_rows, prop)

dataset_profile_wide <- dataset_profiles %>%
  select(analysis_dataset, species, n_individuals, n_source_files, total_rows, behavior, prop) %>%
  distinct() %>%
  pivot_wider(
    id_cols = c(analysis_dataset, species, n_individuals, n_source_files, total_rows),
    names_from = behavior,
    values_from = prop,
    names_prefix = "prop_",
    values_fill = 0
  ) %>%
  mutate(
    entropy_norm = pmap_dbl(list(prop_Foraging, prop_Locomotion, prop_Resting), ~ safe_entropy_norm(c(..1, ..2, ..3))),
    majority_prop = pmax(prop_Foraging, prop_Locomotion, prop_Resting, na.rm = TRUE),
    minority_prop = pmin(prop_Foraging, prop_Locomotion, prop_Resting, na.rm = TRUE),
    imbalance_range = majority_prop - minority_prop,
    dominant_behavior = case_when(
      prop_Foraging >= prop_Locomotion & prop_Foraging >= prop_Resting ~ "Foraging",
      prop_Locomotion >= prop_Foraging & prop_Locomotion >= prop_Resting ~ "Locomotion",
      TRUE ~ "Resting"
    )
  ) %>%
  arrange(desc(total_rows), analysis_dataset)

print(dataset_profile_wide %>% select(analysis_dataset, total_rows, starts_with("prop_"), entropy_norm, majority_prop, dominant_behavior))

prop_sanity <- dataset_profile_wide %>%
  transmute(analysis_dataset, prop_sum = prop_Foraging + prop_Locomotion + prop_Resting)

print(prop_sanity)


# ── 12. Merge H3 inputs ───────────────────────────────────────────────────────
print_section("MERGE H3 INPUTS")

analysis_dt <- pairwise %>%
  left_join(gower_core_long, by = c("train_species", "test_species")) %>%
  filter(is.finite(trait_distance_core), is.finite(macro_f1))

dataset_pair_level <- analysis_dt %>%
  group_by(model, train_dataset, test_dataset, train_species, test_species, same_species, trait_distance_core) %>%
  summarise(
    mean_accuracy = mean(accuracy, na.rm = TRUE),
    mean_macro_recall = mean(macro_recall, na.rm = TRUE),
    mean_macro_precision = mean(macro_precision, na.rm = TRUE),
    mean_macro_f1 = mean(macro_f1, na.rm = TRUE),
    n_rows = n(),
    .groups = "drop"
  ) %>%
  filter(!same_species)

pair_bias <- dataset_pair_level %>%
  left_join(
    dataset_profile_wide %>%
      rename_with(~ paste0("train_", .x), -analysis_dataset) %>%
      rename(train_dataset = analysis_dataset),
    by = "train_dataset"
  ) %>%
  left_join(
    dataset_profile_wide %>%
      rename_with(~ paste0("test_", .x), -analysis_dataset) %>%
      rename(test_dataset = analysis_dataset),
    by = "test_dataset"
  ) %>%
  mutate(
    class_shift_jsd = pmap_dbl(
      list(train_prop_Foraging, train_prop_Locomotion, train_prop_Resting,
           test_prop_Foraging, test_prop_Locomotion, test_prop_Resting),
      ~ safe_jsd(c(..1, ..2, ..3), c(..4, ..5, ..6))
    ),
    class_shift_l1 = pmap_dbl(
      list(train_prop_Foraging, train_prop_Locomotion, train_prop_Resting,
           test_prop_Foraging, test_prop_Locomotion, test_prop_Resting),
      ~ sum(abs(c(..1, ..2, ..3) - c(..4, ..5, ..6)), na.rm = TRUE)
    ),
    entropy_gap = abs(train_entropy_norm - test_entropy_norm),
    majority_gap = abs(train_majority_prop - test_majority_prop),
    size_gap_log10 = abs(log10(train_total_rows) - log10(test_total_rows)),
    size_ratio = train_total_rows / test_total_rows,
    dominant_match = train_dominant_behavior == test_dominant_behavior
  )

pair_bias_missing <- pair_bias %>%
  summarise(
    missing_class_shift_jsd = sum(!is.finite(class_shift_jsd)),
    missing_size_gap_log10 = sum(!is.finite(size_gap_log10)),
    missing_trait_distance_core = sum(!is.finite(trait_distance_core))
  )

print(pair_bias_missing)


# ── 13. Bias summaries ────────────────────────────────────────────────────────
print_section("BIAS SUMMARIES")

behavior_totals <- dataset_profiles %>%
  group_by(behavior) %>%
  summarise(n_rows = sum(n_rows_behavior), .groups = "drop") %>%
  mutate(percent_total = percent(n_rows / sum(n_rows), accuracy = 0.1))

bias_overview <- pair_bias %>%
  summarise(
    n_rows = n(),
    n_models = dplyr::n_distinct(model),
    n_train_datasets = dplyr::n_distinct(train_dataset),
    n_test_datasets = dplyr::n_distinct(test_dataset),
    mean_trait_distance_core = mean(trait_distance_core, na.rm = TRUE),
    mean_class_shift_jsd = mean(class_shift_jsd, na.rm = TRUE),
    mean_class_shift_l1 = mean(class_shift_l1, na.rm = TRUE),
    mean_size_gap_log10 = mean(size_gap_log10, na.rm = TRUE),
    mean_macro_f1 = mean(mean_macro_f1, na.rm = TRUE)
  )

print(behavior_totals)
print(bias_overview)


# ── 14. Correlation checks ────────────────────────────────────────────────────
print_section("CORRELATION CHECKS")

spearman_checks <- bind_rows(
  safe_spearman(pair_bias$trait_distance_core, pair_bias$mean_macro_f1) %>%
    mutate(model = "overall", predictor = "trait_distance_core"),
  safe_spearman(pair_bias$class_shift_jsd, pair_bias$mean_macro_f1) %>%
    mutate(model = "overall", predictor = "class_shift_jsd"),
  safe_spearman(pair_bias$class_shift_l1, pair_bias$mean_macro_f1) %>%
    mutate(model = "overall", predictor = "class_shift_l1"),
  safe_spearman(pair_bias$size_gap_log10, pair_bias$mean_macro_f1) %>%
    mutate(model = "overall", predictor = "size_gap_log10"),
  pair_bias %>%
    group_by(model) %>%
    group_modify(~ bind_rows(
      safe_spearman(.x$trait_distance_core, .x$mean_macro_f1) %>% mutate(predictor = "trait_distance_core"),
      safe_spearman(.x$class_shift_jsd, .x$mean_macro_f1) %>% mutate(predictor = "class_shift_jsd"),
      safe_spearman(.x$class_shift_l1, .x$mean_macro_f1) %>% mutate(predictor = "class_shift_l1"),
      safe_spearman(.x$size_gap_log10, .x$mean_macro_f1) %>% mutate(predictor = "size_gap_log10")
    )) %>%
    ungroup()
) %>%
  select(model, predictor, n, rho, p_value)

predictor_correlations <- safe_cor_matrix(
  pair_bias,
  cols = c("trait_distance_core", "class_shift_jsd", "class_shift_l1", "size_gap_log10", "mean_macro_f1")
)

print(spearman_checks)


# ── 15. Quick control models ──────────────────────────────────────────────────
print_section("CONTROL MODELS")

pair_bias_model_dt <- pair_bias %>%
  mutate(
    z_trait_distance_core = safe_z(trait_distance_core),
    z_class_shift_jsd = safe_z(class_shift_jsd),
    z_size_gap_log10 = safe_z(size_gap_log10)
  ) %>%
  filter(
    is.finite(mean_macro_f1),
    is.finite(z_trait_distance_core),
    is.finite(z_class_shift_jsd),
    is.finite(z_size_gap_log10)
  ) %>%
  droplevels()

lm_overall_unadjusted <- safe_lm(pair_bias_model_dt, mean_macro_f1 ~ z_trait_distance_core + model) %>%
  mutate(model_block = "overall_unadjusted")

lm_overall_adjusted <- safe_lm(pair_bias_model_dt, mean_macro_f1 ~ z_trait_distance_core + z_class_shift_jsd + z_size_gap_log10 + model) %>%
  mutate(model_block = "overall_adjusted")

lm_per_model <- pair_bias_model_dt %>%
  group_by(model) %>%
  group_modify(~ safe_lm(.x, mean_macro_f1 ~ z_trait_distance_core + z_class_shift_jsd + z_size_gap_log10)) %>%
  ungroup() %>%
  mutate(model_block = paste0("per_model_", model))

trait_beta_compare <- bind_rows(lm_overall_unadjusted, lm_overall_adjusted) %>%
  filter(term == "z_trait_distance_core") %>%
  select(model_block, term, estimate, std_error, statistic, p_value, conf_low, conf_high, n, r_squared, adj_r_squared, aic)

print(trait_beta_compare)


# ── 16. Plot data ─────────────────────────────────────────────────────────────
print_section("PLOTS")

plot_dataset_order <- dataset_profile_wide %>%
  arrange(desc(total_rows), analysis_dataset) %>%
  pull(analysis_dataset)

plot_class_heatmap <- dataset_profiles %>%
  mutate(analysis_dataset = factor(analysis_dataset, levels = rev(plot_dataset_order))) %>%
  ggplot(aes(x = behavior, y = analysis_dataset, fill = prop)) +
  geom_tile(color = "white", linewidth = 0.35) +
  geom_text(aes(label = percent(prop, accuracy = 0.1)), size = 3) +
  scale_fill_gradient(low = "grey95", high = "grey25", limits = c(0, 1), labels = percent_format(accuracy = 1)) +
  labs(
    title = "Per-dataset distribution of target behaviours",
    x = NULL,
    y = NULL,
    fill = "Share"
  ) +
  theme_clean()

plot_dataset_size <- dataset_profile_wide %>%
  mutate(analysis_dataset = factor(analysis_dataset, levels = plot_dataset_order)) %>%
  ggplot(aes(x = analysis_dataset, y = total_rows)) +
  geom_col(fill = "grey35", width = 0.75) +
  scale_y_continuous(trans = "log10", labels = comma_format()) +
  labs(
    title = "Dataset size overview",
    x = NULL,
    y = "Rows (log10 scale)"
  ) +
  theme_clean() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

plot_f1_vs_jsd <- pair_bias %>%
  ggplot(aes(x = class_shift_jsd, y = mean_macro_f1)) +
  geom_point(size = 2, alpha = 0.75, color = "grey25") +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.6, color = "black") +
  facet_wrap(~ model, ncol = 3) +
  scale_x_continuous(labels = function(x) sprintf("%.2f", x)) +
  scale_y_continuous(limits = c(0, 1), labels = function(x) sprintf("%.2f", x)) +
  labs(
    title = "Macro-F1 vs class-distribution shift",
    subtitle = "Dataset-pair level, different species only",
    x = "Jensen-Shannon divergence",
    y = "Mean macro-F1"
  ) +
  theme_clean()

coef_plot_dt <- bind_rows(
  lm_overall_adjusted %>% mutate(source = "overall"),
  lm_per_model %>% mutate(source = as.character(model))
) %>%
  filter(term %in% c("z_trait_distance_core", "z_class_shift_jsd", "z_size_gap_log10")) %>%
  mutate(
    term = recode(
      term,
      z_trait_distance_core = "Functional-biomechanical distance",
      z_class_shift_jsd = "Class shift",
      z_size_gap_log10 = "Size gap"
    ),
    source = factor(source, levels = c("overall", model_order))
  )

plot_coefficients <- coef_plot_dt %>%
  ggplot(aes(x = estimate, y = term)) +
  geom_vline(xintercept = 0, linewidth = 0.35, linetype = 2, color = "grey50") +
  geom_errorbarh(aes(xmin = conf_low, xmax = conf_high), height = 0.15, linewidth = 0.45, color = "grey30") +
  geom_point(size = 2.2, color = "black") +
  facet_wrap(~ source, ncol = 3) +
  labs(
    title = "",
    subtitle = "",
    x = "Estimate",
    y = NULL
  ) +
  theme_clean()

save_plot(plot_class_heatmap, "01_dataset_class_balance_heatmap.png", width = 8.5, height = 5.5)
save_plot(plot_dataset_size, "02_dataset_size_barplot.png", width = 9, height = 5.5)
save_plot(plot_f1_vs_jsd, "03_macro_f1_vs_class_shift_jsd_by_model.png", width = 10, height = 6.5)
save_plot(plot_coefficients, "04_h3_bias_coefficients.png", width = 10, height = 6.5)

plot_index <- tibble(
  file = c(
    "01_dataset_class_balance_heatmap.png",
    "02_dataset_size_barplot.png",
    "03_macro_f1_vs_class_shift_jsd_by_model.png",
    "04_h3_bias_coefficients.png"
  ),
  description = c(
    "Per-dataset heatmap of the three target behaviours",
    "Dataset size overview on log10 scale",
    "Macro-F1 vs class-distribution shift, faceted by model",
    "Scaled adjusted coefficients for functional-biomechanical distance, class shift, and size gap"
  )
)


# ── 17. Export CSV ────────────────────────────────────────────────────────────
print_section("WRITE OUTPUTS")

write_csv(behavior_totals, file.path(csv_dir, "01_behavior_totals.csv"))
write_csv(dataset_profile_wide, file.path(csv_dir, "02_dataset_class_profiles.csv"))
write_csv(dataset_profiles, file.path(csv_dir, "03_dataset_class_distribution_long.csv"))
write_csv(dataset_pair_level, file.path(csv_dir, "04_dataset_pair_transfer_metrics.csv"))
write_csv(pair_bias, file.path(csv_dir, "05_dataset_pair_bias_metrics.csv"))
write_csv(spearman_checks, file.path(csv_dir, "06_spearman_bias_checks.csv"))
write_csv(predictor_correlations, file.path(csv_dir, "07_predictor_spearman_matrix.csv"))
write_csv(bind_rows(lm_overall_unadjusted, lm_overall_adjusted, lm_per_model), file.path(csv_dir, "08_control_model_coefficients.csv"))
write_csv(trait_beta_compare, file.path(csv_dir, "09_trait_distance_beta_compare.csv"))
write_csv(plot_index, file.path(csv_dir, "10_plot_index.csv"))


# ── 18. Export TXT ────────────────────────────────────────────────────────────
trait_unadj_row <- lm_overall_unadjusted %>% filter(term == "z_trait_distance_core") %>% slice(1)
trait_adj_row <- lm_overall_adjusted %>% filter(term == "z_trait_distance_core") %>% slice(1)
jsd_adj_row <- lm_overall_adjusted %>% filter(term == "z_class_shift_jsd") %>% slice(1)
size_adj_row <- lm_overall_adjusted %>% filter(term == "z_size_gap_log10") %>% slice(1)

summary_lines <- c(
  "# ===================================================",
  "# H3_bias_class_shift Summary",
  "# ===================================================",
  "",
  paste0("Date: ", Sys.time()),
  paste0("Core traits: ", basename(manual_traits_core_file)),
  paste0("Dataset distribution source: ", source_info$source_type[[1]]),
  paste0("Dataset distribution root: ", source_info$source_root[[1]]),
  paste0("Models used: ", paste(as.character(unique(pairwise$model)), collapse = ", ")),
  "",
  "# ----- Overall class balance -----",
  paste0("Foraging:   ", behavior_totals$percent_total[behavior_totals$behavior == "Foraging"]),
  paste0("Locomotion:", behavior_totals$percent_total[behavior_totals$behavior == "Locomotion"]),
  paste0("Resting:    ", behavior_totals$percent_total[behavior_totals$behavior == "Resting"]),
  "",
  "# ----- Overall Spearman checks (dataset-pair level, different species only) -----",
  paste0(
    "Functional-biomechanical distance: rho = ",
    fmt_num(spearman_checks$rho[spearman_checks$model == "overall" & spearman_checks$predictor == "trait_distance_core"]),
    ", p = ",
    fmt_p(spearman_checks$p_value[spearman_checks$model == "overall" & spearman_checks$predictor == "trait_distance_core"]),
    ", n = ",
    spearman_checks$n[spearman_checks$model == "overall" & spearman_checks$predictor == "trait_distance_core"]
  ),
  paste0(
    "Class shift (JSD): rho = ",
    fmt_num(spearman_checks$rho[spearman_checks$model == "overall" & spearman_checks$predictor == "class_shift_jsd"]),
    ", p = ",
    fmt_p(spearman_checks$p_value[spearman_checks$model == "overall" & spearman_checks$predictor == "class_shift_jsd"]),
    ", n = ",
    spearman_checks$n[spearman_checks$model == "overall" & spearman_checks$predictor == "class_shift_jsd"]
  ),
  paste0(
    "Size gap (log10): rho = ",
    fmt_num(spearman_checks$rho[spearman_checks$model == "overall" & spearman_checks$predictor == "size_gap_log10"]),
    ", p = ",
    fmt_p(spearman_checks$p_value[spearman_checks$model == "overall" & spearman_checks$predictor == "size_gap_log10"]),
    ", n = ",
    spearman_checks$n[spearman_checks$model == "overall" & spearman_checks$predictor == "size_gap_log10"]
  ),
  "",
  "# ----- Trait-distance coefficient check -----",
  paste0(
    "Unadjusted trait beta: ", fmt_num(trait_unadj_row$estimate),
    " [", fmt_num(trait_unadj_row$conf_low), ", ", fmt_num(trait_unadj_row$conf_high), "]",
    ", p = ", fmt_p(trait_unadj_row$p_value),
    ", R2 = ", fmt_num(trait_unadj_row$r_squared)
  ),
  paste0(
    "Adjusted trait beta:   ", fmt_num(trait_adj_row$estimate),
    " [", fmt_num(trait_adj_row$conf_low), ", ", fmt_num(trait_adj_row$conf_high), "]",
    ", p = ", fmt_p(trait_adj_row$p_value),
    ", R2 = ", fmt_num(trait_adj_row$r_squared)
  ),
  paste0(
    "Adjusted class-shift beta: ", fmt_num(jsd_adj_row$estimate),
    " [", fmt_num(jsd_adj_row$conf_low), ", ", fmt_num(jsd_adj_row$conf_high), "]",
    ", p = ", fmt_p(jsd_adj_row$p_value)
  ),
  paste0(
    "Adjusted size-gap beta:    ", fmt_num(size_adj_row$estimate),
    " [", fmt_num(size_adj_row$conf_low), ", ", fmt_num(size_adj_row$conf_high), "]",
    ", p = ", fmt_p(size_adj_row$p_value)
  ),
  "",
  "# ----- Quick verdict -----",
  ifelse(
    is.finite(trait_adj_row$estimate) && trait_adj_row$estimate < 0,
    "Functional-biomechanical distance remains negative after adding class shift and size gap.",
    "Functional-biomechanical distance no longer remains negative after adding class shift and size gap."
  ),
  ifelse(
    is.finite(jsd_adj_row$estimate),
    "Class-distribution shift contributes additional variation and should be reported as a bias control.",
    "Class-distribution shift coefficient could not be estimated."
  )
)

report_lines <- c(
  paste0("H3 bias output folder: ", out_dir),
  paste0("Models input folder: ", models_dir),
  paste0("Trait folder: ", traits_dir),
  paste0("Created on: ", Sys.time()),
  "",
  "This report screens whether the H3 distance effect persists after adding class shift and dataset-size asymmetry.",
  "Core Gower distance uses log_bodymass_g instead of body_size_class.",
  "Analysis level: dataset-pair, different species only.",
  "Primary shift metric: Jensen-Shannon divergence across the three target behaviours.",
  ""
)
report_lines <- append_section(report_lines, "INPUT OVERVIEW", input_overview)
report_lines <- append_section(report_lines, "BEHAVIOR TOTALS", behavior_totals)
report_lines <- append_section(report_lines, "DATASET CLASS PROFILES", dataset_profile_wide)
report_lines <- append_section(report_lines, "PAIR BIAS MISSINGNESS", pair_bias_missing)
report_lines <- append_section(report_lines, "BIAS OVERVIEW", bias_overview)
report_lines <- append_section(report_lines, "SPEARMAN BIAS CHECKS", spearman_checks)
report_lines <- append_section(report_lines, "PREDICTOR CORRELATION MATRIX", predictor_correlations)
report_lines <- append_section(report_lines, "LM OVERALL UNADJUSTED", lm_overall_unadjusted)
report_lines <- append_section(report_lines, "LM OVERALL ADJUSTED", lm_overall_adjusted)
report_lines <- append_section(report_lines, "LM PER MODEL", lm_per_model)
report_lines <- append_section(report_lines, "TRAIT BETA COMPARE", trait_beta_compare)
report_lines <- append_section(report_lines, "PLOT INDEX", plot_index)

write_txt_report(file.path(txt_dir, "01_h3_bias_class_shift_report.txt"), report_lines)
write_txt_report(file.path(txt_dir, "02_h3_bias_class_shift_summary.txt"), summary_lines)
write_txt_report(
  file.path(txt_dir, "03_h3_bias_class_shift_files_written.txt"),
  c(
    paste("CSV:", list.files(csv_dir, full.names = TRUE)),
    paste("PLOT:", list.files(plots_dir, full.names = TRUE)),
    paste("TXT:", list.files(txt_dir, full.names = TRUE))
  )
)


# ── 19. Console output ────────────────────────────────────────────────────────
cat("Done. Output written to:\n", out_dir, "\n", sep = "")
