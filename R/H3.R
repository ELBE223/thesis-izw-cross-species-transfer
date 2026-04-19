library(readr)
library(dplyr)
library(tidyr)
library(tibble)
library(purrr)
library(stringr)
library(cluster)
library(ggplot2)
library(patchwork)
library(grid)
library(gridExtra)
library(scales)

# =============================================================================
# H3 Trait Distance Analysis (revised)
# Main: core biomechanical traits
# Sensitivity: extended trait set
# =============================================================================

# ── User settings ────────────────────────────────────────────────────────────
project_dir <- "/Volumes/Z Slim/07_04_2026_Data_Analysis"
base_dir <- file.path(project_dir, "Output")
traits_dir <- file.path(project_dir, "R_scripts", "H3")

manual_traits_core_file <- file.path(traits_dir, "species_traits_core.csv")
manual_traits_extended_file <- file.path(traits_dir, "species_traits_extended_sensitivity.csv")

out_dir   <- file.path(base_dir, "H3")
csv_dir   <- file.path(out_dir, "csv")
plots_dir <- file.path(out_dir, "plots")
txt_dir   <- file.path(out_dir, "txt")

for (d in c(out_dir, csv_dir, plots_dir, txt_dir)) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

model_files <- tribble(
  ~model,           ~pairwise_file,                                                                                            ~within_file,
  "RF",            file.path(base_dir, "RF",            "Pairwise_RF",            "statistics", "pairwise_metrics_all.csv"),      file.path(base_dir, "RF",            "Within_RF",            "statistics", "within_metrics_all.csv"),
  "LGBM",          file.path(base_dir, "LGBM",          "Pairwise_LGBM",          "statistics", "pairwise_metrics_all.csv"),      file.path(base_dir, "LGBM",          "Within_LGBM",          "statistics", "within_metrics_all.csv"),
  "CNN",           file.path(base_dir, "CNN",           "Pairwise_CNN",           "statistics", "pairwise_summary_metrics.csv"), file.path(base_dir, "CNN",           "Within_CNN",           "statistics", "within_summary_metrics.csv"),
  "InceptionTime", file.path(base_dir, "InceptionTime", "Pairwise_InceptionTime", "statistics", "pairwise_summary_metrics.csv"), file.path(base_dir, "InceptionTime", "Within_InceptionTime", "statistics", "within_summary_metrics.csv"),
  "HYDRA",         file.path(base_dir, "HYDRA",         "Pairwise_HYDRA",         "statistics", "pairwise_summary_metrics.csv"), file.path(base_dir, "HYDRA",         "Within_HYDRA",         "statistics", "within_summary_metrics.csv"),
  "MultiRocket",   file.path(base_dir, "MultiRocket",   "Pairwise_MultiRocket",   "statistics", "pairwise_summary_metrics.csv"), file.path(base_dir, "MultiRocket",   "Within_MultiRocket",   "statistics", "within_summary_metrics.csv")
)

metrics <- c("accuracy", "macro_recall", "macro_precision", "macro_f1")
required_pairwise_cols <- c("pair_id", "train_dataset", "test_dataset", metrics)
required_within_cols   <- c("analysis_dataset", metrics)

required_core_trait_cols <- c(
  "species",
  "body_mass_rank",
  "locomotor_mode",
  "body_plan",
  "movement_amplitude",
  "maneuverability",
  "head_neck_motion_importance",
  "postural_compaction"
)

required_extended_trait_cols <- c(
  "species",
  "body_mass_rank",
  "locomotor_mode",
  "body_plan",
  "movement_amplitude",
  "maneuverability",
  "head_neck_motion_importance",
  "activity_pattern",
  "vertical_motion_scope",
  "defensive_posture_specialisation",
  "postural_compaction"
)

model_order <- c("CNN", "InceptionTime", "HYDRA", "MultiRocket", "RF", "LGBM")

# ── Helpers ──────────────────────────────────────────────────────────────────

must_exist <- function(x) {
  missing <- x[!file.exists(x)]
  if (length(missing) > 0) stop("Missing files:\n", paste(missing, collapse = "\n"))
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

safe_ratio <- function(num, den) {
  out <- num / den
  out[!is.finite(out)] <- NA_real_
  out
}

safe_wilcox <- function(df, value_col, group_col) {
  value <- df[[value_col]]
  group <- df[[group_col]]
  ok <- is.finite(value) & !is.na(group)
  value <- value[ok]
  group <- as.character(group[ok])
  groups <- unique(group)
  
  if (length(groups) != 2) {
    return(tibble(n_group_1 = NA_integer_, n_group_2 = NA_integer_, p_value = NA_real_))
  }
  
  x1 <- value[group == groups[[1]]]
  x2 <- value[group == groups[[2]]]
  
  if (length(x1) < 1 || length(x2) < 1) {
    return(tibble(n_group_1 = length(x1), n_group_2 = length(x2), p_value = NA_real_))
  }
  
  tmp <- suppressWarnings(wilcox.test(x1, x2, exact = FALSE))
  tibble(n_group_1 = length(x1), n_group_2 = length(x2), p_value = tmp$p.value)
}

safe_lm_distance <- function(df, response, predictor) {
  use_dt <- df %>% filter(is.finite(.data[[response]]), is.finite(.data[[predictor]]))
  
  if (nrow(use_dt) < 3) {
    return(tibble(
      n = nrow(use_dt),
      aic = NA_real_,
      r_squared = NA_real_,
      beta_distance = NA_real_,
      p_distance = NA_real_
    ))
  }
  
  fml <- as.formula(paste(response, "~", predictor))
  fit <- lm(fml, data = use_dt)
  sm <- summary(fit)
  coeff <- coef(sm)
  
  tibble(
    n = nobs(fit),
    aic = AIC(fit),
    r_squared = sm$r.squared,
    beta_distance = unname(coeff[predictor, "Estimate"]),
    p_distance = unname(coeff[predictor, "Pr(>|t|)"])
  )
}

print_section <- function(x) {
  cat("\n", strrep("=", 18), x, strrep("=", 18), "\n", sep = " ")
}

append_section <- function(lines, title, obj = NULL) {
  section_header <- c(paste0(strrep("=", 18), " ", title, " ", strrep("=", 18)))
  if (is.null(obj)) {
    return(c(lines, "", section_header, ""))
  }
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

read_pairwise_model <- function(model, pairwise_file) {
  dt <- read_csv(pairwise_file, show_col_types = FALSE)
  missing_cols <- setdiff(required_pairwise_cols, names(dt))
  if (length(missing_cols) > 0) stop(model, " pairwise missing cols: ", paste(missing_cols, collapse = ", "))
  
  dt %>%
    select(any_of(required_pairwise_cols)) %>%
    mutate(model = model, .before = 1)
}

read_within_model <- function(model, within_file) {
  dt <- read_csv(within_file, show_col_types = FALSE)
  missing_cols <- setdiff(required_within_cols, names(dt))
  if (length(missing_cols) > 0) stop(model, " within missing cols: ", paste(missing_cols, collapse = ", "))
  
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
    dup_species <- dt %>% count(species) %>% filter(n > 1)
    stop("Duplicate species in ", basename(file), ":\n", paste(dup_species$species, collapse = "\n"))
  }
  
  if (any(!complete.cases(dt))) {
    stop(basename(file), " contains missing values.")
  }
  
  dt
}

make_core_traits_ordered <- function(dt) {
  dt %>%
    mutate(
      body_mass_rank = ordered(body_mass_rank, levels = c(1, 2, 3, 4)),
      movement_amplitude = ordered(movement_amplitude, levels = c("low", "moderate", "high", "very_high")),
      maneuverability = ordered(maneuverability, levels = c("low", "moderate", "high", "very_high")),
      head_neck_motion_importance = ordered(head_neck_motion_importance, levels = c("low", "moderate", "high", "very_high")),
      postural_compaction = ordered(postural_compaction, levels = c("low", "moderate", "high", "very_high")),
      across(c(locomotor_mode, body_plan), as.factor)
    )
}

make_extended_traits_ordered <- function(dt) {
  dt %>%
    mutate(
      body_mass_rank = ordered(body_mass_rank, levels = c(1, 2, 3, 4)),
      movement_amplitude = ordered(movement_amplitude, levels = c("low", "moderate", "high", "very_high")),
      maneuverability = ordered(maneuverability, levels = c("low", "moderate", "high", "very_high")),
      head_neck_motion_importance = ordered(head_neck_motion_importance, levels = c("low", "moderate", "high", "very_high")),
      defensive_posture_specialisation = ordered(defensive_posture_specialisation, levels = c("low", "moderate", "high", "very_high")),
      postural_compaction = ordered(postural_compaction, levels = c("low", "moderate", "high", "very_high")),
      across(c(locomotor_mode, body_plan, activity_pattern, vertical_motion_scope), as.factor)
    )
}

make_gower_long <- function(mat, value_name) {
  as.data.frame(mat) %>%
    rownames_to_column("train_species") %>%
    pivot_longer(-train_species, names_to = "test_species", values_to = value_name)
}

get_trait_stat <- function(tbl, model_in, level_in, scope_in, version_in, var = c("rho", "p_value")) {
  var <- match.arg(var)
  out <- tbl %>%
    filter(model == model_in, level == level_in, scope == scope_in, distance_version == version_in) %>%
    pull(.data[[var]])
  if (length(out) == 0) return(NA_real_)
  out[[1]]
}

short_species <- function(x) {
  parts <- str_split(x, " ")
  vapply(parts, function(p) {
    if (length(p) >= 2) {
      paste0(substr(p[1], 1, 1), ". ", paste(p[-1], collapse = " "))
    } else {
      p[1]
    }
  }, character(1))
}

facet_spearman <- function(data, x_var, y_var, group_var = "model") {
  data %>%
    group_by(.data[[group_var]]) %>%
    group_modify(~ {
      x <- .x[[x_var]]
      y <- .x[[y_var]]
      sp <- safe_spearman(x, y)
      tibble(
        rho = sp$rho,
        p_value = sp$p_value,
        x_pos = max(x, na.rm = TRUE),
        y_pos = max(y, na.rm = TRUE)
      )
    }) %>%
    ungroup() %>%
    mutate(
      stat_label = paste0(
        "\u03C1 = ", sprintf("%.3f", round(rho, 3)), "\n",
        "p = ", ifelse(
          is.na(p_value),
          "NA",
          ifelse(p_value < 0.001, "< 0.001", sprintf("%.3f", round(p_value, 3)))
        )
      )
    )
}

stat_label_layer <- function(stat_df) {
  geom_label(
    data = stat_df,
    aes(x = x_pos, y = y_pos, label = stat_label),
    inherit.aes = FALSE,
    hjust = 1,
    vjust = 1,
    size = 3.2,
    fontface = "bold",
    label.size = NA,
    fill = alpha("white", 0.7),
    colour = "black",
    label.padding = unit(0.22, "lines"),
    label.r = unit(0, "lines")
  )
}

# ── Plot theme ───────────────────────────────────────────────────────────────

theme_h1_clean <- function() {
  theme_bw(base_size = 10, base_family = "sans") +
    theme(
      plot.title = element_blank(),
      plot.subtitle = element_blank(),
      plot.caption = element_blank(),
      axis.title.x = element_blank(),
      axis.title.y = element_text(size = 11, face = "bold", margin = margin(r = 8)),
      axis.text.x = element_text(size = 10, colour = "black"),
      axis.text.y = element_text(size = 10, colour = "black"),
      strip.background = element_rect(fill = "grey95", colour = "black", linewidth = 0.6),
      strip.text = element_text(face = "bold", size = 11, margin = margin(t = 4, b = 4)),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(colour = "grey85", linewidth = 0.3, linetype = "dashed"),
      panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.8),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      legend.position = "none",
      plot.margin = margin(10, 10, 10, 10)
    )
}

theme_h3_scatter <- function() {
  theme_h1_clean() +
    theme(
      axis.title.x = element_text(size = 11, face = "bold", margin = margin(t = 8)),
      legend.position = "right"
    )
}

save_plot <- function(plot_obj, filename, width = 180, height = 130) {
  ggsave(
    filename = file.path(plots_dir, filename),
    plot = plot_obj,
    width = width,
    height = height,
    units = "mm",
    dpi = 600,
    bg = "white"
  )
}

point_layer <- function(size = 2.4) {
  geom_point(
    shape = 21,
    size = size,
    fill = "black",
    color = "white",
    stroke = 0.35,
    alpha = 0.80
  )
}

smooth_layer <- function() {
  geom_smooth(
    method = "lm",
    se = TRUE,
    level = 0.95,
    linewidth = 0.7,
    color = "black",
    fill = "black",
    alpha = 0.40
  )
}

# =============================================================================
# INPUT CHECKS
# =============================================================================
must_exist(c(
  model_files$pairwise_file,
  model_files$within_file,
  manual_traits_core_file,
  manual_traits_extended_file
))

# ── Dataset → species mapping ────────────────────────────────────────────────
dataset_to_species <- c(
  "Bison" = "Bison bonasus",
  "Cattle" = "Bos taurus",
  "Dog" = "Canis lupus familiaris",
  "Fox_dataset_1" = "Vulpes vulpes",
  "Fox_dataset_2" = "Vulpes vulpes",
  "Giraffe" = "Giraffa camelopardalis",
  "Hedgehog" = "Erinaceus europaeus",
  "Horse_dataset_1" = "Equus ferus przewalskii",
  "Horse_dataset_2" = "Equus caballus",
  "Raccoon_dataset_1" = "Procyon lotor",
  "Raccoon_dataset_2" = "Procyon lotor"
)

# ── Traits ───────────────────────────────────────────────────────────────────
manual_traits_core_raw <- read_trait_file(manual_traits_core_file, required_core_trait_cols)
manual_traits_extended_raw <- read_trait_file(manual_traits_extended_file, required_extended_trait_cols)

manual_traits_core <- make_core_traits_ordered(manual_traits_core_raw)
manual_traits_extended <- make_extended_traits_ordered(manual_traits_extended_raw)

# =============================================================================
# READ INPUTS
# =============================================================================
pairwise <- pmap_dfr(model_files[, c("model", "pairwise_file")], read_pairwise_model) %>%
  mutate(
    train_species = recode(train_dataset, !!!dataset_to_species, .default = NA_character_),
    test_species = recode(test_dataset, !!!dataset_to_species, .default = NA_character_),
    same_species = train_species == test_species,
    species_pair_id = paste(train_species, test_species, sep = "__")
  )

within_ref <- pmap_dfr(model_files[, c("model", "within_file")], read_within_model)

# =============================================================================
# COVERAGE CHECKS
# =============================================================================
print_section("INPUT OVERVIEW")
input_overview <- pairwise %>%
  count(model, name = "pairwise_rows") %>%
  left_join(within_ref %>% count(model, name = "within_rows"), by = "model") %>%
  left_join(pairwise %>% distinct(model, train_dataset) %>% count(model, name = "n_train_datasets"), by = "model") %>%
  left_join(pairwise %>% distinct(model, test_dataset) %>% count(model, name = "n_test_datasets"), by = "model")
print(input_overview)

unmapped_train <- pairwise %>% filter(is.na(train_species)) %>% distinct(model, train_dataset)
unmapped_test  <- pairwise %>% filter(is.na(test_species)) %>% distinct(model, test_dataset)

study_species <- tibble(species = sort(unique(c(pairwise$train_species, pairwise$test_species)))) %>%
  filter(!is.na(species))

missing_traits_core <- study_species %>% anti_join(manual_traits_core, by = "species")
missing_traits_extended <- study_species %>% anti_join(manual_traits_extended, by = "species")

print_section("TRAIT COVERAGE CHECK")
cat("Study species:", nrow(study_species), "\n")
cat("Missing in core trait table:", nrow(missing_traits_core), "\n")
if (nrow(missing_traits_core) > 0) print(missing_traits_core)
cat("Missing in extended trait table:", nrow(missing_traits_extended), "\n")
if (nrow(missing_traits_extended) > 0) print(missing_traits_extended)

if (nrow(unmapped_train) > 0 || nrow(unmapped_test) > 0 || nrow(missing_traits_core) > 0 || nrow(missing_traits_extended) > 0) {
  stop("Stop: mapping or trait coverage is incomplete.")
}

# =============================================================================
# GOWER DISTANCES
# =============================================================================
gower_core <- daisy(manual_traits_core %>% select(-species), metric = "gower")
gower_core_mat <- as.matrix(gower_core)
rownames(gower_core_mat) <- manual_traits_core$species
colnames(gower_core_mat) <- manual_traits_core$species

gower_extended <- daisy(manual_traits_extended %>% select(-species), metric = "gower")
gower_extended_mat <- as.matrix(gower_extended)
rownames(gower_extended_mat) <- manual_traits_extended$species
colnames(gower_extended_mat) <- manual_traits_extended$species

gower_core_long <- make_gower_long(gower_core_mat, "trait_distance_core")
gower_extended_long <- make_gower_long(gower_extended_mat, "trait_distance_extended")

gower_compare <- gower_core_long %>%
  left_join(gower_extended_long, by = c("train_species", "test_species")) %>%
  mutate(delta_distance = trait_distance_extended - trait_distance_core)

# =============================================================================
# MERGE TABLES
# =============================================================================
analysis_dt <- pairwise %>%
  left_join(gower_core_long, by = c("train_species", "test_species")) %>%
  left_join(gower_extended_long, by = c("train_species", "test_species")) %>%
  mutate(model = factor(model, levels = model_order))

analysis_loss <- analysis_dt %>%
  left_join(within_ref, by = c("model", "test_dataset" = "analysis_dataset")) %>%
  mutate(
    transfer_loss_f1 = within_macro_f1 - macro_f1,
    transfer_retention_f1 = safe_ratio(macro_f1, within_macro_f1),
    relative_transfer_f1 = transfer_retention_f1,
    transfer_loss_accuracy = within_accuracy - accuracy,
    transfer_retention_accuracy = safe_ratio(accuracy, within_accuracy)
  )

# =============================================================================
# MISSING BASELINES
# =============================================================================
expected_within_keys <- analysis_dt %>%
  distinct(model, test_dataset) %>%
  arrange(model, test_dataset)

available_within_keys <- within_ref %>%
  distinct(model, analysis_dataset) %>%
  transmute(model, test_dataset = analysis_dataset) %>%
  arrange(model, test_dataset)

missing_within_keys <- expected_within_keys %>%
  anti_join(available_within_keys, by = c("model", "test_dataset")) %>%
  mutate(reason = "No valid within baseline: too few replicates / class imbalance") %>%
  arrange(model, test_dataset)

missing_within_rows <- analysis_loss %>%
  filter(is.na(within_macro_f1)) %>%
  count(model, test_dataset, sort = TRUE) %>%
  left_join(missing_within_keys, by = c("model", "test_dataset"))

# =============================================================================
# REPLICATION CHECKS
# =============================================================================
duplicate_pair_ids <- pairwise %>%
  count(model, pair_id, sort = TRUE) %>%
  filter(n > 1)

duplicate_dataset_pairs <- pairwise %>%
  count(model, train_dataset, test_dataset, sort = TRUE) %>%
  filter(n > 1)

species_pair_reuse <- analysis_dt %>%
  count(model, train_species, test_species, trait_distance_core, sort = TRUE) %>%
  filter(n > 1)

dataset_to_species_collapse <- tibble(
  dataset = names(dataset_to_species),
  species = unname(dataset_to_species)
) %>%
  count(species, sort = TRUE) %>%
  filter(n > 1)

species_pair_reuse_summary <- species_pair_reuse %>%
  summarise(
    reused_pairs = n(),
    max_rows_per_species_pair = max(n),
    mean_rows_per_species_pair = mean(n)
  )

# =============================================================================
# SUMMARIES
# =============================================================================
pairwise_summary_by_model <- analysis_dt %>%
  group_by(model, same_species) %>%
  summarise(
    n = n(),
    mean_accuracy = mean(accuracy, na.rm = TRUE),
    mean_macro_recall = mean(macro_recall, na.rm = TRUE),
    mean_macro_precision = mean(macro_precision, na.rm = TRUE),
    mean_macro_f1 = mean(macro_f1, na.rm = TRUE),
    sd_macro_f1 = sd(macro_f1, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(species_match = ifelse(same_species, "same_species", "different_species")) %>%
  select(model, species_match, n, mean_accuracy, mean_macro_recall, mean_macro_precision, mean_macro_f1, sd_macro_f1)

same_diff_tests <- analysis_dt %>%
  mutate(species_match = ifelse(same_species, "same_species", "different_species")) %>%
  group_by(model) %>%
  group_modify(~ safe_wilcox(.x, "macro_f1", "species_match")) %>%
  ungroup() %>%
  mutate(test = "wilcox_macro_f1_same_vs_different")

same_diff_tests_overall <- analysis_dt %>%
  mutate(species_match = ifelse(same_species, "same_species", "different_species")) %>%
  {
    bind_cols(
      tibble(model = "overall", test = "wilcox_macro_f1_same_vs_different"),
      safe_wilcox(., "macro_f1", "species_match")
    )
  }

# =============================================================================
# PRIMARY H3 DATA
# =============================================================================
plot_dt <- analysis_dt %>%
  filter(is.finite(trait_distance_core), is.finite(macro_f1))

plot_diff_dt <- plot_dt %>%
  filter(!same_species)

species_level_dt <- plot_dt %>%
  group_by(model, train_species, test_species, trait_distance_core, trait_distance_extended, same_species) %>%
  summarise(
    mean_macro_f1 = mean(macro_f1, na.rm = TRUE),
    sd_macro_f1 = sd(macro_f1, na.rm = TRUE),
    n_rows = n(),
    .groups = "drop"
  )

species_level_diff_dt <- species_level_dt %>%
  filter(!same_species)

cor_compare <- bind_rows(
  safe_spearman(plot_dt$trait_distance_core, plot_dt$macro_f1) %>%
    mutate(distance_version = "core", level = "row", model = "overall", scope = "all_pairs"),
  safe_spearman(plot_diff_dt$trait_distance_core, plot_diff_dt$macro_f1) %>%
    mutate(distance_version = "core", level = "row", model = "overall", scope = "different_species_only"),
  safe_spearman(species_level_dt$trait_distance_core, species_level_dt$mean_macro_f1) %>%
    mutate(distance_version = "core", level = "species_pair", model = "overall", scope = "all_pairs"),
  safe_spearman(species_level_diff_dt$trait_distance_core, species_level_diff_dt$mean_macro_f1) %>%
    mutate(distance_version = "core", level = "species_pair", model = "overall", scope = "different_species_only"),
  plot_dt %>%
    group_by(model) %>%
    group_modify(~ safe_spearman(.x$trait_distance_core, .x$macro_f1)) %>%
    mutate(distance_version = "core", level = "row", scope = "all_pairs"),
  plot_diff_dt %>%
    group_by(model) %>%
    group_modify(~ safe_spearman(.x$trait_distance_core, .x$macro_f1)) %>%
    mutate(distance_version = "core", level = "row", scope = "different_species_only"),
  species_level_dt %>%
    group_by(model) %>%
    group_modify(~ safe_spearman(.x$trait_distance_core, .x$mean_macro_f1)) %>%
    mutate(distance_version = "core", level = "species_pair", scope = "all_pairs"),
  species_level_diff_dt %>%
    group_by(model) %>%
    group_modify(~ safe_spearman(.x$trait_distance_core, .x$mean_macro_f1)) %>%
    mutate(distance_version = "core", level = "species_pair", scope = "different_species_only")
) %>%
  select(distance_version, level, scope, model, n, rho, p_value)

print_section("ROW VS SPECIES-PAIR CORRELATION")
print(cor_compare)

# =============================================================================
# SENSITIVITY: CORE VS EXTENDED
# =============================================================================
trait_sensitivity_row <- analysis_dt %>%
  filter(is.finite(trait_distance_core), is.finite(trait_distance_extended), is.finite(macro_f1))
trait_sensitivity_row_diff <- trait_sensitivity_row %>% filter(!same_species)
trait_sensitivity_species <- species_level_dt %>%
  filter(is.finite(trait_distance_core), is.finite(trait_distance_extended), is.finite(mean_macro_f1))
trait_sensitivity_species_diff <- trait_sensitivity_species %>% filter(!same_species)

trait_sensitivity_summary <- bind_rows(
  safe_spearman(trait_sensitivity_row$trait_distance_core, trait_sensitivity_row$macro_f1) %>%
    mutate(level = "row", scope = "all_pairs", distance_version = "core", model = "overall"),
  safe_spearman(trait_sensitivity_row$trait_distance_extended, trait_sensitivity_row$macro_f1) %>%
    mutate(level = "row", scope = "all_pairs", distance_version = "extended", model = "overall"),
  safe_spearman(trait_sensitivity_row_diff$trait_distance_core, trait_sensitivity_row_diff$macro_f1) %>%
    mutate(level = "row", scope = "different_species_only", distance_version = "core", model = "overall"),
  safe_spearman(trait_sensitivity_row_diff$trait_distance_extended, trait_sensitivity_row_diff$macro_f1) %>%
    mutate(level = "row", scope = "different_species_only", distance_version = "extended", model = "overall"),
  safe_spearman(trait_sensitivity_species$trait_distance_core, trait_sensitivity_species$mean_macro_f1) %>%
    mutate(level = "species_pair", scope = "all_pairs", distance_version = "core", model = "overall"),
  safe_spearman(trait_sensitivity_species$trait_distance_extended, trait_sensitivity_species$mean_macro_f1) %>%
    mutate(level = "species_pair", scope = "all_pairs", distance_version = "extended", model = "overall"),
  safe_spearman(trait_sensitivity_species_diff$trait_distance_core, trait_sensitivity_species_diff$mean_macro_f1) %>%
    mutate(level = "species_pair", scope = "different_species_only", distance_version = "core", model = "overall"),
  safe_spearman(trait_sensitivity_species_diff$trait_distance_extended, trait_sensitivity_species_diff$mean_macro_f1) %>%
    mutate(level = "species_pair", scope = "different_species_only", distance_version = "extended", model = "overall"),
  trait_sensitivity_row %>%
    group_by(model) %>%
    group_modify(~ safe_spearman(.x$trait_distance_core, .x$macro_f1)) %>%
    mutate(level = "row", scope = "all_pairs", distance_version = "core"),
  trait_sensitivity_row %>%
    group_by(model) %>%
    group_modify(~ safe_spearman(.x$trait_distance_extended, .x$macro_f1)) %>%
    mutate(level = "row", scope = "all_pairs", distance_version = "extended"),
  trait_sensitivity_row_diff %>%
    group_by(model) %>%
    group_modify(~ safe_spearman(.x$trait_distance_core, .x$macro_f1)) %>%
    mutate(level = "row", scope = "different_species_only", distance_version = "core"),
  trait_sensitivity_row_diff %>%
    group_by(model) %>%
    group_modify(~ safe_spearman(.x$trait_distance_extended, .x$macro_f1)) %>%
    mutate(level = "row", scope = "different_species_only", distance_version = "extended"),
  trait_sensitivity_species %>%
    group_by(model) %>%
    group_modify(~ safe_spearman(.x$trait_distance_core, .x$mean_macro_f1)) %>%
    mutate(level = "species_pair", scope = "all_pairs", distance_version = "core"),
  trait_sensitivity_species %>%
    group_by(model) %>%
    group_modify(~ safe_spearman(.x$trait_distance_extended, .x$mean_macro_f1)) %>%
    mutate(level = "species_pair", scope = "all_pairs", distance_version = "extended"),
  trait_sensitivity_species_diff %>%
    group_by(model) %>%
    group_modify(~ safe_spearman(.x$trait_distance_core, .x$mean_macro_f1)) %>%
    mutate(level = "species_pair", scope = "different_species_only", distance_version = "core"),
  trait_sensitivity_species_diff %>%
    group_by(model) %>%
    group_modify(~ safe_spearman(.x$trait_distance_extended, .x$mean_macro_f1)) %>%
    mutate(level = "species_pair", scope = "different_species_only", distance_version = "extended")
) %>%
  select(model, level, scope, distance_version, n, rho, p_value)

largest_distance_changes <- gower_compare %>%
  arrange(desc(abs(delta_distance))) %>%
  slice_head(n = 20)

print_section("TRAIT SENSITIVITY")
print(trait_sensitivity_summary)

# =============================================================================
# RETENTION AND LOSS
# =============================================================================
loss_dt <- analysis_loss %>%
  filter(
    is.finite(trait_distance_core),
    is.finite(trait_distance_extended),
    is.finite(macro_f1),
    is.finite(within_macro_f1),
    is.finite(transfer_loss_f1),
    is.finite(transfer_retention_f1)
  )

loss_diff_dt <- loss_dt %>% filter(!same_species)

retention_gt1 <- loss_dt %>%
  filter(transfer_retention_f1 > 1) %>%
  mutate(retention_excess = transfer_retention_f1 - 1) %>%
  arrange(desc(transfer_retention_f1))

retention_ge_095 <- loss_dt %>%
  filter(transfer_retention_f1 >= 0.95) %>%
  arrange(desc(transfer_retention_f1))

loss_cor_compare <- bind_rows(
  safe_spearman(loss_dt$trait_distance_core, loss_dt$transfer_loss_f1) %>%
    mutate(distance_version = "core", scope = "all_rows", model = "overall"),
  safe_spearman(loss_diff_dt$trait_distance_core, loss_diff_dt$transfer_loss_f1) %>%
    mutate(distance_version = "core", scope = "different_species_only", model = "overall"),
  safe_spearman(loss_dt$trait_distance_extended, loss_dt$transfer_loss_f1) %>%
    mutate(distance_version = "extended", scope = "all_rows", model = "overall"),
  safe_spearman(loss_diff_dt$trait_distance_extended, loss_diff_dt$transfer_loss_f1) %>%
    mutate(distance_version = "extended", scope = "different_species_only", model = "overall"),
  loss_dt %>%
    group_by(model) %>%
    group_modify(~ safe_spearman(.x$trait_distance_core, .x$transfer_loss_f1)) %>%
    mutate(distance_version = "core", scope = "all_rows"),
  loss_dt %>%
    group_by(model) %>%
    group_modify(~ safe_spearman(.x$trait_distance_extended, .x$transfer_loss_f1)) %>%
    mutate(distance_version = "extended", scope = "all_rows"),
  loss_diff_dt %>%
    group_by(model) %>%
    group_modify(~ safe_spearman(.x$trait_distance_core, .x$transfer_loss_f1)) %>%
    mutate(distance_version = "core", scope = "different_species_only"),
  loss_diff_dt %>%
    group_by(model) %>%
    group_modify(~ safe_spearman(.x$trait_distance_extended, .x$transfer_loss_f1)) %>%
    mutate(distance_version = "extended", scope = "different_species_only")
) %>%
  select(model, distance_version, scope, n, rho, p_value)

print_section("LOSS CORRELATION CHECK")
print(loss_cor_compare)

# =============================================================================
# MODEL SNAPSHOTS
# =============================================================================
model_snapshot <- bind_rows(
  safe_lm_distance(plot_dt, "macro_f1", "trait_distance_core") %>%
    mutate(model = "overall", response = "macro_f1", level = "row", distance_version = "core"),
  safe_lm_distance(species_level_dt, "mean_macro_f1", "trait_distance_core") %>%
    mutate(model = "overall", response = "mean_macro_f1", level = "species_pair", distance_version = "core"),
  safe_lm_distance(plot_dt, "macro_f1", "trait_distance_extended") %>%
    mutate(model = "overall", response = "macro_f1", level = "row", distance_version = "extended"),
  safe_lm_distance(species_level_dt, "mean_macro_f1", "trait_distance_extended") %>%
    mutate(model = "overall", response = "mean_macro_f1", level = "species_pair", distance_version = "extended"),
  plot_dt %>%
    group_by(model) %>%
    group_modify(~ safe_lm_distance(.x, "macro_f1", "trait_distance_core")) %>%
    mutate(response = "macro_f1", level = "row", distance_version = "core"),
  species_level_dt %>%
    group_by(model) %>%
    group_modify(~ safe_lm_distance(.x, "mean_macro_f1", "trait_distance_core")) %>%
    mutate(response = "mean_macro_f1", level = "species_pair", distance_version = "core"),
  plot_dt %>%
    group_by(model) %>%
    group_modify(~ safe_lm_distance(.x, "macro_f1", "trait_distance_extended")) %>%
    mutate(response = "macro_f1", level = "row", distance_version = "extended"),
  species_level_dt %>%
    group_by(model) %>%
    group_modify(~ safe_lm_distance(.x, "mean_macro_f1", "trait_distance_extended")) %>%
    mutate(response = "mean_macro_f1", level = "species_pair", distance_version = "extended")
) %>%
  select(model, response, level, distance_version, n, aic, r_squared, beta_distance, p_distance)

# =============================================================================
# QC FLAGS
# =============================================================================
core_rho_row_all <- get_trait_stat(trait_sensitivity_summary, "overall", "row", "all_pairs", "core", "rho")
extended_rho_row_all <- get_trait_stat(trait_sensitivity_summary, "overall", "row", "all_pairs", "extended", "rho")
core_rho_species_all <- get_trait_stat(trait_sensitivity_summary, "overall", "species_pair", "all_pairs", "core", "rho")
extended_rho_species_all <- get_trait_stat(trait_sensitivity_summary, "overall", "species_pair", "all_pairs", "extended", "rho")
core_p_species_all <- get_trait_stat(trait_sensitivity_summary, "overall", "species_pair", "all_pairs", "core", "p_value")
extended_p_species_all <- get_trait_stat(trait_sensitivity_summary, "overall", "species_pair", "all_pairs", "extended", "p_value")
coverage_rate <- 1 - (sum(is.na(analysis_loss$within_macro_f1)) / nrow(analysis_loss))

qc_flags <- tibble(
  flag = c(
    "within_baseline_coverage",
    "pseudoreplication_species_pairs",
    "multiple_datasets_per_species",
    "trait_sensitivity_row_level",
    "trait_sensitivity_species_pair_level",
    "retention_gt_1_present"
  ),
  severity = c(
    ifelse(nrow(missing_within_keys) > 0, "warning", "ok"),
    ifelse(nrow(species_pair_reuse) > 0, "warning", "ok"),
    ifelse(nrow(dataset_to_species_collapse) > 0, "warning", "ok"),
    ifelse(!is.na(core_rho_row_all) && !is.na(extended_rho_row_all) && abs(extended_rho_row_all - core_rho_row_all) >= 0.05, "warning", "ok"),
    ifelse(
      (!is.na(core_rho_species_all) && !is.na(extended_rho_species_all) && abs(extended_rho_species_all - core_rho_species_all) >= 0.045) ||
        (!is.na(core_p_species_all) && !is.na(extended_p_species_all) && core_p_species_all >= 0.05 && extended_p_species_all < 0.05),
      "warning",
      "ok"
    ),
    ifelse(nrow(retention_gt1) > 0, "warning", "ok")
  ),
  detail = c(
    paste0(nrow(missing_within_keys), " missing model-target baselines; coverage rate = ", fmt_num(coverage_rate, 3)),
    paste0(nrow(species_pair_reuse), " repeated species-pair combinations across rows"),
    paste0(nrow(dataset_to_species_collapse), " species mapped from multiple datasets"),
    paste0("row-level rho delta = ", fmt_num(extended_rho_row_all - core_rho_row_all, 4)),
    paste0(
      "species-pair rho delta = ", fmt_num(extended_rho_species_all - core_rho_species_all, 4),
      "; p shift ", fmt_p(core_p_species_all), " -> ", fmt_p(extended_p_species_all)
    ),
    paste0(nrow(retention_gt1), " rows with retention > 1")
  )
)

# =============================================================================
# REPORTING FOCUS
# =============================================================================
reporting_focus_tbl <- tibble(
  component = c(
    "preferred_primary_result",
    "preferred_inference_level",
    "sensitivity_result",
    "retention_ratio_interpretation"
  ),
  detail = c(
    paste0(
      "Core distance, species-pair level, different-species only: rho = ",
      fmt_num(get_trait_stat(trait_sensitivity_summary, "overall", "species_pair", "different_species_only", "core", "rho"), 3),
      ", p = ", fmt_p(get_trait_stat(trait_sensitivity_summary, "overall", "species_pair", "different_species_only", "core", "p_value")),
      ", n = ",
      trait_sensitivity_summary %>%
        filter(model == "overall", level == "species_pair", scope == "different_species_only", distance_version == "core") %>%
        pull(n) %>%
        .[1]
    ),
    "Use species-pair level as the main inference level; row-level results are supportive only.",
    paste0(
      "Extended sensitivity, species-pair level, different-species only: rho = ",
      fmt_num(get_trait_stat(trait_sensitivity_summary, "overall", "species_pair", "different_species_only", "extended", "rho"), 3),
      ", p = ", fmt_p(get_trait_stat(trait_sensitivity_summary, "overall", "species_pair", "different_species_only", "extended", "p_value"))
    ),
    "transfer_retention_f1 is a ratio (macro-F1 / within macro-F1). Values > 1 indicate cross-species performance above the within baseline."
  )
)

metric_notes_tbl <- tibble(
  metric = c("transfer_retention_f1", "transfer_loss_f1"),
  preferred_label = c("relative transfer to within baseline", "absolute transfer loss to within baseline"),
  formula = c("macro_f1 / within_macro_f1", "within_macro_f1 - macro_f1"),
  interpretation = c(
    "Ratio metric. 1 = equal to within baseline; > 1 = better than within baseline; < 1 = worse than within baseline.",
    "Difference metric. 0 = equal to within baseline; positive = worse than within baseline; negative = better than within baseline."
  )
)

# =============================================================================
# PLOTS
# =============================================================================
print_section("GENERATING PLOTS")

species_levels <- rownames(gower_core_mat)

heat_core_dt <- gower_core_long %>%
  mutate(
    train_species = factor(train_species, levels = species_levels),
    test_species = factor(test_species, levels = species_levels)
  )

heat_extended_dt <- gower_extended_long %>%
  mutate(
    train_species = factor(train_species, levels = species_levels),
    test_species = factor(test_species, levels = species_levels)
  )

p_heat_core <- ggplot(heat_core_dt, aes(x = test_species, y = train_species, fill = trait_distance_core)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.2f", trait_distance_core)), size = 2.6, color = "black") +
  scale_fill_distiller(palette = "YlOrRd", direction = 1, limits = c(0, 1)) +
  scale_x_discrete(labels = short_species) +
  scale_y_discrete(labels = short_species) +
  labs(y = "Train species", fill = "Gower") +
  theme_h1_clean() +
  theme(
    axis.title.x = element_text(size = 11, face = "bold", margin = margin(t = 8)),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    legend.position = "right",
    legend.key.height = unit(1.2, "cm")
  )

p_heat_extended <- ggplot(heat_extended_dt, aes(x = test_species, y = train_species, fill = trait_distance_extended)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.2f", trait_distance_extended)), size = 2.6, color = "black") +
  scale_fill_distiller(palette = "YlOrRd", direction = 1, limits = c(0, 1)) +
  scale_x_discrete(labels = short_species) +
  scale_y_discrete(labels = short_species) +
  labs(y = "Train species", fill = "Gower") +
  theme_h1_clean() +
  theme(
    axis.title.x = element_text(size = 11, face = "bold", margin = margin(t = 8)),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    legend.position = "right",
    legend.key.height = unit(1.2, "cm")
  )

save_plot(p_heat_core, "01_heatmap_gower_core.png", width = 190, height = 150)
save_plot(p_heat_extended, "02_heatmap_gower_extended.png", width = 190, height = 150)

fig_dt <- species_level_dt %>%
  filter(!same_species, is.finite(trait_distance_core), is.finite(mean_macro_f1)) %>%
  mutate(model = factor(model, levels = model_order))

fmt_p_paper <- function(x) {
  ifelse(is.na(x), "    NA", ifelse(x < 0.001, "< 0.001", sprintf("%7.3f", round(x, 3))))
}

fig_stats <- fig_dt %>%
  group_by(model) %>%
  group_modify(~ safe_spearman(.x$trait_distance_core, .x$mean_macro_f1)) %>%
  ungroup() %>%
  mutate(stat_label = paste0("\u03C1 = ", sprintf("%.3f", round(rho, 3)), "\n", "p = ", fmt_p_paper(p_value)))

x_data_rng <- range(fig_dt$trait_distance_core, na.rm = TRUE)
y_data_rng <- range(fig_dt$mean_macro_f1, na.rm = TRUE)

x_pad <- max(0.02, diff(x_data_rng) * 0.06)
y_pad <- max(0.03, diff(y_data_rng) * 0.08)

x_lim <- c(max(0, x_data_rng[1] - x_pad), x_data_rng[2] + x_pad)
y_lim <- c(max(0, y_data_rng[1] - y_pad), min(1.03, y_data_rng[2] + y_pad))

x_label <- x_lim[2] - 0.03 * diff(x_lim)
y_label <- y_lim[2] - 0.03 * diff(y_lim)

make_model_plot <- function(model_name, show_x = TRUE, show_y = TRUE) {
  dt_sub <- fig_dt %>% filter(model == model_name)
  stat_sub <- fig_stats %>% filter(model == model_name)
  
  p <- ggplot(dt_sub, aes(x = trait_distance_core, y = mean_macro_f1)) +
    geom_smooth(
      method = "lm",
      se = TRUE,
      level = 0.95,
      linewidth = 0.7,
      color = "grey30",
      fill = "grey80",
      alpha = 0.40
    ) +
    geom_point(
      shape = 21,
      size = 2.4,
      fill = "black",
      color = "white",
      stroke = 0.35,
      alpha = 0.80
    ) +
    geom_label(
      data = stat_sub,
      aes(x = x_label, y = y_label, label = stat_label),
      inherit.aes = FALSE,
      hjust = 1,
      vjust = 1,
      size = 3.2,
      fontface = "bold",
      label.size = NA,
      fill = alpha("white", 0.7),
      colour = "black",
      label.padding = unit(0.22, "lines"),
      label.r = unit(0, "lines")
    ) +
    scale_x_continuous(
      limits = x_lim,
      breaks = seq(0, 1, 0.25),
      labels = function(x) sprintf("%.2f", x),
      expand = expansion(mult = 0)
    ) +
    scale_y_continuous(
      limits = y_lim,
      breaks = seq(0, 1, 0.25),
      labels = function(x) sprintf("%.2f", x),
      expand = expansion(mult = 0)
    ) +
    coord_cartesian(clip = "off") +
    labs(title = model_name) +
    theme_bw(base_size = 10, base_family = "sans") +
    theme(
      aspect.ratio = 1,
      plot.title = element_text(hjust = 0.5, face = "bold", size = 11),
      axis.title = element_blank(),
      axis.text = element_text(size = 10, color = "black"),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(colour = "grey85", linewidth = 0.3, linetype = "dashed"),
      panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.8),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      legend.position = "none",
      plot.margin = margin(4, 6, 4, 6)
    )
  
  if (!show_x) p <- p + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
  if (!show_y) p <- p + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
  p
}

p_cnn_2x3   <- make_model_plot("CNN",           show_x = FALSE, show_y = TRUE)
p_it_2x3    <- make_model_plot("InceptionTime", show_x = FALSE, show_y = FALSE)
p_hydra_2x3 <- make_model_plot("HYDRA",         show_x = FALSE, show_y = FALSE)
p_mr_2x3    <- make_model_plot("MultiRocket",   show_x = TRUE,  show_y = TRUE)
p_rf_2x3    <- make_model_plot("RF",            show_x = TRUE,  show_y = FALSE)
p_lgbm_2x3  <- make_model_plot("LGBM",          show_x = TRUE,  show_y = FALSE)

p_combined_2x3 <- (p_cnn_2x3 | p_it_2x3 | p_hydra_2x3) / (p_mr_2x3 | p_rf_2x3 | p_lgbm_2x3)
final_grob_2x3 <- patchworkGrob(p_combined_2x3)

ggsave(
  filename = file.path(plots_dir, "03_figure_01_combined_transfer_vs_core_trait_distance_2x3.png"),
  plot = gridExtra::arrangeGrob(
    final_grob_2x3,
    left = textGrob("Mean macro-F1", rot = 90, gp = gpar(fontsize = 11, fontface = "bold")),
    bottom = textGrob("Core trait distance (Gower)", gp = gpar(fontsize = 11, fontface = "bold")),
    padding = unit(0.4, "lines")
  ),
  width = 270,
  height = 180,
  units = "mm",
  dpi = 600,
  bg = "white"
)

p_cnn_3x2   <- make_model_plot("CNN",           show_x = FALSE, show_y = TRUE)
p_it_3x2    <- make_model_plot("InceptionTime", show_x = FALSE, show_y = FALSE)
p_hydra_3x2 <- make_model_plot("HYDRA",         show_x = FALSE, show_y = TRUE)
p_mr_3x2    <- make_model_plot("MultiRocket",   show_x = FALSE, show_y = FALSE)
p_rf_3x2    <- make_model_plot("RF",            show_x = TRUE,  show_y = TRUE)
p_lgbm_3x2  <- make_model_plot("LGBM",          show_x = TRUE,  show_y = FALSE)

p_combined_3x2 <- (p_cnn_3x2 | p_it_3x2) / (p_hydra_3x2 | p_mr_3x2) / (p_rf_3x2 | p_lgbm_3x2)
final_grob_3x2 <- patchworkGrob(p_combined_3x2)

ggsave(
  filename = file.path(plots_dir, "04_figure_01_combined_transfer_vs_core_trait_distance_3x2.png"),
  plot = gridExtra::arrangeGrob(
    final_grob_3x2,
    left = textGrob("Mean macro-F1", rot = 90, gp = gpar(fontsize = 11, fontface = "bold")),
    bottom = textGrob("Core trait distance (Gower)", gp = gpar(fontsize = 11, fontface = "bold")),
    padding = unit(0.4, "lines")
  ),
  width = 180,
  height = 270,
  units = "mm",
  dpi = 600,
  bg = "white"
)

plot_index <- tibble(
  file = c(
    "01_heatmap_gower_core.png",
    "02_heatmap_gower_extended.png",
    "03_figure_01_combined_transfer_vs_core_trait_distance_2x3.png",
    "04_figure_01_combined_transfer_vs_core_trait_distance_3x2.png"
  ),
  description = c(
    "Core trait Gower heatmap",
    "Extended trait Gower heatmap",
    "Paper-ready scatter panel (2 rows x 3 cols) using core traits",
    "Paper-ready scatter panel (3 rows x 2 cols) using core traits"
  )
)

paper_plot_index <- plot_index %>%
  filter(str_detect(file, "figure_01"))

# =============================================================================
# EXPORT CSVs
# =============================================================================
write_csv(model_files,                 file.path(csv_dir, "00_model_file_registry.csv"))
write_csv(input_overview,              file.path(csv_dir, "01_input_overview_by_model.csv"))
write_csv(expected_within_keys,        file.path(csv_dir, "02_expected_within_keys.csv"))
write_csv(available_within_keys,       file.path(csv_dir, "03_available_within_keys.csv"))
write_csv(missing_within_keys,         file.path(csv_dir, "04_missing_within_keys_with_reason.csv"))
write_csv(missing_within_rows,         file.path(csv_dir, "05_missing_within_rows.csv"))
write_csv(pairwise_summary_by_model,   file.path(csv_dir, "06_pairwise_summary_by_model_and_match.csv"))
write_csv(bind_rows(same_diff_tests, same_diff_tests_overall), file.path(csv_dir, "07_same_vs_different_species_tests_macro_f1.csv"))
write_csv(duplicate_pair_ids,          file.path(csv_dir, "08_duplicate_pair_ids.csv"))
write_csv(duplicate_dataset_pairs,     file.path(csv_dir, "09_duplicate_dataset_pairs.csv"))
write_csv(species_pair_reuse,          file.path(csv_dir, "10_species_pair_reuse.csv"))
write_csv(species_pair_reuse_summary,  file.path(csv_dir, "11_species_pair_reuse_summary.csv"))
write_csv(cor_compare,                 file.path(csv_dir, "12_core_correlations.csv"))
write_csv(trait_sensitivity_summary,   file.path(csv_dir, "13_core_vs_extended_sensitivity.csv"))
write_csv(loss_cor_compare,            file.path(csv_dir, "14_loss_correlations.csv"))
write_csv(model_snapshot,              file.path(csv_dir, "15_model_snapshot.csv"))
write_csv(species_level_dt,            file.path(csv_dir, "16_species_pair_performance_summary.csv"))
write_csv(gower_compare,               file.path(csv_dir, "17_gower_core_vs_extended_all_pairs.csv"))
write_csv(dataset_to_species_collapse, file.path(csv_dir, "18_species_with_multiple_datasets.csv"))
write_csv(qc_flags,                    file.path(csv_dir, "19_qc_flags.csv"))
write_csv(manual_traits_core_raw,      file.path(csv_dir, "20_manual_traits_core.csv"))
write_csv(manual_traits_extended_raw,  file.path(csv_dir, "21_manual_traits_extended.csv"))
write_csv(plot_index,                  file.path(csv_dir, "22_plot_index.csv"))
write_csv(paper_plot_index,            file.path(csv_dir, "23_plot_index_paper.csv"))
write_csv(retention_gt1,               file.path(csv_dir, "24_retention_gt1.csv"))
write_csv(retention_ge_095,            file.path(csv_dir, "25_retention_ge_095.csv"))
write_csv(largest_distance_changes,    file.path(csv_dir, "26_trait_sensitivity_top20.csv"))
write_csv(reporting_focus_tbl,         file.path(csv_dir, "27_reporting_focus.csv"))
write_csv(metric_notes_tbl,            file.path(csv_dir, "28_metric_notes.csv"))

report_tbl <- tibble(
  check_name = c(
    "pairwise_rows", "within_rows", "unmapped_train_labels", "unmapped_test_labels",
    "missing_core_traits", "missing_extended_traits", "missing_within_rows", "missing_within_keys",
    "duplicate_pair_ids", "duplicate_dataset_pairs", "reused_species_pairs",
    "species_with_multiple_datasets", "retention_gt_1", "retention_ge_0_95",
    "plot_files_written", "paper_plot_files_written", "reporting_focus_rows", "metric_note_rows"
  ),
  value = c(
    nrow(pairwise),
    nrow(within_ref),
    nrow(unmapped_train),
    nrow(unmapped_test),
    nrow(missing_traits_core),
    nrow(missing_traits_extended),
    sum(is.na(analysis_loss$within_macro_f1)),
    nrow(missing_within_keys),
    nrow(duplicate_pair_ids),
    nrow(duplicate_dataset_pairs),
    nrow(species_pair_reuse),
    nrow(dataset_to_species_collapse),
    nrow(retention_gt1),
    nrow(retention_ge_095),
    nrow(plot_index),
    nrow(paper_plot_index),
    nrow(reporting_focus_tbl),
    nrow(metric_notes_tbl)
  )
)
write_csv(report_tbl, file.path(csv_dir, "29_qc_summary.csv"))

summary_lines <- c(
  "# ===================================================",
  "# Main Analysis Summary - H3_traits_revised.R",
  "# ===================================================",
  "",
  paste0("Date: ", Sys.time()),
  paste0("Models: ", paste(model_files$model, collapse = ", ")),
  paste0("Pairwise rows: ", nrow(pairwise)),
  paste0("Within rows: ", nrow(within_ref)),
  "",
  "# ----- Scope -----",
  "Primary distance = core biomechanical trait distance",
  "Sensitivity distance = extended trait distance",
  paste0(
    "Primary result (overall, species-pair, diff-species): rho = ",
    fmt_num(get_trait_stat(trait_sensitivity_summary, "overall", "species_pair", "different_species_only", "core", "rho"), 3),
    ", p = ", fmt_p(get_trait_stat(trait_sensitivity_summary, "overall", "species_pair", "different_species_only", "core", "p_value"))
  ),
  paste0(
    "Sensitivity result (overall, species-pair, diff-species): rho = ",
    fmt_num(get_trait_stat(trait_sensitivity_summary, "overall", "species_pair", "different_species_only", "extended", "rho"), 3),
    ", p = ", fmt_p(get_trait_stat(trait_sensitivity_summary, "overall", "species_pair", "different_species_only", "extended", "p_value"))
  )
)

report_lines <- c(
  paste0("H3 revised output folder: ", out_dir),
  paste0("Created on: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "",
  "This report contains the revised H3 analysis with core traits as primary and extended traits as sensitivity.",
  "Saved H3 plots: core and extended heatmaps plus the two paper-ready scatter panels.",
  ""
)

report_lines <- append_section(report_lines, "INPUT OVERVIEW", input_overview)
report_lines <- append_section(report_lines, "PAIRWISE SUMMARY BY MODEL", pairwise_summary_by_model)
report_lines <- append_section(report_lines, "CORE CORRELATIONS", cor_compare)
report_lines <- append_section(report_lines, "LOSS CORRELATIONS", loss_cor_compare)
report_lines <- append_section(report_lines, "CORE VS EXTENDED SENSITIVITY", trait_sensitivity_summary)
report_lines <- append_section(report_lines, "QC FLAGS", qc_flags)
report_lines <- append_section(report_lines, "QC SUMMARY", report_tbl)
report_lines <- append_section(report_lines, "PLOT INDEX", plot_index)

write_txt_report(file.path(txt_dir, "01_h3_analysis_report_revised.txt"), report_lines)
write_txt_report(file.path(txt_dir, "02_h3_summary_revised.txt"), summary_lines)

print_section("DONE")
cat("Main output directory:\n", out_dir, "\n", sep = "")
cat("CSV files written to:\n", csv_dir, "\n", sep = "")
cat("PNG plots written to:\n", plots_dir, "\n", sep = "")
cat("TXT files written to:\n", txt_dir, "\n", sep = "")
