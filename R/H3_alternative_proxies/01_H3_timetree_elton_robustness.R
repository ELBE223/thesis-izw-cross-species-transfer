# =============================================================================
# H3 sensitivity add-on: TimeTree and EltonTraits robustness checks
# =============================================================================
# Author : Lucas Beseler
# Date   : 2026-05-17
#
# Purpose:
# - Check whether the TimeTree and EltonTraits sensitivity results are driven
#   by individual species or by aggregation level.
# - This script uses the outputs from H3_timetree_elton_sensitivity_v2.R.
#
# Analyses:
# 1) Leave-one-species-out (LOSO) for:
#    - EltonTraits minimal Gower distance
#    - EltonTraits extended Gower distance
#    - TimeTree divergence time
# 2) Bootstrap 95% CI for Spearman rho
# 3) Aggregation-level check:
#    - raw rows
#    - dataset pairs
#    - species pairs
#
# Output:
# - Output_R/H3_timetree_elton_robustness/
#
# =============================================================================

# ── 1. Setup ──────────────────────────────────────────────────────────────────
if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman", repos = "https://cloud.r-project.org")
}

pacman::p_load(
  readr, dplyr, tidyr, tibble, purrr, stringr,
  ggplot2
)

# ── 2. User settings ──────────────────────────────────────────────────────────
set.seed(42)
n_boot <- 1000L

# ── 3. Paths ──────────────────────────────────────────────────────────────────
base_dir      <- "/Volumes/Z Slim/11_05_2026_Data_Analysis"
output_r_root <- file.path(base_dir, "Output_R")

source_dir <- file.path(output_r_root, "H3_timetree_elton_sensitivity")
source_csv <- file.path(source_dir, "csv")

out_dir  <- file.path(output_r_root, "H3_timetree_elton_robustness")
csv_dir  <- file.path(out_dir, "csv")
plot_dir <- file.path(out_dir, "plots")
txt_dir  <- file.path(out_dir, "txt")

row_file <- file.path(source_csv, "10_h3_pairwise_with_timetree_elton_distances.csv")
species_pair_file <- file.path(source_csv, "11_h3_species_pair_with_timetree_elton_distances.csv")

for (d in c(output_r_root, out_dir, csv_dir, plot_dir, txt_dir)) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

# ── 4. Constants ──────────────────────────────────────────────────────────────
model_order <- c("CNN", "ResNet", "HYDRA", "MultiRocket", "RF", "LGBM")

distance_specs <- tribble(
  ~distance_version,          ~distance_col,              ~x_label,
  "elton_minimal_gower",      "elton_minimal_gower",      "EltonTraits minimal distance (Gower)",
  "elton_extended_gower",     "elton_extended_gower",     "EltonTraits extended distance (Gower)",
  "timetree_divergence_mya",  "timetree_divergence_mya",  "TimeTree divergence time (MYA)"
)

# ── 5. Helpers ────────────────────────────────────────────────────────────────
must_exist <- function(x) {
  missing <- x[!file.exists(x)]
  if (length(missing) > 0) {
    stop("Missing files:\n", paste(missing, collapse = "\n"))
  }
}

fmt_num <- function(x, digits = 3) {
  ifelse(is.na(x), "NA", formatC(x, format = "f", digits = digits))
}

fmt_p <- function(x) {
  ifelse(is.na(x), "NA", ifelse(x < 0.001, "< 0.001", sprintf("%.3f", round(x, 3))))
}

short_species <- function(x) {
  parts <- str_split(as.character(x), " ")
  vapply(parts, function(p) {
    if (length(p) >= 2) {
      paste0(substr(p[1], 1, 1), ". ", paste(p[-1], collapse = " "))
    } else {
      p[1]
    }
  }, character(1))
}

safe_spearman <- function(x, y, conf_level = 0.95) {
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]
  y <- y[ok]
  n_ok <- length(x)
  
  if (n_ok < 3) {
    return(tibble(
      n = n_ok,
      rho = NA_real_,
      p_value = NA_real_,
      ci_low = NA_real_,
      ci_high = NA_real_
    ))
  }
  
  tmp <- suppressWarnings(cor.test(x, y, method = "spearman", exact = FALSE))
  rho <- unname(tmp$estimate)
  
  if (n_ok > 3 && is.finite(rho) && abs(rho) < 1) {
    z <- atanh(pmin(pmax(rho, -0.999999), 0.999999))
    se_z <- 1 / sqrt(n_ok - 3)
    z_crit <- qnorm((1 + conf_level) / 2)
    ci_low <- tanh(z - z_crit * se_z)
    ci_high <- tanh(z + z_crit * se_z)
  } else {
    ci_low <- NA_real_
    ci_high <- NA_real_
  }
  
  tibble(
    n = n_ok,
    rho = rho,
    p_value = tmp$p.value,
    ci_low = ci_low,
    ci_high = ci_high
  )
}

boot_spearman <- function(x, y, n_iter = n_boot) {
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]
  y <- y[ok]
  n <- length(x)
  
  if (n < 5) {
    return(tibble(
      n = n,
      rho_obs = NA_real_,
      ci_low = NA_real_,
      ci_high = NA_real_,
      n_boot = 0L
    ))
  }
  
  rho_obs <- suppressWarnings(cor(x, y, method = "spearman"))
  
  rho_boot <- replicate(n_iter, {
    idx <- sample.int(n, size = n, replace = TRUE)
    suppressWarnings(cor(x[idx], y[idx], method = "spearman"))
  })
  
  rho_boot <- rho_boot[is.finite(rho_boot)]
  ci <- quantile(rho_boot, probs = c(0.025, 0.975), na.rm = TRUE)
  
  tibble(
    n = n,
    rho_obs = rho_obs,
    ci_low = unname(ci[[1]]),
    ci_high = unname(ci[[2]]),
    n_boot = length(rho_boot)
  )
}

calc_spearman_by_model <- function(df, x_col, y_col, distance_version, level, scope) {
  overall <- safe_spearman(df[[x_col]], df[[y_col]]) %>%
    mutate(model = "overall", .before = 1)
  
  per_model <- df %>%
    group_by(model) %>%
    group_modify(~ safe_spearman(.x[[x_col]], .x[[y_col]])) %>%
    ungroup() %>%
    mutate(model = as.character(model))
  
  bind_rows(overall, per_model) %>%
    mutate(
      distance_version = distance_version,
      level = level,
      scope = scope,
      .before = 1
    )
}

theme_clean <- function(base_size = 10) {
  theme_bw(base_size = base_size, base_family = "sans") +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5, colour = "grey35"),
      axis.text = element_text(colour = "black"),
      strip.background = element_rect(fill = "grey95", colour = "black", linewidth = 0.5),
      strip.text = element_text(face = "bold"),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.7),
      legend.position = "none"
    )
}

save_plot <- function(plot_obj, file, width = 10, height = 6, dpi = 600) {
  ggsave(
    filename = file.path(plot_dir, file),
    plot = plot_obj,
    width = width,
    height = height,
    dpi = dpi,
    bg = "white"
  )
}

# ── 6. Load input tables ──────────────────────────────────────────────────────
must_exist(c(row_file, species_pair_file))

row_dt <- read_csv(row_file, show_col_types = FALSE)
species_pair_dt <- read_csv(species_pair_file, show_col_types = FALSE)

required_row_cols <- c(
  "model", "train_dataset", "test_dataset", "train_species", "test_species",
  "same_species", "macro_f1",
  distance_specs$distance_col
)

required_species_cols <- c(
  "model", "train_species", "test_species", "same_species", "mean_macro_f1",
  distance_specs$distance_col
)

missing_row_cols <- setdiff(required_row_cols, names(row_dt))
missing_species_cols <- setdiff(required_species_cols, names(species_pair_dt))

if (length(missing_row_cols) > 0) {
  stop("Row-level table missing columns: ", paste(missing_row_cols, collapse = ", "))
}
if (length(missing_species_cols) > 0) {
  stop("Species-pair table missing columns: ", paste(missing_species_cols, collapse = ", "))
}

row_dt <- row_dt %>%
  mutate(
    same_species = as.logical(same_species),
    model = factor(model, levels = model_order)
  )

species_pair_dt <- species_pair_dt %>%
  mutate(
    same_species = as.logical(same_species),
    model = factor(model, levels = model_order)
  )

row_diff_dt <- row_dt %>% filter(!same_species)
species_diff_dt <- species_pair_dt %>% filter(!same_species)

all_species <- species_diff_dt %>%
  select(train_species, test_species) %>%
  pivot_longer(everything(), values_to = "species") %>%
  distinct(species) %>%
  arrange(species) %>%
  pull(species)

# ── 7. Reference correlations ─────────────────────────────────────────────────
reference_correlations <- pmap_dfr(distance_specs, function(distance_version, distance_col, x_label) {
  bind_rows(
    calc_spearman_by_model(
      species_diff_dt,
      distance_col,
      "mean_macro_f1",
      distance_version,
      "species_pair",
      "different_species_only"
    ),
    calc_spearman_by_model(
      row_diff_dt,
      distance_col,
      "macro_f1",
      distance_version,
      "row",
      "different_species_only"
    )
  )
}) %>%
  select(distance_version, level, scope, model, n, rho, p_value, ci_low, ci_high)

reference_species <- reference_correlations %>%
  filter(level == "species_pair", scope == "different_species_only")

# ── 8. Leave-one-species-out ──────────────────────────────────────────────────
loso_results <- pmap_dfr(distance_specs, function(distance_version, distance_col, x_label) {
  map_dfr(all_species, function(sp) {
    species_diff_dt %>%
      filter(train_species != sp, test_species != sp) %>%
      calc_spearman_by_model(
        distance_col,
        "mean_macro_f1",
        distance_version,
        "species_pair",
        "different_species_only"
      ) %>%
      mutate(
        distance_col = distance_col,
        dropped_species = sp,
        .after = "distance_version"
      )
  })
}) %>%
  left_join(
    reference_species %>%
      select(distance_version, model, rho_full = rho, p_full = p_value, n_full = n),
    by = c("distance_version", "model")
  ) %>%
  mutate(
    delta_rho = rho - rho_full,
    abs_delta_rho = abs(delta_rho),
    sign_flip = is.finite(rho) & is.finite(rho_full) & sign(rho) != sign(rho_full),
    p_crosses_005 = is.finite(p_value) & is.finite(p_full) & ((p_value < 0.05) != (p_full < 0.05)),
    model = factor(model, levels = c("overall", model_order))
  ) %>%
  arrange(distance_version, model, desc(abs_delta_rho))

loso_summary <- loso_results %>%
  group_by(distance_version, model) %>%
  summarise(
    rho_full = first(rho_full),
    p_full = first(p_full),
    n_full = first(n_full),
    n_loso_runs = n(),
    min_rho = min(rho, na.rm = TRUE),
    max_rho = max(rho, na.rm = TRUE),
    max_abs_delta_rho = max(abs_delta_rho, na.rm = TRUE),
    n_sign_flips = sum(sign_flip, na.rm = TRUE),
    n_p_crosses_005 = sum(p_crosses_005, na.rm = TRUE),
    most_influential_species = dropped_species[which.max(abs_delta_rho)],
    rho_without_most_influential = rho[which.max(abs_delta_rho)],
    .groups = "drop"
  ) %>%
  mutate(model = factor(model, levels = c("overall", model_order))) %>%
  arrange(distance_version, model)

# ── 9. Bootstrap CIs ──────────────────────────────────────────────────────────
bootstrap_results <- pmap_dfr(distance_specs, function(distance_version, distance_col, x_label) {
  overall <- boot_spearman(species_diff_dt[[distance_col]], species_diff_dt$mean_macro_f1) %>%
    mutate(model = "overall", .before = 1)
  
  per_model <- species_diff_dt %>%
    group_by(model) %>%
    group_modify(~ boot_spearman(.x[[distance_col]], .x$mean_macro_f1)) %>%
    ungroup() %>%
    mutate(model = as.character(model))
  
  bind_rows(overall, per_model) %>%
    mutate(
      distance_version = distance_version,
      level = "species_pair",
      scope = "different_species_only",
      model = factor(model, levels = c("overall", model_order)),
      ci_crosses_zero = ci_low <= 0 & ci_high >= 0,
      .before = 1
    )
}) %>%
  arrange(distance_version, model)

# ── 10. Aggregation-level check ───────────────────────────────────────────────
aggregation_results <- pmap_dfr(distance_specs, function(distance_version, distance_col, x_label) {
  row_level <- row_diff_dt %>%
    mutate(distance_value = .data[[distance_col]]) %>%
    filter(is.finite(distance_value), is.finite(macro_f1))
  
  dataset_level <- row_diff_dt %>%
    mutate(distance_value = .data[[distance_col]]) %>%
    filter(is.finite(distance_value), is.finite(macro_f1)) %>%
    group_by(model, train_dataset, test_dataset, distance_value) %>%
    summarise(mean_macro_f1 = mean(macro_f1, na.rm = TRUE), .groups = "drop")
  
  species_level <- species_diff_dt %>%
    mutate(distance_value = .data[[distance_col]]) %>%
    filter(is.finite(distance_value), is.finite(mean_macro_f1))
  
  bind_rows(
    calc_spearman_by_model(
      row_level,
      "distance_value",
      "macro_f1",
      distance_version,
      "row",
      "different_species_only"
    ),
    calc_spearman_by_model(
      dataset_level,
      "distance_value",
      "mean_macro_f1",
      distance_version,
      "dataset_pair",
      "different_species_only"
    ),
    calc_spearman_by_model(
      species_level,
      "distance_value",
      "mean_macro_f1",
      distance_version,
      "species_pair",
      "different_species_only"
    )
  )
}) %>%
  mutate(model = factor(model, levels = c("overall", model_order))) %>%
  arrange(distance_version, level, model)

# ── 11. Plots ─────────────────────────────────────────────────────────────────
plot_index <- tibble(file = character(), description = character())

loso_overall_plot <- loso_results %>%
  filter(model == "overall") %>%
  mutate(
    species_short = short_species(dropped_species),
    distance_version = factor(distance_version, levels = distance_specs$distance_version)
  )

ref_overall_plot <- reference_species %>%
  filter(model == "overall") %>%
  mutate(distance_version = factor(distance_version, levels = distance_specs$distance_version))

p_loso_overall <- ggplot(loso_overall_plot, aes(x = reorder(species_short, rho), y = rho)) +
  geom_hline(
    data = ref_overall_plot,
    aes(yintercept = rho),
    linetype = "dashed",
    colour = "grey45",
    linewidth = 0.5
  ) +
  geom_point(shape = 21, size = 3.2, fill = "grey40", colour = "black", stroke = 0.3) +
  facet_wrap(~ distance_version, scales = "free_y", ncol = 1) +
  coord_flip() +
  labs(
    title = "Leave-one-species-out robustness",
    subtitle = "Overall Spearman rho, species-pair level, different species only",
    x = NULL,
    y = "Spearman rho"
  ) +
  theme_clean()

save_plot(p_loso_overall, "01_loso_overall_all_distances.png", width = 8.5, height = 9)

plot_index <- bind_rows(
  plot_index,
  tibble(
    file = "01_loso_overall_all_distances.png",
    description = "Overall LOSO rho for all three distance measures."
  )
)

p_loso_timetree <- loso_results %>%
  filter(distance_version == "timetree_divergence_mya") %>%
  mutate(species_short = short_species(dropped_species)) %>%
  ggplot(aes(x = reorder(species_short, rho), y = rho)) +
  geom_hline(
    data = reference_species %>% filter(distance_version == "timetree_divergence_mya"),
    aes(yintercept = rho),
    linetype = "dashed",
    colour = "grey45",
    linewidth = 0.5
  ) +
  geom_point(shape = 21, size = 3, fill = "grey40", colour = "black", stroke = 0.3) +
  facet_wrap(~ model, scales = "free_y", ncol = 3) +
  coord_flip() +
  labs(
    title = "Leave-one-species-out robustness for TimeTree",
    subtitle = "Spearman rho by model, species-pair level, different species only",
    x = NULL,
    y = "Spearman rho"
  ) +
  theme_clean() +
  theme(axis.text.y = element_text(size = 8))

save_plot(p_loso_timetree, "02_loso_timetree_by_model.png", width = 12, height = 8)

plot_index <- bind_rows(
  plot_index,
  tibble(
    file = "02_loso_timetree_by_model.png",
    description = "LOSO rho for TimeTree divergence time by model."
  )
)

bootstrap_plot_dt <- bootstrap_results %>%
  mutate(
    label = factor(as.character(model), levels = rev(c("overall", model_order))),
    distance_version = factor(distance_version, levels = distance_specs$distance_version)
  )

p_boot <- ggplot(bootstrap_plot_dt, aes(x = rho_obs, y = label)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey55") +
  geom_segment(aes(x = ci_low, xend = ci_high, y = label, yend = label), linewidth = 0.6, colour = "grey35") +
  geom_point(shape = 21, size = 3.2, fill = "grey40", colour = "black", stroke = 0.3) +
  facet_wrap(~ distance_version, ncol = 1) +
  labs(
    title = "Bootstrap 95% confidence intervals",
    subtitle = paste0(format(n_boot, big.mark = ","), " bootstrap resamples, species-pair level, different species only"),
    x = "Spearman rho",
    y = NULL
  ) +
  theme_clean()

save_plot(p_boot, "03_bootstrap_ci_all_distances.png", width = 8.5, height = 9)

plot_index <- bind_rows(
  plot_index,
  tibble(
    file = "03_bootstrap_ci_all_distances.png",
    description = "Bootstrap confidence intervals for all distance measures."
  )
)

aggregation_plot_dt <- aggregation_results %>%
  filter(model == "overall") %>%
  mutate(
    level = factor(level, levels = c("row", "dataset_pair", "species_pair")),
    level_label = recode(level, "row" = "raw rows", "dataset_pair" = "dataset pairs", "species_pair" = "species pairs"),
    distance_version = factor(distance_version, levels = distance_specs$distance_version)
  )

p_agg <- ggplot(aggregation_plot_dt, aes(x = level_label, y = rho)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey55") +
  geom_col(fill = "grey45", colour = "black", width = 0.65) +
  geom_text(aes(label = sprintf("%.3f", rho)), vjust = ifelse(aggregation_plot_dt$rho < 0, 1.3, -0.4), size = 3.2) +
  facet_wrap(~ distance_version, ncol = 1) +
  labs(
    title = "Aggregation-level check",
    subtitle = "Overall Spearman rho, different species only",
    x = NULL,
    y = "Spearman rho"
  ) +
  theme_clean()

save_plot(p_agg, "04_aggregation_level_check.png", width = 8.5, height = 8)

plot_index <- bind_rows(
  plot_index,
  tibble(
    file = "04_aggregation_level_check.png",
    description = "Overall rho across row, dataset-pair, and species-pair aggregation levels."
  )
)

# ── 12. Export CSV ────────────────────────────────────────────────────────────
write_csv(reference_correlations, file.path(csv_dir, "00_reference_correlations.csv"))
write_csv(loso_results, file.path(csv_dir, "01_loso_results.csv"))
write_csv(loso_summary, file.path(csv_dir, "02_loso_summary.csv"))
write_csv(bootstrap_results, file.path(csv_dir, "03_bootstrap_ci.csv"))
write_csv(aggregation_results, file.path(csv_dir, "04_aggregation_level_check.csv"))
write_csv(plot_index, file.path(csv_dir, "05_plot_index.csv"))

input_summary <- tibble(
  item = c(
    "base_dir",
    "source_dir",
    "source_csv",
    "row_file",
    "species_pair_file",
    "output_dir",
    "n_boot",
    "n_row_diff",
    "n_species_pair_diff",
    "n_species",
    "models"
  ),
  value = c(
    base_dir,
    source_dir,
    source_csv,
    row_file,
    species_pair_file,
    out_dir,
    as.character(n_boot),
    as.character(nrow(row_diff_dt)),
    as.character(nrow(species_diff_dt)),
    as.character(length(all_species)),
    paste(model_order, collapse = ", ")
  )
)

write_csv(input_summary, file.path(csv_dir, "06_input_summary.csv"))

# ── 13. Export TXT report ─────────────────────────────────────────────────────
primary_overall <- reference_species %>%
  filter(model == "overall") %>%
  select(distance_version, n, rho, p_value, ci_low, ci_high)

boot_overall <- bootstrap_results %>%
  filter(model == "overall") %>%
  select(distance_version, rho_obs, ci_low, ci_high, ci_crosses_zero)

loso_overall_summary <- loso_summary %>%
  filter(model == "overall") %>%
  select(
    distance_version,
    rho_full,
    p_full,
    min_rho,
    max_rho,
    max_abs_delta_rho,
    n_sign_flips,
    n_p_crosses_005,
    most_influential_species,
    rho_without_most_influential
  )

summary_lines <- c(
  "# ===================================================",
  "# H3 TimeTree and EltonTraits robustness summary",
  "# ===================================================",
  "",
  paste0("Date: ", Sys.time()),
  paste0("Source folder: ", source_dir),
  paste0("Output folder: ", out_dir),
  paste0("Bootstrap samples: ", format(n_boot, big.mark = ",")),
  "",
  "# ----- Reference correlations -----",
  "Species-pair level, different species only:"
)

for (i in seq_len(nrow(primary_overall))) {
  r <- primary_overall[i, ]
  summary_lines <- c(
    summary_lines,
    paste0(
      "- ", r$distance_version,
      ": rho = ", fmt_num(r$rho),
      ", p = ", fmt_p(r$p_value),
      ", n = ", r$n,
      ", approx. CI = [", fmt_num(r$ci_low), ", ", fmt_num(r$ci_high), "]"
    )
  )
}

summary_lines <- c(summary_lines, "", "# ----- Leave-one-species-out verdict -----")

for (i in seq_len(nrow(loso_overall_summary))) {
  r <- loso_overall_summary[i, ]
  summary_lines <- c(
    summary_lines,
    paste0(
      "- ", r$distance_version,
      ": rho range = [", fmt_num(r$min_rho), ", ", fmt_num(r$max_rho), "]",
      "; max |delta rho| = ", fmt_num(r$max_abs_delta_rho, 4),
      "; sign flips = ", r$n_sign_flips,
      "; strongest influence = ", r$most_influential_species,
      " (rho without species = ", fmt_num(r$rho_without_most_influential), ")"
    )
  )
}

summary_lines <- c(summary_lines, "", "# ----- Bootstrap verdict -----")

for (i in seq_len(nrow(boot_overall))) {
  r <- boot_overall[i, ]
  summary_lines <- c(
    summary_lines,
    paste0(
      "- ", r$distance_version,
      ": rho = ", fmt_num(r$rho_obs),
      ", bootstrap 95% CI = [", fmt_num(r$ci_low), ", ", fmt_num(r$ci_high), "]",
      ifelse(r$ci_crosses_zero, " (crosses zero)", " (does not cross zero)")
    )
  )
}

agg_overall <- aggregation_results %>%
  filter(model == "overall") %>%
  arrange(distance_version, level)

summary_lines <- c(summary_lines, "", "# ----- Aggregation-level check -----")

for (dist in distance_specs$distance_version) {
  tmp <- agg_overall %>% filter(distance_version == dist)
  summary_lines <- c(
    summary_lines,
    paste0("- ", dist, ":"),
    paste0(
      "  ", tmp$level,
      ": rho = ", fmt_num(tmp$rho),
      ", p = ", fmt_p(tmp$p_value),
      ", n = ", tmp$n
    )
  )
}

summary_lines <- c(
  summary_lines,
  "",
  "# ----- Interpretation guide -----",
  "TimeTree is robust if rho remains negative after dropping each species and the bootstrap CI does not cross zero.",
  "EltonTraits distances are exploratory sensitivity checks. Weak or unstable Elton results do not weaken the main expert-coded functional-biomechanical H3 analysis.",
  "Report the species-pair/different-species level as the main sensitivity result."
)

writeLines(summary_lines, con = file.path(txt_dir, "01_h3_timetree_elton_robustness_summary.txt"))

files_lines <- c(
  paste0("Output folder: ", out_dir),
  paste0("CSV folder: ", csv_dir),
  paste0("Plots folder: ", plot_dir),
  paste0("TXT folder: ", txt_dir),
  "",
  "Written CSV files:",
  "- 00_reference_correlations.csv",
  "- 01_loso_results.csv",
  "- 02_loso_summary.csv",
  "- 03_bootstrap_ci.csv",
  "- 04_aggregation_level_check.csv",
  "- 05_plot_index.csv",
  "- 06_input_summary.csv",
  "",
  "Written TXT files:",
  "- 01_h3_timetree_elton_robustness_summary.txt",
  "- 02_h3_timetree_elton_robustness_files_written.txt",
  "",
  "Written plot files:",
  paste0("- ", plot_index$file)
)

writeLines(files_lines, con = file.path(txt_dir, "02_h3_timetree_elton_robustness_files_written.txt"))

# ── 14. Console output ────────────────────────────────────────────────────────
cat("\nH3 TimeTree + EltonTraits robustness checks complete.\n")
cat("Output folder:\n", out_dir, "\n\n")
cat("Reference correlations, species-pair level, different species only:\n")
print(primary_overall, n = Inf)
cat("\nLOSO summary, overall:\n")
print(loso_overall_summary, n = Inf)
cat("\nBootstrap summary, overall:\n")
print(boot_overall, n = Inf)
