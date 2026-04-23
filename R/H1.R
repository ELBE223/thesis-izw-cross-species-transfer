# =============================================================================
# H1 Analysis
# H1 (transfer gap): Behaviour classification performance will be higher in
# within-dataset evaluations than in cross-dataset transfer evaluations.
# =============================================================================

# ── 1. Setup ──────────────────────────────────────────────────────────────────
if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman", repos = "https://cloud.r-project.org")
}

pacman::p_load(readr, dplyr, tidyr, ggplot2, purrr, tibble, grid)


# ── 2. User settings ──────────────────────────────────────────────────────────
show_p_values    <- FALSE
p_value_position <- "bottom_right"   # "top_left" or "bottom_right"


# ── 3. Paths ──────────────────────────────────────────────────────────────────
base_dir <- "/Volumes/Z Slim/07_04_2026_Data_Analysis/Output"
out_dir  <- file.path(base_dir, "H1")

csv_dir   <- file.path(out_dir, "csv")
plots_dir <- file.path(out_dir, "plots")
txt_dir   <- file.path(out_dir, "txt")

for (d in c(out_dir, csv_dir, plots_dir, txt_dir)) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}


# ── 4. Constants ──────────────────────────────────────────────────────────────
model_levels      <- c("CNN", "ResNet", "HYDRA", "MultiRocket", "RF", "LGBM")
comparison_levels <- c("cross", "within")
plot_styles       <- c("bw", "color")


# ── 5. Input file registry ────────────────────────────────────────────────────
# For each (model, comparison) we load the per-dataset metrics from the
# standard file `metrics_all.csv`.
metric_paths <- tribble(
  ~model,         ~comparison, ~file,
  "RF",          "within", file.path(base_dir, "RF",          "Within_RF",          "statistics", "metrics_all.csv"),
  "RF",          "cross",  file.path(base_dir, "RF",          "Cross_RF",           "statistics", "metrics_all.csv"),
  "CNN",         "within", file.path(base_dir, "CNN",         "Within_CNN",         "statistics", "metrics_all.csv"),
  "CNN",         "cross",  file.path(base_dir, "CNN",         "Cross_CNN",          "statistics", "metrics_all.csv"),
  "ResNet",      "within", file.path(base_dir, "ResNet",      "Within_ResNet",      "statistics", "metrics_all.csv"),
  "ResNet",      "cross",  file.path(base_dir, "ResNet",      "Cross_ResNet",       "statistics", "metrics_all.csv"),
  "HYDRA",       "within", file.path(base_dir, "HYDRA",       "Within_HYDRA",       "statistics", "metrics_all.csv"),
  "HYDRA",       "cross",  file.path(base_dir, "HYDRA",       "Cross_HYDRA",        "statistics", "metrics_all.csv"),
  "LGBM",        "within", file.path(base_dir, "LGBM",        "Within_LGBM",        "statistics", "metrics_all.csv"),
  "LGBM",        "cross",  file.path(base_dir, "LGBM",        "Cross_LGBM",         "statistics", "metrics_all.csv"),
  "MultiRocket", "within", file.path(base_dir, "MultiRocket", "Within_MultiRocket", "statistics", "metrics_all.csv"),
  "MultiRocket", "cross",  file.path(base_dir, "MultiRocket", "Cross_MultiRocket",  "statistics", "metrics_all.csv")
)


# ── 6. Helpers ────────────────────────────────────────────────────────────────

# ---- 6.1 Formatting ----------------------------------------------------------
# Small utilities for console section headers, ID normalisation, and the
# numeric formats used throughout the main tables.

print_section <- function(x) {
  cat("\n", strrep("=", 18), x, strrep("=", 18), "\n", sep = " ")
}

normalize_id <- function(x) {
  x <- as.character(x)
  x <- trimws(x)
  x <- tolower(x)
  x[x %in% c("", "na", "nan", "null")] <- NA_character_
  x
}

fmt_num <- function(x, digits = 3) {
  ifelse(is.na(x), "NA", formatC(x, format = "f", digits = digits))
}

fmt_p <- function(x) {
  ifelse(is.na(x), "NA", ifelse(x < 0.001, "< 0.001", sprintf("%.3f", round(x, 3))))
}

format_mean_sd <- function(mean_val, sd_val) {
  ifelse(is.na(mean_val), "NA", sprintf("%.3f ± %.3f", mean_val, sd_val))
}

format_median_iqr <- function(med, q1, q3) {
  ifelse(is.na(med), "NA", sprintf("%.3f [%.3f, %.3f]", med, q1, q3))
}


# ---- 6.2 Statistics ----------------------------------------------------------
# Paired Wilcoxon test, matched rank-biserial effect size, and descriptive
# summaries over the paired (within, cross) values per dataset.

signed_rank_biserial <- function(diffs) {
  diffs <- diffs[is.finite(diffs) & !is.na(diffs) & diffs != 0]
  
  if (length(diffs) == 0) {
    return(NA_real_)
  }
  
  ranks <- rank(abs(diffs), ties.method = "average")
  w_pos <- sum(ranks[diffs > 0])
  w_neg <- sum(ranks[diffs < 0])
  
  (w_pos - w_neg) / (w_pos + w_neg)
}

run_paired_wilcox <- function(df, value_col) {
  paired_df <- build_paired_values(df, value_col)
  
  total_cross <- df %>%
    filter(comparison == "cross", is.finite(.data[[value_col]])) %>%
    nrow()
  
  total_within <- df %>%
    filter(comparison == "within", is.finite(.data[[value_col]])) %>%
    nrow()
  
  if (nrow(paired_df) < 2) {
    return(tibble(
      total_cross   = total_cross,
      total_within  = total_within,
      paired_n      = nrow(paired_df),
      statistic     = NA_real_,
      p_value       = NA_real_,
      mean_gap      = NA_real_,
      median_gap    = NA_real_,
      rank_biserial = NA_real_
    ))
  }
  
  diffs <- paired_df$within - paired_df$cross
  
  tmp <- tryCatch(
    wilcox.test(paired_df$within, paired_df$cross, paired = TRUE, exact = FALSE),
    error = function(e) NULL
  )
  
  if (is.null(tmp)) {
    return(tibble(
      total_cross   = total_cross,
      total_within  = total_within,
      paired_n      = nrow(paired_df),
      statistic     = NA_real_,
      p_value       = NA_real_,
      mean_gap      = mean(diffs, na.rm = TRUE),
      median_gap    = median(diffs, na.rm = TRUE),
      rank_biserial = signed_rank_biserial(diffs)
    ))
  }
  
  tibble(
    total_cross   = total_cross,
    total_within  = total_within,
    paired_n      = nrow(paired_df),
    statistic     = unname(tmp$statistic),
    p_value       = tmp$p.value,
    mean_gap      = mean(diffs, na.rm = TRUE),
    median_gap    = median(diffs, na.rm = TRUE),
    rank_biserial = signed_rank_biserial(diffs)
  )
}

summarise_paired_metric <- function(df, value_col) {
  paired_df <- build_paired_values(df, value_col)
  
  if (nrow(paired_df) == 0) {
    return(tibble(
      paired_n      = 0,
      mean_cross    = NA_real_,
      sd_cross      = NA_real_,
      median_cross  = NA_real_,
      q1_cross      = NA_real_,
      q3_cross      = NA_real_,
      mean_within   = NA_real_,
      sd_within     = NA_real_,
      median_within = NA_real_,
      q1_within     = NA_real_,
      q3_within     = NA_real_
    ))
  }
  
  tibble(
    paired_n      = nrow(paired_df),
    mean_cross    = mean(paired_df$cross, na.rm = TRUE),
    sd_cross      = sd(paired_df$cross, na.rm = TRUE),
    median_cross  = median(paired_df$cross, na.rm = TRUE),
    q1_cross      = quantile(paired_df$cross, probs = 0.25, na.rm = TRUE, names = FALSE),
    q3_cross      = quantile(paired_df$cross, probs = 0.75, na.rm = TRUE, names = FALSE),
    mean_within   = mean(paired_df$within, na.rm = TRUE),
    sd_within     = sd(paired_df$within, na.rm = TRUE),
    median_within = median(paired_df$within, na.rm = TRUE),
    q1_within     = quantile(paired_df$within, probs = 0.25, na.rm = TRUE, names = FALSE),
    q3_within     = quantile(paired_df$within, probs = 0.75, na.rm = TRUE, names = FALSE)
  )
}


# ---- 6.3 Data loading and pairing -------------------------------------------
# Read one metrics CSV, pick the best available dataset-ID column, collapse to
# one value per (dataset, comparison), and assemble long/wide paired frames.

read_metric_file <- function(model, comparison, file) {
  if (!file.exists(file)) {
    message("Missing: ", file)
    return(NULL)
  }
  
  read_csv(file, show_col_types = FALSE) %>%
    mutate(
      model       = model,
      comparison  = comparison,
      source_file = file
    )
}

best_dataset_id <- function(df,
                            group_cols = c("model", "comparison"),
                            candidates = c("pair_id", "test_dataset", "analysis_dataset")) {
  present <- candidates[candidates %in% names(df)]
  
  if (length(present) == 0) {
    return(rep(NA_character_, nrow(df)))
  }
  
  out <- rep(NA_character_, nrow(df))
  
  if (!all(group_cols %in% names(df))) {
    group_cols <- intersect(group_cols, names(df))
  }
  
  if (length(group_cols) == 0) {
    group_index <- rep("all_rows", nrow(df))
  } else {
    group_index <- interaction(df[group_cols], drop = TRUE, lex.order = TRUE)
  }
  
  # Within each (model, comparison) group, pick the candidate column with
  # the most distinct non-NA values as the dataset identifier.
  for (g in unique(group_index)) {
    idx <- which(group_index == g)
    
    scores <- sapply(present, function(col) {
      x <- normalize_id(df[[col]][idx])
      dplyr::n_distinct(x, na.rm = TRUE)
    })
    
    best_col <- present[which.max(scores)]
    out[idx] <- as.character(df[[best_col]][idx])
  }
  
  out
}

collapse_to_dataset_level <- function(df, value_col) {
  df %>%
    filter(!is.na(dataset_key), !is.na(comparison), is.finite(.data[[value_col]])) %>%
    group_by(dataset_key, comparison) %>%
    summarise(
      dataset_id = dplyr::first(stats::na.omit(dataset_id)),
      value      = mean(.data[[value_col]], na.rm = TRUE),
      .groups    = "drop"
    )
}

build_paired_values <- function(df, value_col) {
  out <- collapse_to_dataset_level(df, value_col) %>%
    select(dataset_key, dataset_id, comparison, value) %>%
    pivot_wider(
      names_from   = comparison,
      values_from  = value,
      names_prefix = "value_"
    )
  
  if (!"value_cross" %in% names(out)) {
    out$value_cross <- NA_real_
  }
  
  if (!"value_within" %in% names(out)) {
    out$value_within <- NA_real_
  }
  
  out %>%
    filter(
      is.finite(.data$value_cross),
      is.finite(.data$value_within)
    ) %>%
    arrange(dataset_key) %>%
    rename(
      cross  = value_cross,
      within = value_within
    )
}

build_paired_long <- function(df, value_col) {
  paired_df <- build_paired_values(df, value_col)
  
  if (nrow(paired_df) == 0) {
    return(tibble())
  }
  
  paired_df %>%
    pivot_longer(
      cols      = c(cross, within),
      names_to  = "comparison",
      values_to = "value"
    ) %>%
    mutate(comparison = factor(comparison, levels = comparison_levels))
}

build_paired_wide_all <- function(df, value_col) {
  df %>%
    split(.$model) %>%
    imap_dfr(~ build_paired_values(.x, value_col) %>% mutate(model = .y)) %>%
    mutate(model = factor(model, levels = model_levels))
}

build_paired_long_all <- function(df, value_col) {
  df %>%
    split(.$model) %>%
    imap_dfr(~ build_paired_long(.x, value_col) %>% mutate(model = .y)) %>%
    mutate(model = factor(model, levels = model_levels))
}


# ---- 6.4 Plotting ------------------------------------------------------------
# Shared style/theme, p-value annotation placement, and the two main plot
# builders (paired boxplot with jittered points and paired slope plot).

get_plot_style <- function(plot_style = c("bw", "color")) {
  plot_style <- match.arg(plot_style)
  
  if (plot_style == "color") {
    return(list(
      fill_values = c("cross" = "#0072B2", "within" = "#D55E00"),
      point_alpha = 0.90,
      strip_fill  = "grey95",
      label_fill  = "white",
      line_colour = "grey55"
    ))
  }
  
  list(
    fill_values = c("cross" = "grey75", "within" = "white"),
    point_alpha = 0.95,
    strip_fill  = "grey95",
    label_fill  = "white",
    line_colour = "grey55"
  )
}

theme_clean <- function(plot_style = c("bw", "color")) {
  plot_style <- match.arg(plot_style)
  style <- get_plot_style(plot_style)
  
  theme_bw(base_size = 10, base_family = "sans") +
    theme(
      plot.title         = element_blank(),
      plot.subtitle      = element_blank(),
      plot.caption       = element_blank(),
      axis.title.x       = element_blank(),
      axis.title.y       = element_text(size = 10, face = "plain", margin = margin(r = 8)),
      axis.text.x        = element_text(size = 10, colour = "black"),
      axis.text.y        = element_text(size = 10, colour = "black"),
      strip.background   = element_rect(fill = style$strip_fill, colour = "black", linewidth = 0.6),
      strip.text         = element_text(face = "bold", size = 11, margin = margin(t = 4, b = 4)),
      panel.grid.minor   = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(colour = "grey85", linewidth = 0.3, linetype = "dashed"),
      panel.border       = element_rect(colour = "black", fill = NA, linewidth = 0.8),
      plot.background    = element_rect(fill = "white", colour = NA),
      panel.background   = element_rect(fill = "white", colour = NA),
      legend.position    = "none",
      plot.margin        = margin(10, 10, 10, 10)
    )
}

build_annotation_df <- function(stats_df, label_corner = c("top_left", "bottom_right")) {
  label_corner <- match.arg(label_corner)
  
  stats_df %>%
    mutate(
      x     = ifelse(label_corner == "top_left", 1.02, 2.45),
      y     = ifelse(label_corner == "top_left", 1.045, 0.035),
      hjust = ifelse(label_corner == "top_left", 0, 0.90),
      vjust = ifelse(label_corner == "top_left", 1, 0),
      label = paste0("p = ", fmt_p(p_value))
    )
}

build_h1_boxplot <- function(plot_df, stats_df, y_lab,
                             plot_style    = c("bw", "color"),
                             label_corner  = c("top_left", "bottom_right"),
                             show_p_values = TRUE) {
  plot_style   <- match.arg(plot_style)
  label_corner <- match.arg(label_corner)
  style        <- get_plot_style(plot_style)
  
  if (nrow(plot_df) == 0) {
    return(ggplot() + theme_void() + labs(title = paste("No data for", y_lab)))
  }
  
  annot_df <- build_annotation_df(stats_df, label_corner)
  pj       <- position_jitter(width = 0.06, height = 0, seed = 42)
  
  p <- ggplot(plot_df, aes(x = comparison, y = value, fill = comparison)) +
    geom_boxplot(
      width         = 0.45,
      outlier.shape = NA,
      colour        = "black",
      linewidth     = 0.6,
      show.legend   = FALSE
    ) +
    geom_point(
      aes(fill = comparison),
      position    = pj,
      shape       = 21,
      size        = 2.0,
      stroke      = 0.35,
      colour      = "black",
      alpha       = style$point_alpha,
      show.legend = FALSE
    )
  
  if (isTRUE(show_p_values)) {
    p <- p +
      geom_label(
        data          = annot_df,
        aes(x = x, y = y, label = label, hjust = hjust, vjust = vjust),
        inherit.aes   = FALSE,
        size          = 2.8,
        fontface      = "bold",
        label.size    = 0.35,
        label.padding = unit(0.20, "lines"),
        label.r       = unit(0.10, "lines"),
        fill          = style$label_fill,
        colour        = "black"
      )
  }
  
  p +
    facet_wrap(~ model, ncol = 2, drop = TRUE) +
    scale_fill_manual(values = style$fill_values) +
    scale_x_discrete(labels = c("cross" = "Cross", "within" = "Within")) +
    scale_y_continuous(
      limits = c(0, 1.08),
      breaks = seq(0, 1, 0.2),
      expand = expansion(mult = c(0.02, 0))
    ) +
    labs(y = y_lab) +
    theme_clean(plot_style)
}

build_h1_slopeplot <- function(plot_df, stats_df, y_lab,
                               plot_style    = c("bw", "color"),
                               label_corner  = c("top_left", "bottom_right"),
                               show_p_values = TRUE) {
  plot_style   <- match.arg(plot_style)
  label_corner <- match.arg(label_corner)
  style        <- get_plot_style(plot_style)
  
  if (nrow(plot_df) == 0) {
    return(ggplot() + theme_void() + labs(title = paste("No data for", y_lab)))
  }
  
  annot_df <- build_annotation_df(stats_df, label_corner)
  
  point_df <- plot_df %>%
    pivot_longer(
      cols      = c(cross, within),
      names_to  = "comparison",
      values_to = "value"
    ) %>%
    mutate(comparison = factor(comparison, levels = comparison_levels))
  
  p <- ggplot() +
    geom_segment(
      data = plot_df,
      aes(x = 1, xend = 2, y = cross, yend = within, group = dataset_key),
      colour    = style$line_colour,
      linewidth = 0.35,
      alpha     = 0.75
    ) +
    geom_point(
      data = point_df,
      aes(x = comparison, y = value, fill = comparison),
      shape       = 21,
      size        = 2.2,
      stroke      = 0.35,
      colour      = "black",
      alpha       = style$point_alpha,
      show.legend = FALSE
    )
  
  if (isTRUE(show_p_values)) {
    p <- p +
      geom_label(
        data          = annot_df,
        aes(x = x, y = y, label = label, hjust = hjust, vjust = vjust),
        inherit.aes   = FALSE,
        size          = 2.8,
        fontface      = "bold",
        label.size    = 0.35,
        label.padding = unit(0.20, "lines"),
        label.r       = unit(0.10, "lines"),
        fill          = style$label_fill,
        colour        = "black"
      )
  }
  
  p +
    facet_wrap(~ model, ncol = 2, drop = TRUE) +
    scale_fill_manual(values = style$fill_values) +
    scale_x_discrete(labels = c("cross" = "Cross", "within" = "Within")) +
    scale_y_continuous(
      limits = c(0, 1.08),
      breaks = seq(0, 1, 0.2),
      expand = expansion(mult = c(0.02, 0))
    ) +
    labs(y = y_lab) +
    theme_clean(plot_style)
}

save_plot <- function(plot_obj, file, width_mm = 150, height_mm = 130) {
  ggsave(
    filename = file,
    plot     = plot_obj,
    width    = width_mm,
    height   = height_mm,
    units    = "mm",
    dpi      = 600,
    bg       = "white"
  )
}


# ---- 6.5 Reporting -----------------------------------------------------------
# Assemble and write the plain-text report.

append_section <- function(lines, title, obj = NULL) {
  section_header <- c(
    paste0(strrep("=", 18), " ", title, " ", strrep("=", 18))
  )
  
  if (is.null(obj)) {
    return(c(lines, "", section_header, ""))
  }
  
  body <- capture.output(print(obj))
  c(lines, "", section_header, body)
}

write_txt_report <- function(file, lines) {
  writeLines(enc2utf8(lines), con = file, useBytes = TRUE)
}


# ── 7. Load data and prepare ──────────────────────────────────────────────────
print_section("LOADING H1 INPUTS")

metrics_raw <- pmap_dfr(metric_paths, read_metric_file)

if (nrow(metrics_raw) == 0) {
  stop("No H1 input files found.")
}

metrics_h1 <- metrics_raw %>%
  mutate(
    dataset_id  = best_dataset_id(.),
    dataset_key = normalize_id(dataset_id),
    model       = factor(model, levels = model_levels),
    comparison  = factor(comparison, levels = comparison_levels)
  ) %>%
  filter(comparison %in% comparison_levels) %>%
  select(
    model, comparison, dataset_id, dataset_key,
    accuracy, macro_recall, macro_precision, macro_f1,
    source_file, everything()
  )

input_overview <- metrics_h1 %>%
  group_by(model, comparison) %>%
  summarise(
    rows       = n(),
    n_datasets = n_distinct(dataset_key, na.rm = TRUE),
    n_files    = n_distinct(source_file),
    .groups    = "drop"
  )

id_check <- metrics_h1 %>%
  group_by(model, comparison) %>%
  summarise(
    n_rows        = n(),
    n_dataset_ids = n_distinct(dataset_key, na.rm = TRUE),
    sample_ids    = paste(head(sort(unique(dataset_key[!is.na(dataset_key)])), 5), collapse = ", "),
    .groups       = "drop"
  )

pair_check <- metrics_h1 %>%
  filter(!is.na(dataset_key)) %>%
  count(model, dataset_key, comparison) %>%
  pivot_wider(names_from = comparison, values_from = n, values_fill = 0) %>%
  mutate(is_paired = cross > 0 & within > 0) %>%
  arrange(model, dataset_key)

pair_check_summary <- pair_check %>%
  group_by(model) %>%
  summarise(
    paired_ids  = sum(is_paired, na.rm = TRUE),
    only_cross  = sum(cross > 0 & within == 0, na.rm = TRUE),
    only_within = sum(within > 0 & cross == 0, na.rm = TRUE),
    .groups     = "drop"
  )


# ── 8. Summary stats ──────────────────────────────────────────────────────────
print_section("SUMMARY BY MODEL AND COMPARISON")

summary_h1 <- metrics_h1 %>%
  group_by(model, comparison) %>%
  summarise(
    n               = sum(is.finite(macro_f1)),
    n_datasets      = n_distinct(dataset_key[is.finite(macro_f1)], na.rm = TRUE),
    mean_accuracy   = mean(accuracy, na.rm = TRUE),
    sd_accuracy     = sd(accuracy, na.rm = TRUE),
    median_accuracy = median(accuracy, na.rm = TRUE),
    q1_accuracy     = quantile(accuracy, probs = 0.25, na.rm = TRUE, names = FALSE),
    q3_accuracy     = quantile(accuracy, probs = 0.75, na.rm = TRUE, names = FALSE),
    mean_macro_f1   = mean(macro_f1, na.rm = TRUE),
    sd_macro_f1     = sd(macro_f1, na.rm = TRUE),
    median_macro_f1 = median(macro_f1, na.rm = TRUE),
    q1_macro_f1     = quantile(macro_f1, probs = 0.25, na.rm = TRUE, names = FALSE),
    q3_macro_f1     = quantile(macro_f1, probs = 0.75, na.rm = TRUE, names = FALSE),
    .groups         = "drop"
  )


# ── 9. Paired Wilcoxon tests ──────────────────────────────────────────────────
print_section("PAIRED WILCOXON TESTS")

paired_macro_f1 <- metrics_h1 %>%
  group_by(model) %>%
  group_modify(~ run_paired_wilcox(.x, "macro_f1"), .keep = TRUE) %>%
  ungroup()

paired_accuracy <- metrics_h1 %>%
  group_by(model) %>%
  group_modify(~ run_paired_wilcox(.x, "accuracy"), .keep = TRUE) %>%
  ungroup()


# ── 10. Paired summaries ──────────────────────────────────────────────────────
paired_macro_f1_summary <- metrics_h1 %>%
  group_by(model) %>%
  group_modify(~ summarise_paired_metric(.x, "macro_f1"), .keep = TRUE) %>%
  ungroup()

paired_accuracy_summary <- metrics_h1 %>%
  group_by(model) %>%
  group_modify(~ summarise_paired_metric(.x, "accuracy"), .keep = TRUE) %>%
  ungroup()


# ── 11. Plot data ─────────────────────────────────────────────────────────────
paired_plot_macro_f1 <- build_paired_long_all(metrics_h1, "macro_f1")
paired_plot_accuracy <- build_paired_long_all(metrics_h1, "accuracy")

paired_wide_macro_f1 <- build_paired_wide_all(metrics_h1, "macro_f1")
paired_wide_accuracy <- build_paired_wide_all(metrics_h1, "accuracy")

paired_plot_check_macro_f1 <- paired_plot_macro_f1 %>%
  group_by(model, comparison) %>%
  summarise(n_points = n(), n_ids = n_distinct(dataset_key), .groups = "drop")

paired_plot_check_accuracy <- paired_plot_accuracy %>%
  group_by(model, comparison) %>%
  summarise(n_points = n(), n_ids = n_distinct(dataset_key), .groups = "drop")

paired_slope_check_macro_f1 <- paired_wide_macro_f1 %>%
  group_by(model) %>%
  summarise(n_pairs = n(), .groups = "drop")

paired_slope_check_accuracy <- paired_wide_accuracy %>%
  group_by(model) %>%
  summarise(n_pairs = n(), .groups = "drop")

paired_id_registry <- paired_wide_macro_f1 %>%
  distinct(model, dataset_key, dataset_id) %>%
  arrange(model, dataset_key)


# ── 12. Main tables ───────────────────────────────────────────────────────────
h1_macro_f1_table <- paired_macro_f1_summary %>%
  rename(
    paired_n_summary = paired_n
  ) %>%
  left_join(
    paired_macro_f1 %>%
      rename(
        paired_n_test              = paired_n,
        wilcoxon_v_macro_f1        = statistic,
        p_macro_f1                 = p_value,
        paired_mean_gap_macro_f1   = mean_gap,
        paired_median_gap_macro_f1 = median_gap,
        rank_biserial_macro_f1     = rank_biserial
      ),
    by = "model"
  ) %>%
  mutate(
    cross_macro_f1_mean_sd     = format_mean_sd(mean_cross, sd_cross),
    within_macro_f1_mean_sd    = format_mean_sd(mean_within, sd_within),
    cross_macro_f1_median_iqr  = format_median_iqr(median_cross, q1_cross, q3_cross),
    within_macro_f1_median_iqr = format_median_iqr(median_within, q1_within, q3_within)
  ) %>%
  transmute(
    model,
    cross_n           = paired_n_summary,
    within_n          = paired_n_summary,
    paired_n          = paired_n_test,
    cross_macro_f1    = cross_macro_f1_mean_sd,
    within_macro_f1   = within_macro_f1_mean_sd,
    cross_macro_f1_median_iqr,
    within_macro_f1_median_iqr,
    paired_mean_gap   = round(paired_mean_gap_macro_f1, 3),
    paired_median_gap = round(paired_median_gap_macro_f1, 3),
    wilcoxon_v        = round(wilcoxon_v_macro_f1, 3),
    rank_biserial     = round(rank_biserial_macro_f1, 3),
    p_value           = signif(p_macro_f1, 3)
  )

h1_accuracy_table <- paired_accuracy_summary %>%
  rename(
    paired_n_summary = paired_n
  ) %>%
  left_join(
    paired_accuracy %>%
      rename(
        paired_n_test              = paired_n,
        wilcoxon_v_accuracy        = statistic,
        p_accuracy                 = p_value,
        paired_mean_gap_accuracy   = mean_gap,
        paired_median_gap_accuracy = median_gap,
        rank_biserial_accuracy     = rank_biserial
      ),
    by = "model"
  ) %>%
  mutate(
    cross_accuracy_mean_sd     = format_mean_sd(mean_cross, sd_cross),
    within_accuracy_mean_sd    = format_mean_sd(mean_within, sd_within),
    cross_accuracy_median_iqr  = format_median_iqr(median_cross, q1_cross, q3_cross),
    within_accuracy_median_iqr = format_median_iqr(median_within, q1_within, q3_within)
  ) %>%
  transmute(
    model,
    cross_n           = paired_n_summary,
    within_n          = paired_n_summary,
    paired_n          = paired_n_test,
    cross_accuracy    = cross_accuracy_mean_sd,
    within_accuracy   = within_accuracy_mean_sd,
    cross_accuracy_median_iqr,
    within_accuracy_median_iqr,
    paired_mean_gap   = round(paired_mean_gap_accuracy, 3),
    paired_median_gap = round(paired_median_gap_accuracy, 3),
    wilcoxon_v        = round(wilcoxon_v_accuracy, 3),
    rank_biserial     = round(rank_biserial_accuracy, 3),
    p_value           = signif(p_accuracy, 3)
  )


# ── 13. Reporting focus ───────────────────────────────────────────────────────
reporting_focus <- bind_rows(
  h1_macro_f1_table %>%
    transmute(
      component = paste0("h1_macro_f1_", model),
      detail    = paste0(
        "cross = ", cross_macro_f1,
        "; within = ", within_macro_f1,
        "; paired n = ", paired_n,
        "; mean gap = ", fmt_num(paired_mean_gap),
        "; r_rb = ", fmt_num(rank_biserial),
        "; p = ", fmt_p(p_value)
      )
    ),
  h1_accuracy_table %>%
    transmute(
      component = paste0("h1_accuracy_", model),
      detail    = paste0(
        "cross = ", cross_accuracy,
        "; within = ", within_accuracy,
        "; paired n = ", paired_n,
        "; mean gap = ", fmt_num(paired_mean_gap),
        "; r_rb = ", fmt_num(rank_biserial),
        "; p = ", fmt_p(p_value)
      )
    )
)


# ── 14. Plots ─────────────────────────────────────────────────────────────────
print_section("GENERATING PLOTS")

plot_index <- tibble(
  file        = character(),
  description = character()
)

for (plot_style in plot_styles) {
  style_suffix <- ifelse(plot_style == "bw", "bw", "color")
  label_note   <- ifelse(show_p_values, "with p-value label", "without p-value label")
  
  p_macro_f1_box <- build_h1_boxplot(
    plot_df       = paired_plot_macro_f1,
    stats_df      = paired_macro_f1,
    y_lab         = "Macro-F1 Score",
    plot_style    = plot_style,
    label_corner  = p_value_position,
    show_p_values = show_p_values
  )
  
  p_accuracy_box <- build_h1_boxplot(
    plot_df       = paired_plot_accuracy,
    stats_df      = paired_accuracy,
    y_lab         = "Accuracy",
    plot_style    = plot_style,
    label_corner  = p_value_position,
    show_p_values = show_p_values
  )
  
  p_macro_f1_slope <- build_h1_slopeplot(
    plot_df       = paired_wide_macro_f1,
    stats_df      = paired_macro_f1,
    y_lab         = "Macro-F1 Score",
    plot_style    = plot_style,
    label_corner  = p_value_position,
    show_p_values = show_p_values
  )
  
  p_accuracy_slope <- build_h1_slopeplot(
    plot_df       = paired_wide_accuracy,
    stats_df      = paired_accuracy,
    y_lab         = "Accuracy",
    plot_style    = plot_style,
    label_corner  = p_value_position,
    show_p_values = show_p_values
  )
  
  macro_box_file   <- paste0("01_h1_macro_f1_boxplot_points_", style_suffix, ".png")
  acc_box_file     <- paste0("02_h1_accuracy_boxplot_points_", style_suffix, ".png")
  macro_slope_file <- paste0("03_h1_macro_f1_paired_slope_",   style_suffix, ".png")
  acc_slope_file   <- paste0("04_h1_accuracy_paired_slope_",   style_suffix, ".png")
  
  save_plot(p_macro_f1_box,   file.path(plots_dir, macro_box_file))
  save_plot(p_accuracy_box,   file.path(plots_dir, acc_box_file))
  save_plot(p_macro_f1_slope, file.path(plots_dir, macro_slope_file))
  save_plot(p_accuracy_slope, file.path(plots_dir, acc_slope_file))
  
  plot_index <- bind_rows(
    plot_index,
    tibble(
      file        = c(macro_box_file, acc_box_file, macro_slope_file, acc_slope_file),
      description = c(
        paste0("Combined Macro-F1 boxplot with points, ",  label_note, " (", style_suffix, ")"),
        paste0("Combined Accuracy boxplot with points, ",  label_note, " (", style_suffix, ")"),
        paste0("Combined Macro-F1 paired slope plot, ",    label_note, " (", style_suffix, ")"),
        paste0("Combined Accuracy paired slope plot, ",    label_note, " (", style_suffix, ")")
      )
    )
  )
}


# ── 15. Export CSV ────────────────────────────────────────────────────────────
print_section("EXPORTING CSV")

write_csv(metric_paths,          file.path(csv_dir, "00_h1_input_registry.csv"))
write_csv(input_overview,        file.path(csv_dir, "01_h1_input_overview.csv"))
write_csv(metrics_h1,            file.path(csv_dir, "02_h1_metrics_long.csv"))
write_csv(summary_h1,            file.path(csv_dir, "03_h1_summary_by_model_comparison.csv"))
write_csv(paired_macro_f1,       file.path(csv_dir, "04_h1_paired_wilcox_macro_f1.csv"))
write_csv(paired_accuracy,       file.path(csv_dir, "05_h1_paired_wilcox_accuracy.csv"))
write_csv(h1_macro_f1_table,     file.path(csv_dir, "06_h1_macro_f1_table_paired.csv"))
write_csv(h1_accuracy_table,     file.path(csv_dir, "07_h1_accuracy_table_paired.csv"))
write_csv(pair_check_summary,    file.path(csv_dir, "08_h1_pair_check_summary.csv"))
write_csv(paired_id_registry,    file.path(csv_dir, "09_h1_paired_dataset_registry.csv"))
write_csv(paired_plot_macro_f1,  file.path(csv_dir, "10_h1_macro_f1_paired_plot_long.csv"))
write_csv(paired_plot_accuracy,  file.path(csv_dir, "11_h1_accuracy_paired_plot_long.csv"))
write_csv(paired_wide_macro_f1,  file.path(csv_dir, "12_h1_macro_f1_paired_plot_wide.csv"))
write_csv(paired_wide_accuracy,  file.path(csv_dir, "13_h1_accuracy_paired_plot_wide.csv"))
write_csv(plot_index,            file.path(csv_dir, "14_h1_plot_index.csv"))
write_csv(reporting_focus,       file.path(csv_dir, "15_h1_reporting_focus.csv"))


# ── 16. Export TXT ────────────────────────────────────────────────────────────
print_section("EXPORTING TXT")

report_lines <- c(
  paste0("H1 output folder: ", out_dir),
  paste0("Created on: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "",
  "This report contains the main H1 results and checks.",
  ""
)

report_lines <- append_section(report_lines, "INPUT OVERVIEW",                 input_overview)
report_lines <- append_section(report_lines, "DATASET ID CHECK",               id_check)
report_lines <- append_section(report_lines, "PAIR CHECK SUMMARY",             pair_check_summary)
report_lines <- append_section(report_lines, "SUMMARY BY MODEL AND COMPARISON", summary_h1)
report_lines <- append_section(report_lines, "PAIRED WILCOXON MACRO-F1",       paired_macro_f1)
report_lines <- append_section(report_lines, "PAIRED WILCOXON ACCURACY",       paired_accuracy)
report_lines <- append_section(report_lines, "PAIRED MACRO-F1 SUMMARY",        paired_macro_f1_summary)
report_lines <- append_section(report_lines, "PAIRED ACCURACY SUMMARY",        paired_accuracy_summary)
report_lines <- append_section(report_lines, "MACRO-F1 MAIN TABLE",            h1_macro_f1_table)
report_lines <- append_section(report_lines, "ACCURACY MAIN TABLE",            h1_accuracy_table)
report_lines <- append_section(report_lines, "PAIRED PLOT CHECK MACRO-F1",     paired_plot_check_macro_f1)
report_lines <- append_section(report_lines, "PAIRED PLOT CHECK ACCURACY",     paired_plot_check_accuracy)
report_lines <- append_section(report_lines, "PAIRED SLOPE CHECK MACRO-F1",    paired_slope_check_macro_f1)
report_lines <- append_section(report_lines, "PAIRED SLOPE CHECK ACCURACY",    paired_slope_check_accuracy)
report_lines <- append_section(report_lines, "PLOT INDEX",                     plot_index)
report_lines <- append_section(report_lines, "REPORTING FOCUS",                reporting_focus)

write_txt_report(file.path(txt_dir, "01_h1_analysis_report.txt"), report_lines)

files_lines <- c(
  paste0("Output folder: ", out_dir),
  paste0("CSV folder: ",    csv_dir),
  paste0("Plots folder: ",  plots_dir),
  paste0("TXT folder: ",    txt_dir),
  "",
  "Written plot files:",
  paste0("- ", plot_index$file),
  "",
  "Written CSV files:",
  paste0("- ", c(
    "00_h1_input_registry.csv",
    "01_h1_input_overview.csv",
    "02_h1_metrics_long.csv",
    "03_h1_summary_by_model_comparison.csv",
    "04_h1_paired_wilcox_macro_f1.csv",
    "05_h1_paired_wilcox_accuracy.csv",
    "06_h1_macro_f1_table_paired.csv",
    "07_h1_accuracy_table_paired.csv",
    "08_h1_pair_check_summary.csv",
    "09_h1_paired_dataset_registry.csv",
    "10_h1_macro_f1_paired_plot_long.csv",
    "11_h1_accuracy_paired_plot_long.csv",
    "12_h1_macro_f1_paired_plot_wide.csv",
    "13_h1_accuracy_paired_plot_wide.csv",
    "14_h1_plot_index.csv",
    "15_h1_reporting_focus.csv"
  ))
)

write_txt_report(file.path(txt_dir, "02_h1_files_written.txt"), files_lines)


# ── 17. Console output ────────────────────────────────────────────────────────
print_section("RESULTS")
print(summary_h1)
print(paired_macro_f1)
print(paired_accuracy)
print(h1_macro_f1_table)
print(h1_accuracy_table)
print(reporting_focus)

print_section("FILES WRITTEN")
print(plot_index)
cat("\nCSV folder:",   csv_dir,   "\n")
cat("Plots folder:",   plots_dir, "\n")
cat("TXT folder:",     txt_dir,   "\n")
