if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman", repos = "https://cloud.r-project.org")
}

pacman::p_load(readr, dplyr, tidyr, ggplot2, purrr, tibble, grid)

# =============================================================================
# H2 Analysis
# Cross-dataset performance differs across broad behavioural classes.
# =============================================================================

# ── User settings ─────────────────────────────────────────────────────────────
show_p_values   <- TRUE
p_value_position <- "bottom_right"   # "top_left" or "bottom_right"

# ── Paths ─────────────────────────────────────────────────────────────────────
base_dir <- "/Volumes/Z Slim/07_04_2026_Data_Analysis/Output"
out_dir  <- file.path(base_dir, "H2")

csv_dir   <- file.path(out_dir, "csv")
plots_dir <- file.path(out_dir, "plots")
txt_dir   <- file.path(out_dir, "txt")

for (d in c(out_dir, csv_dir, plots_dir, txt_dir)) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

# ── Constants ─────────────────────────────────────────────────────────────────
behavior_levels <- c("Foraging", "Locomotion", "Resting")
model_levels    <- c("CNN", "ResNet", "HYDRA", "MultiRocket", "RF", "LGBM")
comparison_levels <- c("cross", "within")
plot_styles     <- c("bw", "color")
required_cols   <- c("behavior", "support", "predicted", "tp", "recall", "precision", "f1")

# ── Input registry ────────────────────────────────────────────────────────────
behavior_paths <- tribble(
  ~model,          ~comparison, ~aggregated_file,                                                                 ~stats_dir,
  "RF",            "within",    file.path(base_dir, "RF",            "Within_RF",            "statistics", "within_behavior_metrics_all.csv"),   file.path(base_dir, "RF",            "Within_RF",            "statistics"),
  "RF",            "cross",     file.path(base_dir, "RF",            "Cross_RF",             "statistics", "cross_behavior_metrics_all.csv"),    file.path(base_dir, "RF",            "Cross_RF",             "statistics"),
  "LGBM",          "within",    file.path(base_dir, "LGBM",          "Within_LGBM",          "statistics", "within_behavior_metrics_all.csv"), file.path(base_dir, "LGBM",          "Within_LGBM",          "statistics"),
  "LGBM",          "cross",     file.path(base_dir, "LGBM",          "Cross_LGBM",           "statistics", "cross_behavior_metrics_all.csv"),  file.path(base_dir, "LGBM",          "Cross_LGBM",           "statistics"),
  "CNN",           "within",    NA_character_,                                                                       file.path(base_dir, "CNN",           "Within_CNN",            "statistics"),
  "CNN",           "cross",     NA_character_,                                                                       file.path(base_dir, "CNN",           "Cross_CNN",             "statistics"),
  "ResNet",       "within",    NA_character_,                                                                       file.path(base_dir, "ResNet",       "Within_ResNet",        "statistics"),
  "ResNet",       "cross",     NA_character_,                                                                       file.path(base_dir, "ResNet",       "Cross_ResNet",         "statistics"),
  "HYDRA",         "within",    NA_character_,                                                                       file.path(base_dir, "HYDRA",         "Within_HYDRA",          "statistics"),
  "HYDRA",         "cross",     NA_character_,                                                                       file.path(base_dir, "HYDRA",         "Cross_HYDRA",           "statistics"),
  "MultiRocket",   "within",    NA_character_,                                                                       file.path(base_dir, "MultiRocket",   "Within_MultiRocket",    "statistics"),
  "MultiRocket",   "cross",     NA_character_,                                                                       file.path(base_dir, "MultiRocket",   "Cross_MultiRocket",     "statistics")
)

# ── Helpers ───────────────────────────────────────────────────────────────────
print_section <- function(x) {
  cat("\n", strrep("=", 18), x, strrep("=", 18), "\n", sep = " ")
}

fmt_num <- function(x, digits = 3) {
  ifelse(is.na(x), "NA", formatC(x, format = "f", digits = digits))
}

fmt_p <- function(x) {
  ifelse(is.na(x), "NA", ifelse(x < 0.001, "< 0.001", sprintf("%.3f", round(x, 3))))
}

format_median_iqr <- function(median_x, q1_x, q3_x) {
  ifelse(
    is.na(median_x),
    "NA",
    paste0(fmt_num(median_x), " [", fmt_num(q1_x), ", ", fmt_num(q3_x), "]")
  )
}

first_existing_col <- function(df, candidates) {
  hit <- candidates[candidates %in% names(df)][1]
  
  if (length(hit) == 0 || is.na(hit)) {
    return(rep(NA_character_, nrow(df)))
  }
  
  as.character(df[[hit]])
}

safe_ratio <- function(num, den) {
  out <- num / den
  out[!is.finite(out)] <- NA_real_
  out
}

direction_from_diff <- function(x) {
  case_when(
    is.na(x) ~ "NA",
    x > 0 ~ "first > second",
    x < 0 ~ "first < second",
    TRUE ~ "no median difference"
  )
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

collect_recursive_behavior_files <- function(stats_dir) {
  if (!dir.exists(stats_dir)) {
    return(character())
  }
  
  files <- list.files(stats_dir, recursive = TRUE, full.names = TRUE)
  keep  <- basename(files) %in% c("behavior_metrics.csv", "metrics_by_behavior.csv")
  files[keep]
}

read_behavior_source <- function(model, comparison, aggregated_file, stats_dir) {
  if (!is.na(aggregated_file) && file.exists(aggregated_file)) {
    dt <- read_csv(aggregated_file, show_col_types = FALSE)
    missing_cols <- setdiff(required_cols, names(dt))
    
    if (length(missing_cols) > 0) {
      stop(
        model, " ", comparison, " aggregated file missing cols: ",
        paste(missing_cols, collapse = ", ")
      )
    }
    
    return(
      dt %>%
        mutate(
          model       = model,
          comparison  = comparison,
          source_file = aggregated_file,
          dataset_id  = first_existing_col(., c("analysis_dataset", "test_dataset", "pair_id"))
        )
    )
  }
  
  files <- collect_recursive_behavior_files(stats_dir)
  
  if (length(files) == 0) {
    return(tibble())
  }
  
  map_dfr(files, function(f) {
    dt <- read_csv(f, show_col_types = FALSE)
    missing_cols <- setdiff(required_cols, names(dt))
    
    if (length(missing_cols) > 0) {
      stop(
        model, " ", comparison, " file missing cols: ", f,
        " | missing: ", paste(missing_cols, collapse = ", ")
      )
    }
    
    parent_id <- basename(dirname(f))
    
    dt %>%
      mutate(
        model       = model,
        comparison  = comparison,
        source_file = f,
        dataset_id  = first_existing_col(., c("analysis_dataset", "test_dataset", "pair_id")),
        dataset_id  = ifelse(is.na(dataset_id) | dataset_id == "", parent_id, dataset_id)
      )
  })
}

# ── Statistics ────────────────────────────────────────────────────────────────
safe_friedman_extended <- function(df, value_col, group_col, block_col,
                                   group_levels = behavior_levels) {
  use_dt <- df %>%
    filter(is.finite(.data[[value_col]]), !is.na(.data[[group_col]]), !is.na(.data[[block_col]])) %>%
    mutate(group_tmp = factor(.data[[group_col]], levels = group_levels)) %>%
    filter(!is.na(group_tmp)) %>%
    select(all_of(c(value_col, block_col)), group_tmp)
  
  complete_blocks <- use_dt %>%
    count(.data[[block_col]], group_tmp, name = "n") %>%
    count(.data[[block_col]], name = "n_groups") %>%
    filter(n_groups == length(group_levels)) %>%
    pull(.data[[block_col]])
  
  use_dt <- use_dt %>% filter(.data[[block_col]] %in% complete_blocks)
  n_blocks <- dplyr::n_distinct(use_dt[[block_col]])
  n_groups <- length(group_levels)
  
  if (n_blocks < 2 || n_groups < 2) {
    return(tibble(
      n_blocks = n_blocks,
      statistic = NA_real_,
      df = NA_real_,
      p_value = NA_real_,
      kendalls_w = NA_real_
    ))
  }
  
  fml <- as.formula(paste(value_col, "~ group_tmp |", block_col))
  tmp <- tryCatch(friedman.test(fml, data = use_dt), error = function(e) NULL)
  
  if (is.null(tmp)) {
    return(tibble(
      n_blocks = n_blocks,
      statistic = NA_real_,
      df = NA_real_,
      p_value = NA_real_,
      kendalls_w = NA_real_
    ))
  }
  
  chi_sq <- unname(tmp$statistic)
  
  tibble(
    n_blocks   = n_blocks,
    statistic  = chi_sq,
    df         = unname(tmp$parameter),
    p_value    = tmp$p.value,
    kendalls_w = chi_sq / (n_blocks * (n_groups - 1))
  )
}

safe_pairwise_wilcox_extended <- function(df, value_col, group_col, block_col,
                                          group_levels = behavior_levels) {
  use_dt <- df %>%
    filter(is.finite(.data[[value_col]]), !is.na(.data[[group_col]]), !is.na(.data[[block_col]])) %>%
    mutate(group_tmp = factor(.data[[group_col]], levels = group_levels)) %>%
    filter(!is.na(group_tmp)) %>%
    select(all_of(c(value_col, block_col)), group_tmp)
  
  complete_blocks <- use_dt %>%
    count(.data[[block_col]], group_tmp, name = "n") %>%
    count(.data[[block_col]], name = "n_groups") %>%
    filter(n_groups == length(group_levels)) %>%
    pull(.data[[block_col]])
  
  use_dt <- use_dt %>% filter(.data[[block_col]] %in% complete_blocks)
  
  if (nrow(use_dt) == 0 || dplyr::n_distinct(use_dt[[block_col]]) < 2) {
    return(tibble(
      contrast = character(),
      n_pairs = integer(),
      statistic = numeric(),
      p_value = numeric(),
      p_adj_holm = numeric(),
      median_diff = numeric(),
      mean_diff = numeric(),
      rank_biserial = numeric(),
      direction = character()
    ))
  }
  
  wide_dt <- use_dt %>%
    mutate(group_tmp = as.character(group_tmp)) %>%
    select(all_of(block_col), group_tmp, all_of(value_col)) %>%
    distinct() %>%
    pivot_wider(names_from = group_tmp, values_from = all_of(value_col))
  
  pairs <- combn(group_levels, 2, simplify = FALSE)
  
  out <- map_dfr(pairs, function(pair_now) {
    x <- wide_dt[[pair_now[1]]]
    y <- wide_dt[[pair_now[2]]]
    
    ok <- is.finite(x) & is.finite(y)
    x <- x[ok]
    y <- y[ok]
    
    diffs <- x - y
    keep_nonzero <- is.finite(diffs) & diffs != 0
    x_test <- x[keep_nonzero]
    y_test <- y[keep_nonzero]
    diffs_test <- diffs[keep_nonzero]
    n_pairs <- length(diffs_test)
    
    if (n_pairs < 2) {
      return(tibble(
        contrast = paste(pair_now, collapse = " vs "),
        n_pairs = n_pairs,
        statistic = NA_real_,
        p_value = NA_real_,
        median_diff = ifelse(length(diffs) == 0, NA_real_, median(diffs, na.rm = TRUE)),
        mean_diff = ifelse(length(diffs) == 0, NA_real_, mean(diffs, na.rm = TRUE)),
        rank_biserial = NA_real_,
        direction = direction_from_diff(ifelse(length(diffs) == 0, NA_real_, median(diffs, na.rm = TRUE)))
      ))
    }
    
    tmp <- tryCatch(
      wilcox.test(x_test, y_test, paired = TRUE, exact = FALSE),
      error = function(e) NULL
    )
    
    stat <- if (is.null(tmp)) NA_real_ else unname(tmp$statistic)
    total_rank_sum <- n_pairs * (n_pairs + 1) / 2
    rbc <- if (is.na(stat)) NA_real_ else (2 * stat / total_rank_sum) - 1
    
    tibble(
      contrast = paste(pair_now, collapse = " vs "),
      n_pairs = n_pairs,
      statistic = stat,
      p_value = if (is.null(tmp)) NA_real_ else tmp$p.value,
      median_diff = median(diffs_test, na.rm = TRUE),
      mean_diff = mean(diffs_test, na.rm = TRUE),
      rank_biserial = rbc,
      direction = direction_from_diff(median(diffs_test, na.rm = TRUE))
    )
  })
  
  out %>%
    mutate(p_adj_holm = p.adjust(p_value, method = "holm"))
}

# ── Summaries ─────────────────────────────────────────────────────────────────
mean_ci_tbl <- function(df, value_col) {
  df %>%
    filter(is.finite(.data[[value_col]])) %>%
    group_by(model, behavior) %>%
    summarise(
      n          = n(),
      mean_value = mean(.data[[value_col]], na.rm = TRUE),
      sd_value   = sd(.data[[value_col]], na.rm = TRUE),
      se_value   = sd_value / sqrt(n),
      ci_low     = mean_value - 1.96 * se_value,
      ci_high    = mean_value + 1.96 * se_value,
      .groups    = "drop"
    )
}

median_iqr_tbl <- function(df, value_col) {
  df %>%
    filter(is.finite(.data[[value_col]])) %>%
    group_by(model, behavior) %>%
    summarise(
      n            = n(),
      n_datasets   = n_distinct(dataset_id),
      mean_value   = mean(.data[[value_col]], na.rm = TRUE),
      sd_value     = sd(.data[[value_col]], na.rm = TRUE),
      median_value = median(.data[[value_col]], na.rm = TRUE),
      q1_value     = quantile(.data[[value_col]], 0.25, na.rm = TRUE, names = FALSE),
      q3_value     = quantile(.data[[value_col]], 0.75, na.rm = TRUE, names = FALSE),
      iqr_value    = IQR(.data[[value_col]], na.rm = TRUE),
      .groups      = "drop"
    )
}

# ── Plot style ────────────────────────────────────────────────────────────────
get_plot_style <- function(plot_style = c("bw", "color")) {
  plot_style <- match.arg(plot_style)
  
  if (plot_style == "color") {
    return(list(
      behavior_fill = c(
        "Foraging" = "#E69F00",
        "Locomotion" = "#009E73",
        "Resting" = "#CC79A7"
      ),
      comparison_fill = c(
        "cross" = "#0072B2",
        "within" = "#D55E00"
      ),
      line_colour = "grey70",
      point_colour = "grey20",
      point_alpha = 0.90,
      strip_fill = "grey95",
      label_fill = "white",
      heatmap_low = "#F7FBFF",
      heatmap_mid = "#6BAED6",
      heatmap_high = "#08306B"
    ))
  }
  
  list(
    behavior_fill = c(
      "Foraging" = "grey85",
      "Locomotion" = "grey60",
      "Resting" = "white"
    ),
    comparison_fill = c(
      "cross" = "grey80",
      "within" = "white"
    ),
    line_colour = "grey70",
    point_colour = "grey20",
    point_alpha = 0.95,
    strip_fill = "grey95",
    label_fill = "white",
    heatmap_low = "white",
    heatmap_mid = "grey70",
    heatmap_high = "black"
  )
}

theme_h2_clean <- function(plot_style = c("bw", "color")) {
  plot_style <- match.arg(plot_style)
  style <- get_plot_style(plot_style)
  
  theme_bw(base_size = 10, base_family = "sans") +
    theme(
      plot.title       = element_blank(),
      plot.subtitle    = element_blank(),
      plot.caption     = element_blank(),
      axis.title.x     = element_blank(),
      axis.title.y     = element_text(size = 10, face = "plain", margin = margin(r = 8)),
      axis.text.x      = element_text(size = 10, colour = "black"),
      axis.text.y      = element_text(size = 10, colour = "black"),
      strip.background = element_rect(fill = style$strip_fill, colour = "black", linewidth = 0.6),
      strip.text       = element_text(face = "bold", size = 10, margin = margin(t = 4, b = 4)),
      panel.grid.minor   = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(colour = "grey85", linewidth = 0.3, linetype = "dashed"),
      panel.border     = element_rect(colour = "black", fill = NA, linewidth = 0.8),
      plot.background  = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      legend.position  = "none",
      plot.margin      = margin(10, 10, 10, 10)
    )
}

build_facet_p_labels <- function(stats_df,
                                 label_col = "p_value",
                                 position = c("top_left", "bottom_right")) {
  position <- match.arg(position)
  
  if (!label_col %in% names(stats_df) || !"model" %in% names(stats_df)) {
    return(tibble())
  }
  
  if (position == "top_left") {
    return(
      stats_df %>%
        transmute(
          model,
          x = 0.62,
          y = Inf,
          hjust = 0,
          vjust = 1.1,
          label = paste0("p = ", fmt_p(.data[[label_col]]))
        )
    )
  }
  
  stats_df %>%
    transmute(
      model,
      x = Inf,
      y = -Inf,
      hjust = 1.02,
      vjust = -0.20,
      label = paste0("p = ", fmt_p(.data[[label_col]]))
    )
}

add_facet_p_labels <- function(plot_obj,
                               annot_df,
                               plot_style = c("bw", "color"),
                               show_p_values = TRUE) {
  plot_style <- match.arg(plot_style)
  style <- get_plot_style(plot_style)
  
  if (!isTRUE(show_p_values) || nrow(annot_df) == 0) {
    return(plot_obj)
  }
  
  plot_obj +
    geom_label(
      data = annot_df,
      aes(x = x, y = y, label = label, hjust = hjust, vjust = vjust),
      inherit.aes = FALSE,
      size = 2.8,
      fontface = "bold",
      label.size = 0.35,
      label.padding = unit(0.20, "lines"),
      label.r = unit(0.10, "lines"),
      fill = style$label_fill,
      colour = "black"
    ) +
    coord_cartesian(clip = "off")
}

save_plot <- function(plot_obj, file, width_mm = 180, height_mm = 130) {
  ggsave(
    filename = file.path(plots_dir, file),
    plot = plot_obj,
    width = width_mm,
    height = height_mm,
    units = "mm",
    dpi = 600,
    bg = "white"
  )
}

# ── Load data ─────────────────────────────────────────────────────────────────
print_section("LOADING H2 INPUTS")

behavior_long <- pmap_dfr(behavior_paths, read_behavior_source) %>%
  mutate(
    model      = factor(model, levels = model_levels),
    comparison = factor(comparison, levels = comparison_levels),
    behavior   = factor(behavior, levels = behavior_levels),
    dataset_id = as.character(dataset_id)
  ) %>%
  select(
    model, comparison, dataset_id, behavior,
    support, predicted, tp, recall, precision, f1,
    source_file, everything()
  )

if (nrow(behavior_long) == 0) {
  stop("No behavior-specific files found.")
}

input_overview <- behavior_long %>%
  group_by(model, comparison) %>%
  summarise(
    rows       = n(),
    n_datasets = n_distinct(dataset_id),
    n_files    = n_distinct(source_file),
    .groups    = "drop"
  )

summary_behavior <- behavior_long %>%
  group_by(model, comparison, behavior) %>%
  summarise(
    n              = n(),
    n_datasets     = n_distinct(dataset_id),
    mean_f1        = mean(f1, na.rm = TRUE),
    sd_f1          = sd(f1, na.rm = TRUE),
    median_f1      = median(f1, na.rm = TRUE),
    q1_f1          = quantile(f1, 0.25, na.rm = TRUE, names = FALSE),
    q3_f1          = quantile(f1, 0.75, na.rm = TRUE, names = FALSE),
    mean_recall    = mean(recall, na.rm = TRUE),
    mean_precision = mean(precision, na.rm = TRUE),
    mean_support   = mean(support, na.rm = TRUE),
    .groups        = "drop"
  )

# ── H2 core: cross behaviour effect ──────────────────────────────────────────
print_section("CROSS BEHAVIOUR EFFECT")

cross_dt <- behavior_long %>%
  filter(comparison == "cross")

cross_complete_blocks <- cross_dt %>%
  count(model, dataset_id, behavior, name = "n") %>%
  group_by(model, dataset_id) %>%
  summarise(n_behaviors = n_distinct(behavior), .groups = "drop") %>%
  filter(n_behaviors == length(behavior_levels))

cross_dt_complete <- cross_dt %>%
  semi_join(cross_complete_blocks, by = c("model", "dataset_id")) %>%
  arrange(model, dataset_id, behavior)

cross_friedman <- cross_dt_complete %>%
  group_by(model) %>%
  group_modify(~ safe_friedman_extended(.x, "f1", "behavior", "dataset_id")) %>%
  ungroup()

cross_posthoc <- cross_dt_complete %>%
  group_by(model) %>%
  group_modify(~ safe_pairwise_wilcox_extended(.x, "f1", "behavior", "dataset_id")) %>%
  ungroup()

print(cross_friedman)
print(cross_posthoc)

# ── H2 core: transfer loss ───────────────────────────────────────────────────
print_section("TRANSFER LOSS")

within_dt <- behavior_long %>%
  filter(comparison == "within") %>%
  select(
    model, dataset_id, behavior,
    within_support = support,
    within_recall = recall,
    within_precision = precision,
    within_f1 = f1
  )

cross_only_dt <- cross_dt %>%
  select(
    model, dataset_id, behavior,
    cross_support = support,
    cross_recall = recall,
    cross_precision = precision,
    cross_f1 = f1
  )

transfer_dt <- cross_only_dt %>%
  left_join(within_dt, by = c("model", "dataset_id", "behavior")) %>%
  mutate(
    transfer_loss_f1        = within_f1 - cross_f1,
    transfer_retention_f1   = safe_ratio(cross_f1, within_f1),
    transfer_loss_recall    = within_recall - cross_recall,
    transfer_loss_precision = within_precision - cross_precision
  )

transfer_summary <- transfer_dt %>%
  group_by(model, behavior) %>%
  summarise(
    n                          = sum(is.finite(transfer_loss_f1)),
    mean_cross_f1              = mean(cross_f1, na.rm = TRUE),
    mean_within_f1             = mean(within_f1, na.rm = TRUE),
    mean_transfer_loss_f1      = mean(transfer_loss_f1, na.rm = TRUE),
    median_transfer_loss_f1    = median(transfer_loss_f1, na.rm = TRUE),
    sd_transfer_loss_f1        = sd(transfer_loss_f1, na.rm = TRUE),
    mean_transfer_retention_f1 = mean(transfer_retention_f1, na.rm = TRUE),
    .groups = "drop"
  )

transfer_complete_blocks <- transfer_dt %>%
  filter(is.finite(transfer_loss_f1)) %>%
  count(model, dataset_id, behavior) %>%
  group_by(model, dataset_id) %>%
  summarise(n_behaviors = n_distinct(behavior), .groups = "drop") %>%
  filter(n_behaviors == length(behavior_levels))

transfer_dt_complete <- transfer_dt %>%
  semi_join(transfer_complete_blocks, by = c("model", "dataset_id")) %>%
  arrange(model, dataset_id, behavior)

transfer_friedman <- transfer_dt_complete %>%
  group_by(model) %>%
  group_modify(~ safe_friedman_extended(.x, "transfer_loss_f1", "behavior", "dataset_id")) %>%
  ungroup()

transfer_posthoc <- transfer_dt_complete %>%
  group_by(model) %>%
  group_modify(~ safe_pairwise_wilcox_extended(.x, "transfer_loss_f1", "behavior", "dataset_id")) %>%
  ungroup()

print(transfer_friedman)
print(transfer_posthoc)

# ── Tables ────────────────────────────────────────────────────────────────────
print_section("TABLES")

cross_summary_paper <- median_iqr_tbl(cross_dt_complete, "f1") %>%
  rename(
    mean_f1   = mean_value,
    sd_f1     = sd_value,
    median_f1 = median_value,
    q1_f1     = q1_value,
    q3_f1     = q3_value,
    iqr_f1    = iqr_value
  )

h2_friedman_overview <- cross_friedman %>%
  transmute(
    model,
    n_datasets = n_blocks,
    friedman_chi_sq = round(statistic, 3),
    friedman_df = df,
    kendalls_w = round(kendalls_w, 3),
    p_value = fmt_p(p_value)
  )

h2_table_paper <- cross_summary_paper %>%
  mutate(
    median_iqr = format_median_iqr(median_f1, q1_f1, q3_f1)
  ) %>%
  select(model, behavior, n_datasets, median_iqr, median_f1, q1_f1, q3_f1, iqr_f1, mean_f1, sd_f1) %>%
  left_join(
    cross_friedman %>%
      transmute(
        model,
        friedman_n_blocks = n_blocks,
        friedman_chi_sq = round(statistic, 3),
        friedman_df = df,
        friedman_p = p_value,
        friedman_p_label = fmt_p(p_value),
        kendalls_w = round(kendalls_w, 3)
      ),
    by = "model"
  )

h2_posthoc_appendix <- cross_posthoc %>%
  separate(contrast, into = c("group_1", "group_2"), sep = " vs ", remove = FALSE) %>%
  mutate(
    median_diff = round(median_diff, 3),
    mean_diff = round(mean_diff, 3),
    statistic = round(statistic, 3),
    rank_biserial = round(rank_biserial, 3),
    p_label = fmt_p(p_value),
    p_adj_holm_label = fmt_p(p_adj_holm)
  ) %>%
  select(
    model, group_1, group_2, n_pairs, statistic,
    median_diff, mean_diff, rank_biserial,
    p_value, p_label, p_adj_holm, p_adj_holm_label, direction
  )

transfer_posthoc_appendix <- transfer_posthoc %>%
  separate(contrast, into = c("group_1", "group_2"), sep = " vs ", remove = FALSE) %>%
  mutate(
    median_diff = round(median_diff, 3),
    mean_diff = round(mean_diff, 3),
    statistic = round(statistic, 3),
    rank_biserial = round(rank_biserial, 3),
    p_label = fmt_p(p_value),
    p_adj_holm_label = fmt_p(p_adj_holm)
  ) %>%
  select(
    model, group_1, group_2, n_pairs, statistic,
    median_diff, mean_diff, rank_biserial,
    p_value, p_label, p_adj_holm, p_adj_holm_label, direction
  )

h2_posthoc_text_summary <- h2_posthoc_appendix %>%
  mutate(
    contrast = paste(group_1, "vs", group_2),
    result = paste0(
      contrast,
      ": median diff = ", fmt_num(median_diff),
      ", r_rb = ", fmt_num(rank_biserial),
      ", Holm p = ", p_adj_holm_label
    )
  ) %>%
  group_by(model) %>%
  summarise(
    posthoc_summary = paste(result, collapse = "; "),
    .groups = "drop"
  )

reporting_focus <- bind_rows(
  cross_friedman %>%
    transmute(
      component = paste0("cross_friedman_", model),
      detail = paste0(
        "n_blocks = ", n_blocks,
        "; chi_sq = ", fmt_num(statistic),
        "; Kendall_W = ", fmt_num(kendalls_w),
        "; p = ", fmt_p(p_value)
      )
    ),
  transfer_friedman %>%
    transmute(
      component = paste0("transfer_loss_friedman_", model),
      detail = paste0(
        "n_blocks = ", n_blocks,
        "; chi_sq = ", fmt_num(statistic),
        "; Kendall_W = ", fmt_num(kendalls_w),
        "; p = ", fmt_p(p_value)
      )
    )
)

print(h2_friedman_overview, n = Inf)
print(h2_table_paper, n = Inf)
print(h2_posthoc_appendix, n = Inf)

# ── Plot prep ─────────────────────────────────────────────────────────────────
print_section("GENERATING H2 PLOTS")

cross_heatmap_dt <- cross_dt_complete %>%
  mutate(dataset_label = factor(dataset_id, levels = rev(sort(unique(dataset_id)))))

cross_annot <- build_facet_p_labels(
  stats_df = cross_friedman,
  label_col = "p_value",
  position = p_value_position
)

plot_index <- tibble(
  file = character(),
  description = character()
)

for (plot_style in plot_styles) {
  style <- get_plot_style(plot_style)
  style_suffix <- ifelse(plot_style == "bw", "bw", "color")
  label_note <- ifelse(show_p_values, "with p-value label", "without p-value label")
  
  p_h2_main_boxplot <- ggplot(cross_dt_complete, aes(x = behavior, y = f1, fill = behavior)) +
    geom_boxplot(
      outlier.shape = NA,
      width = 0.6,
      colour = "black",
      linewidth = 0.6,
      alpha = 0.95
    ) +
    geom_jitter(
      shape = 21,
      size = 2.1,
      stroke = 0.35,
      width = 0.15,
      height = 0,
      colour = "black",
      alpha = style$point_alpha
    ) +
    facet_wrap(~ model, ncol = 2) +
    scale_fill_manual(values = style$behavior_fill) +
    scale_y_continuous(
      limits = c(0, 1.08),
      breaks = seq(0, 1, 0.2),
      expand = expansion(mult = c(0.02, 0))
    ) +
    labs(y = "Cross-dataset F1 Score") +
    theme_h2_clean(plot_style)
  p_h2_main_boxplot <- add_facet_p_labels(p_h2_main_boxplot, cross_annot, plot_style, show_p_values)
  
  p_h2_main_point <- ggplot(cross_summary_paper, aes(x = behavior, y = median_f1, fill = behavior)) +
    geom_linerange(aes(ymin = q1_f1, ymax = q3_f1), linewidth = 0.7, colour = "black") +
    geom_point(shape = 21, size = 3, colour = "black", stroke = 0.35) +
    facet_wrap(~ model, ncol = 2) +
    scale_fill_manual(values = style$behavior_fill) +
    scale_y_continuous(
      limits = c(0, 1.08),
      breaks = seq(0, 1, 0.2),
      expand = expansion(mult = c(0.02, 0))
    ) +
    labs(y = "Median F1 Score (IQR)") +
    theme_h2_clean(plot_style)
  p_h2_main_point <- add_facet_p_labels(p_h2_main_point, cross_annot, plot_style, show_p_values)
  
  p_h2_paired_lines <- ggplot(cross_dt_complete, aes(x = behavior, y = f1, group = dataset_id)) +
    geom_line(alpha = 0.45, linewidth = 0.45, colour = style$line_colour) +
    geom_point(size = 1.9, alpha = style$point_alpha, colour = style$point_colour) +
    facet_wrap(~ model, ncol = 2) +
    scale_y_continuous(
      limits = c(-0.08, 1),
      breaks = seq(0, 1, 0.2),
      expand = expansion(mult = c(0.02, 0.02))
    ) +
    labs(y = "Cross-dataset F1 Score") +
    theme_h2_clean(plot_style)
  p_h2_paired_lines <- add_facet_p_labels(p_h2_paired_lines, cross_annot, plot_style, show_p_values)
  
  p_h2_heatmap_appendix <- ggplot(cross_heatmap_dt, aes(x = behavior, y = dataset_label, fill = f1)) +
    geom_tile(colour = "white", linewidth = 0.4) +
    facet_wrap(~ model, ncol = 2, scales = "free_y") +
    scale_fill_gradient2(
      low = style$heatmap_low,
      mid = style$heatmap_mid,
      high = style$heatmap_high,
      midpoint = 0.5,
      limits = c(0, 1)
    ) +
    labs(y = "Dataset", fill = "F1 Score") +
    theme_h2_clean(plot_style) +
    theme(
      legend.position = "right",
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  plot_files <- c(
    paste0("01_h2_main_cross_f1_boxplot_points_", style_suffix, ".png"),
    paste0("02_h2_cross_f1_median_iqr_", style_suffix, ".png"),
    paste0("03_h2_cross_f1_paired_lines_", style_suffix, ".png"),
    paste0("04_h2_cross_f1_heatmap_appendix_", style_suffix, ".png")
  )
  
  save_plot(p_h2_main_boxplot,     plot_files[1])
  save_plot(p_h2_main_point,       plot_files[2])
  save_plot(p_h2_paired_lines,     plot_files[3])
  save_plot(p_h2_heatmap_appendix, plot_files[4], width_mm = 190, height_mm = 150)
  
  plot_index <- bind_rows(
    plot_index,
    tibble(
      file = plot_files,
      description = c(
        paste0("H2 main plot: combined cross-dataset boxplot with points by behaviour and model, ", label_note, " (", style_suffix, ")"),
        paste0("H2 support plot: combined cross-dataset median F1 with IQR by behaviour and model, ", label_note, " (", style_suffix, ")"),
        paste0("H2 support plot: combined cross-dataset paired lines by behaviour and model, ", label_note, " (", style_suffix, ")"),
        paste0("H2 appendix plot: combined cross-dataset heatmap by behaviour, dataset, and model (", style_suffix, ")")
      )
    )
  )
}

# ── Export CSV ────────────────────────────────────────────────────────────────
print_section("EXPORTING CSV")

csv_exports <- list(
  "00_h2_input_registry.csv" = behavior_paths,
  "01_h2_input_overview.csv" = input_overview,
  "02_h2_behavior_metrics_long.csv" = behavior_long,
  "03_h2_summary_by_model_comparison_behavior.csv" = summary_behavior,
  "04_h2_cross_complete_blocks.csv" = cross_complete_blocks,
  "05_h2_cross_behavior_effect_friedman.csv" = cross_friedman,
  "06_h2_cross_behavior_effect_posthoc_raw.csv" = cross_posthoc,
  "07_h2_cross_behavior_effect_posthoc_appendix.csv" = h2_posthoc_appendix,
  "08_h2_cross_summary_median_iqr.csv" = cross_summary_paper,
  "09_h2_cross_main_table.csv" = h2_table_paper,
  "10_h2_transfer_long.csv" = transfer_dt,
  "11_h2_transfer_summary_by_behavior.csv" = transfer_summary,
  "12_h2_transfer_loss_friedman.csv" = transfer_friedman,
  "13_h2_transfer_loss_posthoc_raw.csv" = transfer_posthoc,
  "14_h2_transfer_loss_posthoc_appendix.csv" = transfer_posthoc_appendix,
  "15_h2_plot_index.csv" = plot_index,
  "16_h2_reporting_focus.csv" = reporting_focus,
  "17_h2_posthoc_text_summary.csv" = h2_posthoc_text_summary
)

walk2(
  .x = csv_exports,
  .y = names(csv_exports),
  .f = ~ write_csv(.x, file.path(csv_dir, .y))
)

# ── Export TXT ────────────────────────────────────────────────────────────────
print_section("EXPORTING TXT")

report_lines <- c(
  paste0("H2 output folder: ", out_dir),
  paste0("Created on: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  paste0("show_p_values: ", show_p_values),
  paste0("p_value_position: ", p_value_position),
  "",
  "This report contains the main H2 results, checks, and exported file overview.",
  "Saved H2 plots: main boxplot-with-points, median/IQR support plot, paired-lines support plot, and appendix heatmap (each in bw and color).",
  ""
)

report_lines <- append_section(report_lines, "INPUT OVERVIEW", input_overview)
report_lines <- append_section(report_lines, "SUMMARY BY MODEL, COMPARISON, AND BEHAVIOUR", summary_behavior)
report_lines <- append_section(report_lines, "CROSS COMPLETE BLOCKS", cross_complete_blocks)
report_lines <- append_section(report_lines, "CROSS FRIEDMAN OVERVIEW", h2_friedman_overview)
report_lines <- append_section(report_lines, "CROSS MAIN TABLE", h2_table_paper)
report_lines <- append_section(report_lines, "CROSS POSTHOC APPENDIX", h2_posthoc_appendix)
report_lines <- append_section(report_lines, "CROSS POSTHOC TEXT SUMMARY", h2_posthoc_text_summary)
report_lines <- append_section(report_lines, "TRANSFER SUMMARY", transfer_summary)
report_lines <- append_section(report_lines, "TRANSFER FRIEDMAN", transfer_friedman)
report_lines <- append_section(report_lines, "TRANSFER POSTHOC APPENDIX", transfer_posthoc_appendix)
report_lines <- append_section(report_lines, "REPORTING FOCUS", reporting_focus)
report_lines <- append_section(report_lines, "PLOT INDEX", plot_index)

write_txt_report(file.path(txt_dir, "01_h2_analysis_report.txt"), report_lines)

files_lines <- c(
  paste0("Output folder: ", out_dir),
  paste0("CSV folder: ", csv_dir),
  paste0("Plots folder: ", plots_dir),
  paste0("TXT folder: ", txt_dir),
  "",
  "Written plot files:",
  paste0("- ", plot_index$file),
  "",
  "Written CSV files:",
  paste0("- ", names(csv_exports))
)

write_txt_report(file.path(txt_dir, "02_h2_files_written.txt"), files_lines)

# ── Console output ────────────────────────────────────────────────────────────
print_section("RESULTS")
print(summary_behavior)
print(h2_friedman_overview, n = Inf)
print(h2_table_paper, n = Inf)
print(h2_posthoc_appendix, n = Inf)
print(transfer_summary)
print(transfer_posthoc_appendix, n = Inf)
print(reporting_focus)
print(plot_index)

print_section("FILES WRITTEN")
cat("\nCSV folder:", csv_dir, "\n")
cat("Plots folder:", plots_dir, "\n")
cat("TXT folder:", txt_dir, "\n")
