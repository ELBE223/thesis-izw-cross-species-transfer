# =============================================================================
# H3 forest plot: distance effects on cross-species transfer
# =============================================================================
# Author : Lucas Beseler
# Date   : 2026-06-17
#
# Purpose:
# - Combine the final H3 core/extended trait-distance results with
#   EltonTraits and TimeTree sensitivity results.
# - Use the final H3 functional-biomechanical trait framework:
#   7-trait core distance plus extended sensitivity traits.
# - Draw a thesis-ready forest plot with 95% confidence intervals.
# - Export plot data, an output index, and a short text report.
#
# Run after:
# - H3.R / H3_final_traits.R
# - H3_timetree_elton_sensitivity.R
#
# Expected primary H3 input:
# - Output_R/H3/csv/13_core_vs_extended_sensitivity.csv
#   containing core and extended results from species_traits_core_h3_final.csv and
#   species_traits_extended_sensitivity_full_final.csv.
#
# Output:
# - Output_R/H3_distance_effect_forest_plot/
# =============================================================================

# -- 1. Setup ------------------------------------------------------------------
library(readr)
library(dplyr)
library(tibble)
library(ggplot2)

# -- 2. User settings ----------------------------------------------------------
base_dir <- "/Volumes/Z Slim/11_05_2026_Data_Analysis"

alpha <- 0.05
sig_by <- "p_value"          # "p_value" or "ci_excludes_zero"
show_value_labels <- FALSE   # TRUE adds rho and p labels on the right

plot_width_mm <- 230
plot_height_mm <- 130
plot_dpi <- 600

# -- 3. Paths ------------------------------------------------------------------
output_r_root <- file.path(base_dir, "Output_R")

h3_csv_dir <- file.path(output_r_root, "H3", "csv")
sens_csv_dir <- file.path(output_r_root, "H3_timetree_elton_sensitivity", "csv")

h3_file_candidates <- c(
  file.path(h3_csv_dir, "13_core_vs_extended_sensitivity.csv"),
  file.path(h3_csv_dir, "13_core_vs_extended_sensitivity_final_traits.csv"),
  file.path(h3_csv_dir, "core_vs_extended_sensitivity.csv")
)

sens_file_candidates <- c(
  file.path(sens_csv_dir, "13_h3_primary_sensitivity_summary.csv"),
  file.path(sens_csv_dir, "h3_primary_sensitivity_summary.csv")
)

out_dir <- file.path(output_r_root, "H3_distance_effect_forest_plot")
csv_dir <- file.path(out_dir, "csv")
plot_dir <- file.path(out_dir, "plots")
txt_dir <- file.path(out_dir, "txt")

for (d in c(out_dir, csv_dir, plot_dir, txt_dir)) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

# -- 4. Constants --------------------------------------------------------------
required_cols <- c(
  "distance_version", "model", "level", "scope",
  "n", "rho", "p_value", "ci_low", "ci_high"
)

h3_distance_versions <- c("core", "extended")
sensitivity_distance_versions <- c(
  "elton_minimal_gower",
  "elton_extended_gower",
  "timetree_divergence_mya"
)

plot_order <- c(h3_distance_versions, sensitivity_distance_versions)

label_map <- c(
  "core" = "Functional-biomechanical traits (core)",
  "extended" = "Functional-biomechanical traits (extended)",
  "elton_minimal_gower" = "EltonTraits (core)",
  "elton_extended_gower" = "EltonTraits (extended)",
  "timetree_divergence_mya" = "Phylogenetic distance"
)

# -- 5. Helpers ----------------------------------------------------------------
pick_existing_file <- function(candidates, label) {
  hit <- candidates[file.exists(candidates)][1]
  if (length(hit) == 0 || is.na(hit)) {
    stop(
      "No valid ", label, " file found. Checked:\n",
      paste(candidates, collapse = "\n")
    )
  }
  hit
}

check_cols <- function(df, file) {
  missing_cols <- setdiff(required_cols, names(df))
  if (length(missing_cols) > 0) {
    stop(
      basename(file), " is missing columns: ",
      paste(missing_cols, collapse = ", ")
    )
  }
}

format_p <- function(x) {
  ifelse(
    is.na(x),
    "p = NA",
    ifelse(x < 0.001, "p < 0.001", paste0("p = ", sprintf("%.3f", x)))
  )
}

format_rho <- function(x) {
  ifelse(is.na(x), "rho = NA", paste0("rho = ", sprintf("%.3f", x)))
}

pick_primary <- function(file, distance_versions, source_block) {
  dt <- read_csv(file, show_col_types = FALSE)
  check_cols(dt, file)
  
  out <- dt %>%
    mutate(
      model = as.character(model),
      distance_version = as.character(distance_version),
      level = as.character(level),
      scope = as.character(scope)
    ) %>%
    filter(
      distance_version %in% distance_versions,
      model == "overall",
      level == "species_pair",
      scope == "different_species_only"
    ) %>%
    select(distance_version, model, level, scope, n, rho, p_value, ci_low, ci_high) %>%
    mutate(source_block = source_block, source_file = file)
  
  missing_versions <- setdiff(distance_versions, unique(out$distance_version))
  if (length(missing_versions) > 0) {
    available <- dt %>%
      distinct(distance_version, model, level, scope) %>%
      arrange(distance_version, model, level, scope)
    
    stop(
      "Missing primary rows in ", basename(file), ": ",
      paste(missing_versions, collapse = ", "),
      "\nExpected rows with model='overall', level='species_pair', ",
      "scope='different_species_only'.\nAvailable combinations:\n",
      paste(capture.output(print(available, n = Inf)), collapse = "\n")
    )
  }
  
  out
}

write_lines <- function(lines, path) {
  writeLines(enc2utf8(lines), con = path, useBytes = TRUE)
}

# -- 6. Load results -----------------------------------------------------------
h3_file <- pick_existing_file(h3_file_candidates, "H3 core/extended")
sens_file <- pick_existing_file(sens_file_candidates, "TimeTree/Elton sensitivity")

h3_results <- pick_primary(
  file = h3_file,
  distance_versions = h3_distance_versions,
  source_block = "final_h3_core_extended_traits"
)

sens_results <- pick_primary(
  file = sens_file,
  distance_versions = sensitivity_distance_versions,
  source_block = "timetree_elton_sensitivity"
)

plot_dt <- bind_rows(h3_results, sens_results) %>%
  mutate(
    distance_version = factor(distance_version, levels = plot_order),
    variable = recode(as.character(distance_version), !!!label_map),
    variable = factor(variable, levels = rev(unname(label_map[plot_order]))),
    ci_excludes_zero = is.finite(ci_low) & is.finite(ci_high) &
      (ci_low > 0 | ci_high < 0),
    significant = if (sig_by == "p_value") {
      is.finite(p_value) & p_value < alpha
    } else if (sig_by == "ci_excludes_zero") {
      ci_excludes_zero
    } else {
      stop("sig_by must be 'p_value' or 'ci_excludes_zero'.")
    },
    stat_label = paste0(format_rho(rho), ", ", format_p(p_value), ", n = ", n),
    trait_framework_note = case_when(
      as.character(distance_version) %in% c("core", "extended") ~
        "Final H3 traits: 7-trait core distance; extended distance adds four sensitivity traits.",
      TRUE ~ "External biological distance proxy used for sensitivity comparison."
    )
  ) %>%
  arrange(distance_version)

if (nrow(plot_dt) != length(plot_order)) {
  stop(
    "Expected ", length(plot_order), " rows for the forest plot, but found ", nrow(plot_dt), ".\n",
    "Check whether H3 and H3_timetree_elton_sensitivity ran successfully."
  )
}

forest_data_file <- file.path(csv_dir, "01_h3_distance_effect_forest_plot_data.csv")
write_csv(plot_dt, forest_data_file)

input_index <- tibble(
  input = c("H3 core/extended", "TimeTree/Elton sensitivity"),
  file = c(h3_file, sens_file),
  distance_versions = c(
    paste(h3_distance_versions, collapse = ", "),
    paste(sensitivity_distance_versions, collapse = ", ")
  )
)
write_csv(input_index, file.path(csv_dir, "00_input_file_index.csv"))

# -- 7. Plot limits ------------------------------------------------------------
x_min <- min(c(plot_dt$ci_low, 0), na.rm = TRUE) - 0.05
x_max <- max(c(plot_dt$ci_high, 0), na.rm = TRUE) + 0.05

x_min <- max(-1, x_min)
x_max <- min(1, x_max)

label_x <- x_max + 0.03
x_plot_max <- if (show_value_labels) x_max + 0.42 else x_max

# -- 8. Forest plot ------------------------------------------------------------
p <- ggplot(plot_dt, aes(x = rho, y = variable)) +
  geom_vline(xintercept = 0, colour = "black", linewidth = 0.7) +
  geom_vline(
    xintercept = c(-0.10, 0.10),
    colour = "grey65",
    linetype = "dashed",
    linewidth = 0.45
  ) +
  geom_vline(
    xintercept = c(-0.20, 0.20),
    colour = "grey75",
    linetype = "dotted",
    linewidth = 0.45
  ) +
  geom_segment(
    aes(x = ci_low, xend = ci_high, y = variable, yend = variable),
    linewidth = 0.9,
    colour = "black"
  ) +
  geom_errorbarh(
    aes(xmin = ci_low, xmax = ci_high),
    height = 0.16,
    linewidth = 0.9,
    colour = "black"
  ) +
  geom_point(
    aes(fill = significant),
    shape = 21,
    size = 4.8,
    colour = "black",
    stroke = 0.35
  ) +
  scale_fill_manual(
    values = c("TRUE" = "#D95A4A", "FALSE" = "grey55"),
    breaks = c(TRUE, FALSE),
    labels = c("p < 0.05", "n.s."),
    name = NULL
  ) +
  coord_cartesian(xlim = c(x_min, x_plot_max), clip = "off") +
  labs(
    x = "Spearman’s ρ: distance vs. cross-species Macro-F1",
    y = NULL
  ) +
  theme_classic(base_size = 12, base_family = "sans") +
  theme(
    axis.title.x = element_text(size = 13, margin = margin(t = 8)),
    axis.text.x = element_text(size = 11, colour = "black"),
    axis.text.y = element_text(size = 12, colour = "black"),
    axis.line.y = element_line(colour = "black", linewidth = 0.7),
    axis.line.x = element_line(colour = "black", linewidth = 0.7),
    axis.ticks.y = element_line(colour = "black"),
    legend.position = "bottom",
    legend.text = element_text(size = 10),
    plot.margin = margin(10, 35, 14, 10)
  )

if (show_value_labels) {
  p <- p +
    geom_text(
      aes(x = label_x, label = stat_label),
      hjust = 0,
      size = 3.2,
      colour = "black"
    ) +
    theme(plot.margin = margin(8, 125, 8, 8))
}

# -- 9. Save outputs -----------------------------------------------------------
png_file <- file.path(plot_dir, "01_h3_distance_effect_forest_plot.png")
pdf_file <- file.path(plot_dir, "01_h3_distance_effect_forest_plot.pdf")

ggsave(
  filename = png_file,
  plot = p,
  width = plot_width_mm,
  height = plot_height_mm,
  units = "mm",
  dpi = plot_dpi,
  bg = "white"
)

ggsave(
  filename = pdf_file,
  plot = p,
  width = plot_width_mm,
  height = plot_height_mm,
  units = "mm",
  bg = "white"
)

output_index <- tibble(
  item = c("plot_data", "png_plot", "pdf_plot", "input_index", "text_report"),
  file = c(
    forest_data_file,
    png_file,
    pdf_file,
    file.path(csv_dir, "00_input_file_index.csv"),
    file.path(txt_dir, "01_h3_distance_effect_forest_plot_report.txt")
  ),
  description = c(
    "Forest plot input data with rho, p-value and confidence intervals.",
    "High-resolution PNG forest plot.",
    "Vector PDF forest plot.",
    "Input files used to build the plot.",
    "Short text report for checking the exported values."
  )
)
write_csv(output_index, file.path(csv_dir, "02_output_index.csv"))

report_lines <- c(
  "# ===================================================",
  "# H3 distance-effect forest plot report",
  "# ===================================================",
  "",
  paste0("Date: ", Sys.time()),
  paste0("Base dir: ", base_dir),
  paste0("H3 input: ", h3_file),
  paste0("Sensitivity input: ", sens_file),
  "",
  "Functional-biomechanical core/extended rows use the final H3 trait framework:",
  "core = 7 functional-biomechanical traits; extended = core plus activity_pattern, vertical_use_motion_scope, postural_compactness_body_profile_reduction, and defensive_posture_specialization.",
  "",
  "Forest plot rows:",
  paste0(
    "- ", as.character(plot_dt$variable),
    ": rho = ", sprintf("%.3f", plot_dt$rho),
    ", 95% CI [", sprintf("%.3f", plot_dt$ci_low), ", ", sprintf("%.3f", plot_dt$ci_high), "]",
    ", p = ", ifelse(plot_dt$p_value < 0.001, "< 0.001", sprintf("%.3f", plot_dt$p_value)),
    ", n = ", plot_dt$n
  ),
  "",
  paste0("Significance rule: ", sig_by, ", alpha = ", alpha),
  paste0("PNG: ", png_file),
  paste0("PDF: ", pdf_file),
  paste0("CSV: ", forest_data_file)
)

write_lines(report_lines, file.path(txt_dir, "01_h3_distance_effect_forest_plot_report.txt"))

cat("\nForest plot saved:\n", png_file, "\n", sep = "")
cat("PDF saved:\n", pdf_file, "\n", sep = "")
cat("Plot data saved:\n", forest_data_file, "\n", sep = "")
