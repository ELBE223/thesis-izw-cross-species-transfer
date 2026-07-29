# =============================================================================
# 01_H3_mixed_effects_primary_final.R
# =============================================================================
# Author : Lucas Beseler
#
# Purpose:
# - Reconstruct 432 H3 observations from 72 directed species pairs and six
#   classifiers.
# - Fit the predefined dyadic mixed-effects model for primary inference.
# - Retain Spearman correlations as descriptive secondary analyses.
# - Test classifier-specific slope heterogeneity with an interaction model.
#
# Random-effects structures:
# - Primary: unordered pair + directed pair + source species + target species.
# - Directional sensitivity: directed pair + source species + target species.
# - Pair-only sensitivity: directed pair.
#
# The additive dyadic model remains primary regardless of the interaction test.

# =============================================================================


# ── 1. Setup ──────────────────────────────────────────────────────────────────

required_packages <- c(
  "readr",
  "dplyr",
  "tidyr",
  "tibble",
  "purrr",
  "stringr",
  "cluster",
  "ggplot2",
  "lme4",
  "lmerTest",
  "performance",
  "emmeans"
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

model_order <- c(
  "CNN",
  "ResNet",
  "HYDRA",
  "MultiRocket",
  "RF",
  "LGBM"
)

# Color-blind-safe Okabe-Ito palette, fixed by classifier.
model_colors <- c(
  CNN         = "#0072B2",
  ResNet      = "#D55E00",
  HYDRA       = "#009E73",
  MultiRocket = "#CC79A7",
  RF          = "#E69F00",
  LGBM        = "#56B4E9"
)

core_trait_filename <- "species_traits_core_h3_final.csv"

confidence_level <- 0.95
singularity_tolerance <- 1e-4

plot_dpi <- 600
plot_width <- 10
plot_height <- 6

axis_label_distance <- "Core functional-biomechanical Gower distance"
axis_label_observed <- "Mean cross-species Macro-F1"
axis_label_predicted <- "Predicted cross-species Macro-F1"


# ── 3. Paths and output structure ─────────────────────────────────────────────

models_dir <- file.path(base_dir, "Models")
output_r_root <- file.path(base_dir, "Output_R")

out_dir <- file.path(
  output_r_root,
  "H3_mixed_effects_primary"
)

csv_dir <- file.path(out_dir, "csv")
plots_dir <- file.path(out_dir, "plots")
txt_dir <- file.path(out_dir, "txt")

for (dir_path in c(
  output_r_root,
  out_dir,
  csv_dir,
  plots_dir,
  txt_dir
)) {
  dir.create(
    dir_path,
    recursive = TRUE,
    showWarnings = FALSE
  )
}


# ── 4. Generic helpers ────────────────────────────────────────────────────────

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

must_exist <- function(paths) {
  missing <- paths[!file.exists(paths)]

  if (length(missing) > 0L) {
    stop(
      "Missing required files or folders:\n",
      paste(missing, collapse = "\n"),
      call. = FALSE
    )
  }
}

resolve_first_existing <- function(paths) {
  hits <- paths[file.exists(paths)]

  if (length(hits) == 0L) {
    return(NA_character_)
  }

  hits[[1]]
}

resolve_metric_file <- function(dir_path, candidate_names) {
  resolve_first_existing(
    file.path(dir_path, candidate_names)
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

fmt_ci <- function(low, high, digits = 3) {
  paste0(
    "[",
    fmt_num(low, digits),
    ", ",
    fmt_num(high, digits),
    "]"
  )
}

write_txt <- function(path, lines) {
  writeLines(
    enc2utf8(lines),
    con = path,
    useBytes = TRUE
  )
}

# Convert to tibble first: base R print.data.frame() partially matches `n`
# to its `na.print` argument.
print_table <- function(x, width = Inf) {
  print(
    tibble::as_tibble(x),
    n = Inf,
    width = width
  )
  invisible(x)
}


# ── 5. Plot helpers ───────────────────────────────────────────────────────────

# Minimal publication theme: no plot titles, captions live in the manuscript.
theme_h3 <- function(base_size = 11) {
  theme_bw(
    base_size = base_size,
    base_family = "sans"
  ) +
    theme(
      panel.border = element_rect(
        colour = "grey20",
        fill = NA,
        linewidth = 0.4
      ),
      panel.grid.major = element_line(
        colour = "grey92",
        linewidth = 0.3
      ),
      panel.grid.minor = element_blank(),
      axis.text = element_text(colour = "black"),
      axis.title = element_text(colour = "black"),
      axis.ticks = element_line(
        colour = "grey20",
        linewidth = 0.3
      ),
      strip.background = element_rect(
        fill = "grey96",
        colour = "grey20",
        linewidth = 0.4
      ),
      strip.text = element_text(
        face = "bold",
        colour = "black"
      ),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.key = element_blank(),
      legend.background = element_blank(),
      plot.margin = margin(6, 8, 6, 8)
    )
}

# Save one high-resolution PNG file.
save_plot_png <- function(
    plot_object,
    file_stub,
    width = plot_width,
    height = plot_height
) {
  ggsave(
    filename = file.path(
      plots_dir,
      paste0(file_stub, ".png")
    ),
    plot = plot_object,
    width = width,
    height = height,
    units = "in",
    dpi = plot_dpi,
    bg = "white"
  )
}


# ── 6. Statistical helpers ────────────────────────────────────────────────────

safe_spearman <- function(
    x,
    y,
    conf_level = confidence_level
) {
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]
  y <- y[ok]

  n_obs <- length(x)

  if (n_obs < 4L) {
    return(
      tibble(
        n = n_obs,
        rho = NA_real_,
        p_value_naive = NA_real_,
        ci_low_naive = NA_real_,
        ci_high_naive = NA_real_
      )
    )
  }

  test <- suppressWarnings(
    cor.test(
      x,
      y,
      method = "spearman",
      exact = FALSE
    )
  )

  rho <- unname(test$estimate)

  # Fisher z back-transformation for an approximate interval.
  if (is.finite(rho) && abs(rho) < 1 && n_obs > 3L) {
    z <- atanh(
      pmin(pmax(rho, -0.999999), 0.999999)
    )

    se_z <- 1 / sqrt(n_obs - 3)
    z_crit <- qnorm((1 + conf_level) / 2)

    ci_low <- tanh(z - z_crit * se_z)
    ci_high <- tanh(z + z_crit * se_z)
  } else {
    ci_low <- NA_real_
    ci_high <- NA_real_
  }

  tibble(
    n = n_obs,
    rho = rho,
    p_value_naive = test$p.value,
    ci_low_naive = ci_low,
    ci_high_naive = ci_high
  )
}

# Fit an lmer model and capture errors and warnings instead of aborting.
safe_lmer <- function(
    formula,
    data,
    reml,
    label
) {
  warning_messages <- character()

  fit <- withCallingHandlers(
    tryCatch(
      lmerTest::lmer(
        formula = formula,
        data = data,
        REML = reml,
        control = lmerControl(
          optimizer = "bobyqa",
          optCtrl = list(maxfun = 200000),
          check.conv.singular = "ignore"
        )
      ),
      error = function(error_condition) {
        structure(
          list(
            message = conditionMessage(error_condition)
          ),
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

extract_fixed_effects <- function(
    fit,
    model_name,
    conf_level = confidence_level
) {
  if (is.null(fit)) {
    return(
      tibble(
        model_name = model_name,
        term = NA_character_,
        estimate = NA_real_,
        std_error = NA_real_,
        df = NA_real_,
        statistic = NA_real_,
        p_value = NA_real_,
        ci_low = NA_real_,
        ci_high = NA_real_
      )
    )
  }

  coefficients <- as.data.frame(
    coef(summary(fit))
  ) %>%
    rownames_to_column("term") %>%
    as_tibble()

  estimate_col <- intersect("Estimate", names(coefficients))[[1]]
  se_col <- intersect("Std. Error", names(coefficients))[[1]]
  df_col <- intersect("df", names(coefficients))
  statistic_col <- intersect("t value", names(coefficients))[[1]]
  p_col <- intersect("Pr(>|t|)", names(coefficients))

  coefficients %>%
    transmute(
      model_name = model_name,
      term = term,
      estimate = .data[[estimate_col]],
      std_error = .data[[se_col]],
      df = if (length(df_col) > 0L) {
        .data[[df_col[[1]]]]
      } else {
        NA_real_
      },
      statistic = .data[[statistic_col]],
      p_value = if (length(p_col) > 0L) {
        .data[[p_col[[1]]]]
      } else {
        NA_real_
      }
    ) %>%
    mutate(
      critical_value = if_else(
        is.finite(df),
        qt((1 + conf_level) / 2, df = df),
        qnorm((1 + conf_level) / 2)
      ),
      ci_low = estimate - critical_value * std_error,
      ci_high = estimate + critical_value * std_error
    ) %>%
    select(-critical_value) %>%
    as_tibble()
}

extract_random_effects <- function(fit, model_name) {
  if (is.null(fit)) {
    return(
      tibble(
        model_name = model_name,
        group = NA_character_,
        term = NA_character_,
        variance = NA_real_,
        std_dev = NA_real_
      )
    )
  }

  as.data.frame(VarCorr(fit)) %>%
    transmute(
      model_name = model_name,
      group = grp,
      term = var1,
      variance = vcov,
      std_dev = sdcor
    ) %>%
    as_tibble()
}

extract_model_diagnostics <- function(
    model_object,
    singular_tolerance = singularity_tolerance
) {
  fit <- model_object$fit

  if (is.null(fit)) {
    return(
      tibble(
        model_name = model_object$label,
        reml = NA,
        n_observations = NA_integer_,
        n_groups_unordered_pair = NA_integer_,
        n_groups_ordered_pair = NA_integer_,
        n_groups_train_species = NA_integer_,
        n_groups_test_species = NA_integer_,
        singular = NA,
        optimizer_convergence_code = NA_integer_,
        convergence_messages = NA_character_,
        warnings = paste(model_object$warnings, collapse = " | "),
        error = model_object$error,
        log_likelihood = NA_real_,
        aic = NA_real_,
        bic = NA_real_
      )
    )
  }

  group_counts <- lme4::ngrps(fit)

  count_groups <- function(group_name) {
    if (group_name %in% names(group_counts)) {
      unname(group_counts[group_name])
    } else {
      NA_integer_
    }
  }

  tibble(
    model_name = model_object$label,
    reml = isREML(fit),
    n_observations = nobs(fit),
    n_groups_unordered_pair = count_groups("unordered_species_pair"),
    n_groups_ordered_pair = count_groups("ordered_species_pair"),
    n_groups_train_species = count_groups("train_species"),
    n_groups_test_species = count_groups("test_species"),
    singular = isSingular(fit, tol = singular_tolerance),
    optimizer_convergence_code = get_optimizer_convergence_code(fit),
    convergence_messages = get_convergence_messages(fit),
    warnings = paste(model_object$warnings, collapse = " | "),
    error = model_object$error,
    log_likelihood = as.numeric(logLik(fit)),
    aic = AIC(fit),
    bic = BIC(fit)
  )
}

extract_r2 <- function(fit, model_name) {
  empty_r2 <- tibble(
    model_name = model_name,
    marginal_r2 = NA_real_,
    conditional_r2 = NA_real_
  )

  if (is.null(fit)) {
    return(empty_r2)
  }

  r2_result <- tryCatch(
    performance::r2_nakagawa(fit, tolerance = 1e-8),
    error = function(error_condition) NULL
  )

  if (is.null(r2_result)) {
    return(empty_r2)
  }

  tibble(
    model_name = model_name,
    marginal_r2 = as.numeric(r2_result$R2_marginal),
    conditional_r2 = as.numeric(r2_result$R2_conditional)
  )
}

compare_nested_models <- function(
    reduced_fit,
    full_fit,
    comparison_name
) {
  if (is.null(reduced_fit) || is.null(full_fit)) {
    return(
      tibble(
        comparison = comparison_name,
        df_difference = NA_real_,
        likelihood_ratio_chisq = NA_real_,
        p_value = NA_real_,
        aic_reduced = NA_real_,
        aic_full = NA_real_
      )
    )
  }

  # Keep a distinct object name: a column called "comparison" is created below
  # and tibble() evaluates columns sequentially.
  anova_table <- suppressMessages(
    stats::anova(reduced_fit, full_fit)
  ) %>%
    as.data.frame()

  required_anova_columns <- c("Df", "Chisq", "Pr(>Chisq)")

  missing_anova_columns <- setdiff(
    required_anova_columns,
    names(anova_table)
  )

  if (length(missing_anova_columns) > 0L) {
    stop(
      "Nested-model comparison table is missing: ",
      paste(missing_anova_columns, collapse = ", "),
      call. = FALSE
    )
  }

  comparison_row <- nrow(anova_table)

  tibble(
    comparison = comparison_name,
    df_difference = as.numeric(anova_table[comparison_row, "Df"]),
    likelihood_ratio_chisq = as.numeric(anova_table[comparison_row, "Chisq"]),
    p_value = as.numeric(anova_table[comparison_row, "Pr(>Chisq)"]),
    aic_reduced = AIC(reduced_fit),
    aic_full = AIC(full_fit)
  )
}

# Fixed-effect predictions with random effects set to zero.
make_fixed_prediction_grid <- function(
    fit,
    data,
    n_points = 100L
) {
  if (is.null(fit)) {
    return(tibble())
  }

  distance_mean <- mean(data$trait_distance_core, na.rm = TRUE)
  distance_sd <- sd(data$trait_distance_core, na.rm = TRUE)

  grid <- expand_grid(
    trait_distance_core = seq(
      min(data$trait_distance_core, na.rm = TRUE),
      max(data$trait_distance_core, na.rm = TRUE),
      length.out = n_points
    ),
    model = factor(
      levels(data$model),
      levels = levels(data$model)
    )
  ) %>%
    mutate(
      z_trait_distance_core =
        (trait_distance_core - distance_mean) / distance_sd
    )

  fixed_formula <- lme4::nobars(formula(fit))

  design_matrix <- model.matrix(
    delete.response(terms(fixed_formula)),
    data = grid
  )

  beta <- fixef(fit)
  variance_covariance <- as.matrix(vcov(fit))

  estimate <- as.numeric(design_matrix %*% beta)

  standard_error <- sqrt(
    diag(
      design_matrix %*% variance_covariance %*% t(design_matrix)
    )
  )

  critical_value <- qnorm((1 + confidence_level) / 2)

  grid %>%
    mutate(
      predicted_macro_f1 = estimate,
      std_error = standard_error,
      ci_low = predicted_macro_f1 - critical_value * std_error,
      ci_high = predicted_macro_f1 + critical_value * std_error
    )
}


# Average fixed-effect predictions equally across classifiers.
# Compute uncertainty from the averaged design vector.
make_marginal_prediction_grid <- function(
    fit,
    data,
    n_points = 100L
) {
  if (is.null(fit)) {
    return(tibble())
  }

  distance_mean <- mean(data$trait_distance_core, na.rm = TRUE)
  distance_sd <- sd(data$trait_distance_core, na.rm = TRUE)

  distance_values <- seq(
    min(data$trait_distance_core, na.rm = TRUE),
    max(data$trait_distance_core, na.rm = TRUE),
    length.out = n_points
  )

  fixed_formula <- lme4::nobars(formula(fit))
  beta <- fixef(fit)
  variance_covariance <- as.matrix(vcov(fit))
  critical_value <- qnorm((1 + confidence_level) / 2)

  map_dfr(
    distance_values,
    function(distance_value) {
      grid_one_distance <- tibble(
        trait_distance_core = distance_value,
        z_trait_distance_core =
          (distance_value - distance_mean) / distance_sd,
        model = factor(
          levels(data$model),
          levels = levels(data$model)
        )
      )

      design_matrix <- model.matrix(
        delete.response(terms(fixed_formula)),
        data = grid_one_distance
      )

      average_design <- colMeans(design_matrix)

      estimate <- as.numeric(
        average_design %*% beta
      )

      standard_error <- sqrt(
        as.numeric(
          t(average_design) %*%
            variance_covariance %*%
            average_design
        )
      )

      tibble(
        trait_distance_core = distance_value,
        predicted_macro_f1 = estimate,
        std_error = standard_error,
        ci_low = estimate - critical_value * standard_error,
        ci_high = estimate + critical_value * standard_error
      )
    }
  )
}


# ── 7. Input registry ─────────────────────────────────────────────────────────

print_section("1. INPUT REGISTRY")

must_exist(c(base_dir, models_dir))

trait_dir_candidates <- unique(
  c(
    getwd(),
    file.path(base_dir, "data", "traits"),
    file.path(base_dir, "R_analysis", "H3"),
    file.path(base_dir, "R_scripts", "H3")
  )
)

core_trait_file <- resolve_first_existing(
  file.path(trait_dir_candidates, core_trait_filename)
)

if (is.na(core_trait_file)) {
  stop(
    "Core trait file not found: ",
    core_trait_filename,
    call. = FALSE
  )
}

model_registry <- tibble(
  model = model_order,
  pairwise_dir = file.path(
    models_dir,
    model_order,
    paste0("Pairwise_", model_order),
    "statistics"
  )
) %>%
  mutate(
    pairwise_file = map_chr(
      pairwise_dir,
      resolve_metric_file,
      candidate_names = c(
        "metrics_all.csv",
        "pairwise_metrics_all.csv",
        "pairwise_summary_metrics.csv"
      )
    ),
    pairwise_exists = file.exists(pairwise_file)
  )

write_csv(
  model_registry,
  file.path(csv_dir, "00_model_file_registry.csv")
)

if (!all(model_registry$pairwise_exists)) {
  stop(
    "Missing pairwise metric files for: ",
    paste(
      model_registry$model[!model_registry$pairwise_exists],
      collapse = ", "
    ),
    call. = FALSE
  )
}

cat("Core traits:", core_trait_file, "\n")
cat("Output:", out_dir, "\n")


# ── 8. Load pairwise metrics ──────────────────────────────────────────────────

print_section("2. LOAD PAIRWISE METRICS")

required_pairwise_columns <- c(
  "pair_id",
  "train_dataset",
  "test_dataset",
  "accuracy",
  "macro_recall",
  "macro_precision",
  "macro_f1"
)

read_pairwise_model <- function(model, pairwise_file) {
  data <- read_csv(pairwise_file, show_col_types = FALSE)

  missing_columns <- setdiff(
    required_pairwise_columns,
    names(data)
  )

  if (length(missing_columns) > 0L) {
    stop(
      model,
      " pairwise file is missing: ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }

  data %>%
    select(all_of(required_pairwise_columns)) %>%
    mutate(model = model, .before = 1)
}

pairwise_raw <- pmap_dfr(
  model_registry %>%
    select(model, pairwise_file),
  read_pairwise_model
)

cat("Raw pairwise rows:", nrow(pairwise_raw), "\n")


# ── 9. Dataset-to-species mapping ─────────────────────────────────────────────

print_section("3. DATASET-TO-SPECIES MAPPING")

# Species labels must match the trait table exactly.
dataset_to_species <- c(
  "Bison" = "Bison bonasus",
  "Cattle" = "Bos taurus",
  "Dog" = "Canis lupus familiaris",
  "Giraffe" = "Giraffa camelopardalis",
  "Hedgehog" = "Erinaceus europaeus",

  "Bison_dataset_1" = "Bison bonasus",
  "Cattle_dataset_1" = "Bos taurus",
  "Dog_dataset_1" = "Canis lupus familiaris",
  "Dog_dataset_2" = "Canis lupus familiaris",
  "Giraffe_dataset_1" = "Giraffa camelopardalis",
  "Hedgehog_dataset_1" = "Erinaceus europaeus",

  "Fox_dataset_1" = "Vulpes vulpes",
  "Fox_dataset_2" = "Vulpes vulpes",
  "Horse_dataset_1" = "Equus ferus przewalskii",
  "Horse_dataset_2" = "Equus caballus",
  "Raccoon_dataset_1" = "Procyon lotor",
  "Raccoon_dataset_2" = "Procyon lotor"
)

pairwise_mapped <- pairwise_raw %>%
  mutate(
    train_dataset = str_trim(train_dataset),
    test_dataset = str_trim(test_dataset),
    train_species = recode(
      train_dataset,
      !!!dataset_to_species,
      .default = NA_character_
    ),
    test_species = recode(
      test_dataset,
      !!!dataset_to_species,
      .default = NA_character_
    ),
    same_species = train_species == test_species,
    macro_f1 = as.numeric(macro_f1)
  )

unmapped_datasets <- bind_rows(
  pairwise_mapped %>%
    filter(is.na(train_species)) %>%
    distinct(role = "train", dataset = train_dataset),
  pairwise_mapped %>%
    filter(is.na(test_species)) %>%
    distinct(role = "test", dataset = test_dataset)
)

if (nrow(unmapped_datasets) > 0L) {
  write_csv(
    unmapped_datasets,
    file.path(csv_dir, "error_unmapped_datasets.csv")
  )

  stop(
    "Unmapped dataset names found. See error_unmapped_datasets.csv.",
    call. = FALSE
  )
}

if (any(!is.finite(pairwise_mapped$macro_f1))) {
  stop("Non-finite Macro-F1 values found.", call. = FALSE)
}


# ── 10. Core traits and Gower distance ────────────────────────────────────────

print_section("4. CORE TRAITS AND GOWER DISTANCE")

required_core_trait_columns <- c(
  "species",
  "log_body_mass_g",
  "dominant_locomotor_mode",
  "body_configuration_type",
  "expected_trunk_motion_amplitude",
  "expected_directional_change_tendency",
  "expected_head_neck_contribution_to_acceleration_relevant_motion",
  "dominant_feeding_foraging_mode"
)

core_traits_raw <- read_csv(
  core_trait_file,
  show_col_types = FALSE
) %>%
  mutate(across(where(is.character), str_trim)) %>%
  filter(
    !if_all(everything(), ~ is.na(.x) | .x == "")
  )

missing_trait_columns <- setdiff(
  required_core_trait_columns,
  names(core_traits_raw)
)

if (length(missing_trait_columns) > 0L) {
  stop(
    "Core trait file is missing: ",
    paste(missing_trait_columns, collapse = ", "),
    call. = FALSE
  )
}

core_traits_raw <- core_traits_raw %>%
  select(all_of(required_core_trait_columns))

if (anyDuplicated(core_traits_raw$species) > 0L) {
  stop("Duplicated species found in core trait table.", call. = FALSE)
}

ordinal_levels <- c("low", "moderate", "high", "very_high")

core_traits_typed <- core_traits_raw %>%
  mutate(
    log_body_mass_g = as.numeric(log_body_mass_g),
    across(
      c(
        expected_trunk_motion_amplitude,
        expected_directional_change_tendency,
        expected_head_neck_contribution_to_acceleration_relevant_motion
      ),
      ~ ordered(str_to_lower(.x), levels = ordinal_levels)
    ),
    across(
      c(
        dominant_locomotor_mode,
        body_configuration_type,
        dominant_feeding_foraging_mode
      ),
      as.factor
    )
  )

if (any(!complete.cases(core_traits_typed))) {
  stop("Missing or invalid values after trait conversion.", call. = FALSE)
}

study_species <- sort(
  unique(
    c(
      pairwise_mapped$train_species,
      pairwise_mapped$test_species
    )
  )
)

missing_trait_species <- setdiff(
  study_species,
  core_traits_typed$species
)

if (length(missing_trait_species) > 0L) {
  stop(
    "Missing traits for: ",
    paste(missing_trait_species, collapse = ", "),
    call. = FALSE
  )
}

# Gower distance supports the mixed numeric, ordinal, and nominal traits.
# All seven core traits receive equal weight.
core_trait_names <- setdiff(
  required_core_trait_columns,
  "species"
)

core_trait_weights <- rep(
  1,
  length(core_trait_names)
)

names(core_trait_weights) <- core_trait_names

core_gower_matrix <- core_traits_typed %>%
  select(all_of(core_trait_names)) %>%
  daisy(
    metric = "gower",
    weights = core_trait_weights
  ) %>%
  as.matrix()

rownames(core_gower_matrix) <- core_traits_typed$species
colnames(core_gower_matrix) <- core_traits_typed$species

core_gower_wide <- as.data.frame(core_gower_matrix) %>%
  rownames_to_column("train_species")

core_gower_long <- core_gower_wide %>%
  pivot_longer(
    -train_species,
    names_to = "test_species",
    values_to = "trait_distance_core"
  )

write_csv(
  core_traits_raw,
  file.path(csv_dir, "01_core_trait_table.csv")
)

write_csv(
  tibble(
    trait = names(core_trait_weights),
    gower_weight = as.numeric(core_trait_weights)
  ),
  file.path(csv_dir, "01b_core_trait_gower_weights.csv")
)

write_csv(
  core_gower_wide,
  file.path(csv_dir, "02_core_gower_distance_matrix.csv")
)

write_csv(
  core_gower_long,
  file.path(csv_dir, "03_core_gower_distance_long.csv")
)


# ── 11. Reconstruct the 432-row H3 table ──────────────────────────────────────

print_section("5. RECONSTRUCT H3 TABLE")

h3_data <- pairwise_mapped %>%
  filter(!same_species) %>%
  group_by(model, train_species, test_species) %>%
  summarise(
    mean_macro_f1 = mean(macro_f1, na.rm = TRUE),
    n_dataset_pair_rows = n(),
    .groups = "drop"
  ) %>%
  mutate(
    ordered_species_pair = paste(
      train_species,
      test_species,
      sep = "__"
    ),
    unordered_species_pair = map2_chr(
      train_species,
      test_species,
      ~ paste(sort(c(.x, .y)), collapse = "__")
    ),
    trait_distance_core = map2_dbl(
      train_species,
      test_species,
      ~ core_gower_matrix[.x, .y]
    )
  )

distance_mean <- mean(h3_data$trait_distance_core)
distance_sd <- sd(h3_data$trait_distance_core)

h3_data <- h3_data %>%
  mutate(
    z_trait_distance_core =
      (trait_distance_core - distance_mean) / distance_sd,
    model = factor(model, levels = model_order),
    ordered_species_pair = factor(ordered_species_pair),
    unordered_species_pair = factor(unordered_species_pair),
    train_species = factor(train_species),
    test_species = factor(test_species)
  )

structure_summary <- tibble(
  quantity = c(
    "model_specific_observations",
    "classifiers",
    "species",
    "ordered_cross_species_pairs",
    "unordered_species_combinations",
    "mean_raw_dataset_pair_rows_per_model_species_pair",
    "minimum_raw_dataset_pair_rows_per_model_species_pair",
    "maximum_raw_dataset_pair_rows_per_model_species_pair",
    "trait_distance_mean",
    "trait_distance_sd"
  ),
  value = c(
    nrow(h3_data),
    n_distinct(h3_data$model),
    n_distinct(
      c(
        as.character(h3_data$train_species),
        as.character(h3_data$test_species)
      )
    ),
    n_distinct(h3_data$ordered_species_pair),
    n_distinct(h3_data$unordered_species_pair),
    mean(h3_data$n_dataset_pair_rows),
    min(h3_data$n_dataset_pair_rows),
    max(h3_data$n_dataset_pair_rows),
    distance_mean,
    distance_sd
  )
)

expected_structure <- c(
  model_specific_observations = 432,
  classifiers = 6,
  species = 9,
  ordered_cross_species_pairs = 72,
  unordered_species_combinations = 36
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
  stop(
    "Unexpected H3 data structure. Check 05_data_structure_summary.csv.",
    call. = FALSE
  )
}

write_csv(
  h3_data,
  file.path(csv_dir, "04_h3_mixed_model_data.csv")
)

write_csv(
  structure_summary,
  file.path(csv_dir, "05_data_structure_summary.csv")
)

cat("H3 rows:", nrow(h3_data), "\n")
cat(
  "Ordered species pairs:",
  n_distinct(h3_data$ordered_species_pair),
  "\n"
)


# ── 12. Descriptive Spearman correlations ─────────────────────────────────────

print_section("6. DESCRIPTIVE SPEARMAN CORRELATIONS")

descriptive_pooled <- safe_spearman(
  h3_data$trait_distance_core,
  h3_data$mean_macro_f1
) %>%
  mutate(
    analysis = "pooled_descriptive",
    model = "All models pooled",
    n_unique_ordered_species_pairs = n_distinct(
      h3_data$ordered_species_pair
    ),
    .before = 1
  )

descriptive_by_model <- h3_data %>%
  group_by(model) %>%
  group_modify(
    ~ safe_spearman(
      .x$trait_distance_core,
      .x$mean_macro_f1
    )
  ) %>%
  ungroup() %>%
  mutate(
    analysis = "model_specific_descriptive",
    model = as.character(model),
    n_unique_ordered_species_pairs = n,
    .before = 1
  ) %>%
  mutate(
    p_value_holm = p.adjust(p_value_naive, method = "holm")
  )

descriptive_spearman <- bind_rows(
  descriptive_pooled %>%
    mutate(p_value_holm = NA_real_),
  descriptive_by_model
)

write_csv(
  descriptive_spearman,
  file.path(csv_dir, "06_descriptive_spearman_correlations.csv")
)

print_table(
  descriptive_spearman %>%
    select(
      analysis,
      model,
      n,
      rho,
      ci_low_naive,
      ci_high_naive,
      p_value_naive,
      p_value_holm
    )
)


# ── 13. Fit mixed-effects models ──────────────────────────────────────────────

print_section("7. FIT MIXED-EFFECTS MODELS")

# Primary dependence structure:
# - Unordered pair links reciprocal directions.
# - Directed pair links the six classifier results for one direction.
# - Source and target intercepts account for repeated species use.
formula_full_null <- mean_macro_f1 ~ model +
  (1 | unordered_species_pair) +
  (1 | ordered_species_pair) +
  (1 | train_species) +
  (1 | test_species)

formula_full_additive <- mean_macro_f1 ~ z_trait_distance_core + model +
  (1 | unordered_species_pair) +
  (1 | ordered_species_pair) +
  (1 | train_species) +
  (1 | test_species)

formula_full_interaction <- mean_macro_f1 ~ z_trait_distance_core * model +
  (1 | unordered_species_pair) +
  (1 | ordered_species_pair) +
  (1 | train_species) +
  (1 | test_species)

# Sensitivity model without the unordered-pair intercept.
formula_directional_null <- mean_macro_f1 ~ model +
  (1 | ordered_species_pair) +
  (1 | train_species) +
  (1 | test_species)

formula_directional_additive <- mean_macro_f1 ~
  z_trait_distance_core + model +
  (1 | ordered_species_pair) +
  (1 | train_species) +
  (1 | test_species)

# Reduced sensitivity model for repeated classifiers within each directed pair.
# It does not account for repeated source or target species.
formula_pair_only_null <- mean_macro_f1 ~ model +
  (1 | ordered_species_pair)

formula_pair_only_additive <- mean_macro_f1 ~
  z_trait_distance_core + model +
  (1 | ordered_species_pair)

# ML fits for likelihood-ratio tests, REML fits for coefficient reporting.
model_specifications <- tibble::tribble(
  ~label,                         ~formula,                       ~reml,
  "full_null_ml",                 formula_full_null,              FALSE,
  "full_additive_ml",             formula_full_additive,          FALSE,
  "full_interaction_ml",          formula_full_interaction,       FALSE,
  "full_additive_reml",           formula_full_additive,          TRUE,
  "full_interaction_reml",        formula_full_interaction,       TRUE,
  "directional_null_ml",          formula_directional_null,       FALSE,
  "directional_additive_ml",      formula_directional_additive,   FALSE,
  "directional_additive_reml",    formula_directional_additive,   TRUE,
  "pair_only_null_ml",            formula_pair_only_null,         FALSE,
  "pair_only_additive_ml",        formula_pair_only_additive,     FALSE,
  "pair_only_additive_reml",      formula_pair_only_additive,     TRUE
)

fitted_models <- pmap(
  model_specifications,
  function(label, formula, reml) {
    safe_lmer(
      formula = formula,
      data = h3_data,
      reml = reml,
      label = label
    )
  }
) %>%
  setNames(model_specifications$label)

get_fit <- function(label) {
  fitted_models[[label]]$fit
}

model_diagnostics <- map_dfr(
  fitted_models,
  extract_model_diagnostics
)

write_csv(
  model_diagnostics,
  file.path(csv_dir, "07_model_diagnostics.csv")
)

print_table(
  model_diagnostics %>%
    select(
      model_name,
      reml,
      n_groups_unordered_pair,
      singular,
      optimizer_convergence_code,
      convergence_messages,
      warnings,
      error,
      aic,
      bic
    )
)


# ── 14. Fixed and random effects ──────────────────────────────────────────────

print_section("8. FIXED AND RANDOM EFFECTS")

full_additive_fixed <- extract_fixed_effects(
  get_fit("full_additive_reml"),
  "full_additive_reml"
)

full_additive_random <- extract_random_effects(
  get_fit("full_additive_reml"),
  "full_additive_reml"
)

full_interaction_fixed <- extract_fixed_effects(
  get_fit("full_interaction_reml"),
  "full_interaction_reml"
)

directional_fixed <- extract_fixed_effects(
  get_fit("directional_additive_reml"),
  "directional_additive_reml"
)

directional_random <- extract_random_effects(
  get_fit("directional_additive_reml"),
  "directional_additive_reml"
)

pair_only_fixed <- extract_fixed_effects(
  get_fit("pair_only_additive_reml"),
  "pair_only_additive_reml"
)

pair_only_random <- extract_random_effects(
  get_fit("pair_only_additive_reml"),
  "pair_only_additive_reml"
)

write_csv(
  full_additive_fixed,
  file.path(csv_dir, "08_full_additive_fixed_effects.csv")
)

write_csv(
  full_additive_random,
  file.path(csv_dir, "09_full_additive_random_effects.csv")
)

write_csv(
  full_interaction_fixed,
  file.path(csv_dir, "10_full_interaction_fixed_effects.csv")
)

write_csv(
  directional_fixed,
  file.path(csv_dir, "11_directional_pair_sensitivity_fixed_effects.csv")
)

write_csv(
  directional_random,
  file.path(csv_dir, "12_directional_pair_sensitivity_random_effects.csv")
)

write_csv(
  pair_only_fixed,
  file.path(csv_dir, "13_pair_only_sensitivity_fixed_effects.csv")
)

write_csv(
  pair_only_random,
  file.path(csv_dir, "14_pair_only_sensitivity_random_effects.csv")
)

print_table(full_additive_fixed)


# ── 15. Nested model comparisons ──────────────────────────────────────────────

print_section("9. MODEL COMPARISONS")

model_comparisons <- bind_rows(
  compare_nested_models(
    reduced_fit = get_fit("full_null_ml"),
    full_fit = get_fit("full_additive_ml"),
    comparison_name = "full_null_vs_full_additive_trait_effect"
  ),
  compare_nested_models(
    reduced_fit = get_fit("full_additive_ml"),
    full_fit = get_fit("full_interaction_ml"),
    comparison_name = "full_additive_vs_full_interaction"
  ),
  compare_nested_models(
    reduced_fit = get_fit("directional_null_ml"),
    full_fit = get_fit("directional_additive_ml"),
    comparison_name =
      "directional_null_vs_directional_additive_trait_effect"
  ),
  compare_nested_models(
    reduced_fit = get_fit("pair_only_null_ml"),
    full_fit = get_fit("pair_only_additive_ml"),
    comparison_name = "pair_only_null_vs_pair_only_additive_trait_effect"
  )
)

write_csv(
  model_comparisons,
  file.path(csv_dir, "15_model_comparisons_likelihood_ratio.csv")
)

print_table(model_comparisons)


# ── 16. R-squared and added explanatory contribution ─────────────────────────

print_section("10. MIXED-MODEL R-SQUARED")

r2_targets <- c(
  "full_null_ml",
  "full_additive_ml",
  "full_interaction_ml",
  "directional_null_ml",
  "directional_additive_ml",
  "pair_only_null_ml",
  "pair_only_additive_ml"
)

model_r2 <- map_dfr(
  r2_targets,
  ~ extract_r2(get_fit(.x), .x)
)

get_marginal_r2 <- function(label) {
  model_r2 %>%
    filter(model_name == label) %>%
    pull(marginal_r2)
}

# Difference in marginal R-squared between otherwise identical ML models.
# This approximates the added fixed-effect contribution; it is not rho squared.
r2_change <- tibble(
  comparison = c(
    "primary_dyadic_model_added_trait_distance",
    "directional_sensitivity_added_trait_distance",
    "pair_only_sensitivity_added_trait_distance"
  ),
  delta_marginal_r2 = c(
    get_marginal_r2("full_additive_ml") -
      get_marginal_r2("full_null_ml"),
    get_marginal_r2("directional_additive_ml") -
      get_marginal_r2("directional_null_ml"),
    get_marginal_r2("pair_only_additive_ml") -
      get_marginal_r2("pair_only_null_ml")
  )
)

write_csv(
  model_r2,
  file.path(csv_dir, "16_model_r2_nakagawa.csv")
)

write_csv(
  r2_change,
  file.path(csv_dir, "17_added_trait_distance_r2.csv")
)

print_table(model_r2)
print_table(r2_change)


# ── 17. Model-specific trait-distance slopes ──────────────────────────────────

print_section("11. MODEL-SPECIFIC TRAIT-DISTANCE SLOPES")

interaction_slopes <- tibble(
  model = character(),
  trend = numeric(),
  std_error = numeric(),
  df = numeric(),
  ci_low = numeric(),
  ci_high = numeric(),
  statistic = numeric(),
  p_value = numeric()
)

interaction_slope_contrasts <- tibble(
  contrast = character(),
  estimate = numeric(),
  SE = numeric(),
  df = numeric(),
  lower.CL = numeric(),
  upper.CL = numeric(),
  t.ratio = numeric(),
  p.value = numeric()
)

if (!is.null(get_fit("full_interaction_reml"))) {
  emtrend_object <- emmeans::emtrends(
    get_fit("full_interaction_reml"),
    specs = "model",
    var = "z_trait_distance_core"
  )

  interaction_slopes <- summary(
    emtrend_object,
    infer = c(TRUE, TRUE),
    level = confidence_level,
    adjust = "holm"
  ) %>%
    as.data.frame() %>%
    as_tibble() %>%
    rename(
      trend = z_trait_distance_core.trend,
      std_error = SE,
      ci_low = lower.CL,
      ci_high = upper.CL,
      statistic = t.ratio,
      p_value = p.value
    )

  interaction_slope_contrasts <- summary(
    pairs(emtrend_object, adjust = "tukey"),
    infer = c(TRUE, TRUE),
    level = confidence_level
  ) %>%
    as.data.frame() %>%
    as_tibble()
}

write_csv(
  interaction_slopes,
  file.path(csv_dir, "18_interaction_model_specific_slopes.csv")
)

write_csv(
  interaction_slope_contrasts,
  file.path(csv_dir, "19_interaction_slope_contrasts.csv")
)

print_table(interaction_slopes)


# ── 18. Predictions ───────────────────────────────────────────────────────────

print_section("12. FIXED-EFFECT PREDICTIONS")

prediction_grid <- make_fixed_prediction_grid(
  get_fit("full_additive_reml"),
  h3_data
)

marginal_prediction_grid <- make_marginal_prediction_grid(
  get_fit("full_additive_reml"),
  h3_data
)

observed_ordered_pair_means <- h3_data %>%
  group_by(
    ordered_species_pair,
    unordered_species_pair,
    train_species,
    test_species,
    trait_distance_core
  ) %>%
  summarise(
    mean_macro_f1_across_models = mean(mean_macro_f1),
    .groups = "drop"
  )

write_csv(
  prediction_grid,
  file.path(csv_dir, "20_primary_model_specific_fixed_predictions.csv")
)

write_csv(
  marginal_prediction_grid,
  file.path(csv_dir, "21_primary_marginal_fixed_predictions.csv")
)

write_csv(
  observed_ordered_pair_means,
  file.path(csv_dir, "22_observed_ordered_pair_means.csv")
)


# ── 19. Figures ───────────────────────────────────────────────────────────────

print_section("13. FIGURES")

# Descriptive 3 x 2 panel with Spearman rho.
# Mixed-effects models provide the primary inference.
descriptive_plot_labels <- descriptive_by_model %>%
  transmute(
    model = factor(model, levels = model_order),
    label = paste0(
      "ρ = ",
      formatC(rho, format = "f", digits = 3)
    )
  )

plot_descriptive <- ggplot(
  h3_data,
  aes(
    x = trait_distance_core,
    y = mean_macro_f1
  )
) +
  geom_smooth(
    method = "lm",
    formula = y ~ x,
    se = TRUE,
    colour = "grey20",
    fill = "grey80",
    linewidth = 0.75,
    alpha = 0.55
  ) +
  geom_point(
    shape = 21,
    colour = "grey15",
    fill = "grey55",
    stroke = 0.35,
    size = 2.0,
    alpha = 0.90
  ) +
  geom_label(
    data = descriptive_plot_labels,
    aes(
      x = Inf,
      y = Inf,
      label = label
    ),
    inherit.aes = FALSE,
    hjust = 1.08,
    vjust = 1.20,
    size = 3.2,
    fontface = "bold",
    colour = "black",
    fill = "white",
    linewidth = 0.25,
    label.padding = grid::unit(0.12, "lines"),
    label.r = grid::unit(0.08, "lines")
  ) +
  facet_wrap(
    ~ model,
    ncol = 2,
    drop = FALSE
  ) +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, by = 0.25),
    expand = expansion(mult = c(0.02, 0.02))
  ) +
  scale_y_continuous(
    limits = c(0, 1.05),
    breaks = seq(0, 1, by = 0.25),
    expand = expansion(mult = c(0.01, 0.02))
  ) +
  labs(
    x = "Core trait distance (Gower)",
    y = "Mean Macro-F1"
  ) +
  theme_h3(base_size = 12) +
  theme(
    panel.grid.major = element_line(
      colour = "grey85",
      linewidth = 0.35,
      linetype = "dashed"
    ),
    strip.background = element_rect(
      fill = "grey92",
      colour = "black",
      linewidth = 0.5
    ),
    strip.text = element_text(
      face = "bold",
      colour = "black",
      size = 11
    ),
    panel.border = element_rect(
      colour = "black",
      fill = NA,
      linewidth = 0.5
    ),
    axis.text = element_text(
      colour = "black",
      size = 10
    ),
    axis.title = element_text(
      colour = "black",
      size = 12
    ),
    legend.position = "none",
    panel.spacing.x = grid::unit(1.6, "lines"),
    panel.spacing.y = grid::unit(0.9, "lines"),
    axis.text.x = element_text(
      margin = margin(t = 3)
    ),
    axis.text.y = element_text(
      margin = margin(r = 3)
    ),
    axis.title.x = element_text(
      margin = margin(t = 9)
    ),
    axis.title.y = element_text(
      margin = margin(r = 9)
    ),
    plot.margin = margin(10, 14, 12, 14)
  )

save_plot_png(
  plot_descriptive,
  "S01_trait_distance_descriptive_by_model_bw_3x2",
  width = 9.6,
  height = 10.8
)

if (nrow(marginal_prediction_grid) > 0L) {
  plot_primary_marginal <- ggplot() +
    geom_point(
      data = observed_ordered_pair_means,
      aes(
        x = trait_distance_core,
        y = mean_macro_f1_across_models
      ),
      colour = "grey45",
      size = 1.8,
      alpha = 0.75
    ) +
    geom_ribbon(
      data = marginal_prediction_grid,
      aes(
        x = trait_distance_core,
        ymin = ci_low,
        ymax = ci_high
      ),
      fill = "grey75",
      alpha = 0.45
    ) +
    geom_line(
      data = marginal_prediction_grid,
      aes(
        x = trait_distance_core,
        y = predicted_macro_f1
      ),
      colour = "grey15",
      linewidth = 0.9
    ) +
    scale_x_continuous(expand = expansion(mult = 0.03)) +
    scale_y_continuous(expand = expansion(mult = 0.06)) +
    labs(
      x = axis_label_distance,
      y = "Mean Macro-F1 across classifiers"
    ) +
    theme_h3()

  save_plot_png(
    plot_primary_marginal,
    "02_primary_dyadic_marginal_prediction",
    width = 7.2,
    height = 5.2
  )
}

# Adjusted model-specific predictions. Lines are parallel because the additive
# model estimates one shared trait-distance slope.
if (nrow(prediction_grid) > 0L) {
  plot_adjusted_by_model <- ggplot() +
    geom_point(
      data = h3_data,
      aes(
        x = trait_distance_core,
        y = mean_macro_f1
      ),
      colour = "grey82",
      size = 1.1,
      alpha = 0.45
    ) +
    geom_line(
      data = prediction_grid,
      aes(
        x = trait_distance_core,
        y = predicted_macro_f1,
        colour = model
      ),
      linewidth = 0.8
    ) +
    scale_colour_manual(values = model_colors) +
    scale_x_continuous(expand = expansion(mult = 0.02)) +
    labs(
      x = axis_label_distance,
      y = axis_label_predicted
    ) +
    guides(colour = guide_legend(nrow = 1)) +
    theme_h3()

  save_plot_png(
    plot_adjusted_by_model,
    "S02_primary_adjusted_predictions_by_model",
    width = 8.5,
    height = 6
  )
}

# Residual diagnostics for the primary additive model.
if (!is.null(get_fit("full_additive_reml"))) {
  additive_fit <- get_fit("full_additive_reml")

  diagnostic_data <- tibble(
    fitted = fitted(additive_fit),
    residual = resid(additive_fit),
    standardized_residual = as.numeric(scale(resid(additive_fit)))
  )

  write_csv(
    diagnostic_data,
    file.path(csv_dir, "23_full_additive_residual_diagnostics.csv")
  )

  plot_residual <- ggplot(
    diagnostic_data,
    aes(
      x = fitted,
      y = residual
    )
  ) +
    geom_hline(
      yintercept = 0,
      linetype = 2,
      colour = "grey40",
      linewidth = 0.4
    ) +
    geom_point(
      colour = "#0072B2",
      size = 1.5,
      alpha = 0.6
    ) +
    geom_smooth(
      method = "loess",
      formula = y ~ x,
      se = FALSE,
      colour = "grey20",
      linewidth = 0.7
    ) +
    labs(
      x = "Fitted Macro-F1",
      y = "Residual"
    ) +
    theme_h3()

  save_plot_png(
    plot_residual,
    "03_full_additive_residuals_vs_fitted",
    width = 6.5,
    height = 5
  )

  plot_qq <- ggplot(
    diagnostic_data,
    aes(sample = standardized_residual)
  ) +
    stat_qq_line(
      colour = "grey40",
      linetype = 2,
      linewidth = 0.4
    ) +
    stat_qq(
      colour = "#0072B2",
      size = 1.5,
      alpha = 0.6
    ) +
    labs(
      x = "Theoretical quantiles",
      y = "Standardized residuals"
    ) +
    theme_h3()

  save_plot_png(
    plot_qq,
    "04_full_additive_residual_qq",
    width = 6.5,
    height = 5
  )
}

# Variance components for the primary additive model.
if (
  nrow(full_additive_random) > 0L &&
  any(is.finite(full_additive_random$std_dev))
) {
  plot_random_sd <- full_additive_random %>%
    filter(is.finite(std_dev)) %>%
    mutate(
      display_group = recode(
        group,
        "unordered_species_pair" = "Unordered species pair",
        "ordered_species_pair" = "Ordered species pair",
        "train_species" = "Source species",
        "test_species" = "Target species",
        "Residual" = "Residual"
      )
    ) %>%
    ggplot(
      aes(
        x = reorder(display_group, std_dev),
        y = std_dev
      )
    ) +
    geom_col(
      fill = "grey55",
      colour = "grey20",
      linewidth = 0.3,
      width = 0.65
    ) +
    coord_flip() +
    scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
    labs(
      x = NULL,
      y = "Standard deviation"
    ) +
    theme_h3()

  save_plot_png(
    plot_random_sd,
    "05_full_additive_random_effect_sd",
    width = 6.5,
    height = 4
  )
}

# Exploratory slopes; interpret them with the omnibus interaction test.
if (nrow(interaction_slopes) > 0L) {
  plot_slopes <- interaction_slopes %>%
    mutate(
      model = factor(
        as.character(model),
        levels = rev(model_order)
      )
    ) %>%
    ggplot(
      aes(
        x = trend,
        y = model,
        colour = model
      )
    ) +
    geom_vline(
      xintercept = 0,
      linetype = 2,
      colour = "grey40",
      linewidth = 0.4
    ) +
    geom_pointrange(
      aes(
        xmin = ci_low,
        xmax = ci_high
      ),
      linewidth = 0.6,
      size = 0.45
    ) +
    scale_colour_manual(
      values = model_colors,
      guide = "none"
    ) +
    labs(
      x = "Slope per SD of trait distance",
      y = NULL
    ) +
    theme_h3()

  save_plot_png(
    plot_slopes,
    "S06_interaction_model_specific_slopes",
    width = 6.5,
    height = 4
  )
}


# ── 20. Reproducibility exports ───────────────────────────────────────────────

print_section("14. REPRODUCIBILITY EXPORTS")

full_diagnostic <- model_diagnostics %>%
  filter(model_name == "full_additive_reml")

trait_effect <- full_additive_fixed %>%
  filter(term == "z_trait_distance_core")

trait_lrt <- model_comparisons %>%
  filter(comparison == "full_null_vs_full_additive_trait_effect")

interaction_lrt <- model_comparisons %>%
  filter(comparison == "full_additive_vs_full_interaction")

pooled_rho <- descriptive_pooled$rho[[1]]

full_singular <- if (nrow(full_diagnostic) == 1L) {
  full_diagnostic$singular[[1]]
} else {
  NA
}

full_optimizer_code <- if (nrow(full_diagnostic) == 1L) {
  full_diagnostic$optimizer_convergence_code[[1]]
} else {
  NA_integer_
}

full_convergence_message <- if (nrow(full_diagnostic) == 1L) {
  full_diagnostic$convergence_messages[[1]]
} else {
  NA_character_
}

full_warning_message <- if (nrow(full_diagnostic) == 1L) {
  full_diagnostic$warnings[[1]]
} else {
  NA_character_
}

full_error_message <- if (nrow(full_diagnostic) == 1L) {
  full_diagnostic$error[[1]]
} else {
  NA_character_
}

full_converged <- (
  nrow(full_diagnostic) == 1L &&
    (is.na(full_optimizer_code) || full_optimizer_code == 0L) &&
    (is.na(full_error_message) || full_error_message == "") &&
    (is.na(full_convergence_message) || full_convergence_message == "") &&
    (is.na(full_warning_message) || full_warning_message == "")
)

interaction_p <- if (nrow(interaction_lrt) == 1L) {
  interaction_lrt$p_value[[1]]
} else {
  NA_real_
}

# The additive dyadic model remains primary regardless of the interaction
# p-value. The interaction is a separate heterogeneity test.
primary_status <- if (
  isTRUE(full_converged) &&
  isFALSE(full_singular)
) {
  "primary_additive_dyadic_model_valid"
} else {
  "primary_additive_dyadic_model_requires_review"
}

interaction_status <- if (
  is.finite(interaction_p) &&
  interaction_p < 0.05
) {
  "classifier_slope_heterogeneity_supported"
} else {
  "classifier_slope_heterogeneity_not_supported"
}

formula_registry <- tibble(
  analysis = c(
    "full_null_ml",
    "full_additive_ml_and_reml",
    "full_interaction_ml_and_reml",
    "directional_null_ml",
    "directional_additive_ml_and_reml",
    "pair_only_null_ml",
    "pair_only_additive_ml_and_reml"
  ),
  formula = c(
    paste(deparse(formula_full_null), collapse = " "),
    paste(deparse(formula_full_additive), collapse = " "),
    paste(deparse(formula_full_interaction), collapse = " "),
    paste(deparse(formula_directional_null), collapse = " "),
    paste(deparse(formula_directional_additive), collapse = " "),
    paste(deparse(formula_pair_only_null), collapse = " "),
    paste(deparse(formula_pair_only_additive), collapse = " ")
  )
)

write_csv(
  formula_registry,
  file.path(csv_dir, "24_model_formula_registry.csv")
)

run_manifest <- tibble(
  item = c(
    "script",
    "run_time",
    "base_dir",
    "core_trait_file",
    "output_dir",
    "random_seed",
    "confidence_level",
    "singularity_tolerance",
    "full_model_optimizer_convergence_code",
    "full_model_convergence_messages",
    "full_model_warnings",
    "full_model_error",
    "full_model_converged",
    "full_model_singular",
    "primary_inference_status",
    "interaction_status"
  ),
  value = c(
    "01_H3_mixed_effects_primary_final.R",
    as.character(Sys.time()),
    base_dir,
    core_trait_file,
    out_dir,
    "42",
    as.character(confidence_level),
    as.character(singularity_tolerance),
    as.character(full_optimizer_code),
    ifelse(
      is.na(full_convergence_message) || full_convergence_message == "",
      "none",
      full_convergence_message
    ),
    ifelse(
      is.na(full_warning_message) || full_warning_message == "",
      "none",
      full_warning_message
    ),
    ifelse(
      is.na(full_error_message) || full_error_message == "",
      "none",
      full_error_message
    ),
    as.character(full_converged),
    as.character(full_singular),
    primary_status,
    interaction_status
  )
)

write_csv(
  run_manifest,
  file.path(csv_dir, "25_run_manifest.csv")
)

primary_diagnostic_lines <- c(
  "PRIMARY MODEL DIAGNOSTICS",
  paste(
    "Optimizer convergence code:",
    ifelse(is.na(full_optimizer_code), "NA", full_optimizer_code)
  ),
  paste(
    "Convergence messages:",
    ifelse(
      is.na(full_convergence_message) || full_convergence_message == "",
      "none",
      full_convergence_message
    )
  ),
  paste(
    "Captured warnings:",
    ifelse(
      is.na(full_warning_message) || full_warning_message == "",
      "none",
      full_warning_message
    )
  ),
  paste(
    "Model-fitting error:",
    ifelse(
      is.na(full_error_message) || full_error_message == "",
      "none",
      full_error_message
    )
  ),
  paste(
    "Singular fit:",
    ifelse(is.na(full_singular), "NA", full_singular)
  ),
  paste(
    "Primary model converged without warnings:",
    full_converged
  ),
  paste(
    "Primary model valid for predefined inference:",
    primary_status == "primary_additive_dyadic_model_valid"
  )
)

write_txt(
  file.path(txt_dir, "01_model_summaries.txt"),
  c(
    primary_diagnostic_lines,
    "",
    "FULL ADDITIVE REML MODEL",
    capture.output(summary(get_fit("full_additive_reml"))),
    "",
    "FULL INTERACTION REML MODEL",
    capture.output(summary(get_fit("full_interaction_reml"))),
    "",
    "DIRECTIONAL-PAIR ADDITIVE REML SENSITIVITY MODEL",
    capture.output(summary(get_fit("directional_additive_reml"))),
    "",
    "PAIR-ONLY ADDITIVE REML SENSITIVITY MODEL",
    capture.output(summary(get_fit("pair_only_additive_reml"))),
    "",
    "ML MODEL COMPARISONS",
    capture.output(print_table(model_comparisons))
  )
)

write_txt(
  file.path(txt_dir, "02_session_info.txt"),
  capture.output(sessionInfo())
)


# ── 21. Console summary ───────────────────────────────────────────────────────

print_section("15. FINAL CONSOLE SUMMARY")

cat("Primary inference status:", primary_status, "\n")
cat("Interaction status:", interaction_status, "\n")
cat(
  "Optimizer convergence code:",
  ifelse(is.na(full_optimizer_code), "NA", full_optimizer_code),
  "\n"
)
cat("Full model singular:", full_singular, "\n")

cat(
  "Convergence message:",
  ifelse(
    is.na(full_convergence_message) || full_convergence_message == "",
    "none",
    full_convergence_message
  ),
  "\n"
)

cat(
  "Captured warnings:",
  ifelse(
    is.na(full_warning_message) || full_warning_message == "",
    "none",
    full_warning_message
  ),
  "\n"
)

cat(
  "Model-fitting error:",
  ifelse(
    is.na(full_error_message) || full_error_message == "",
    "none",
    full_error_message
  ),
  "\n\n"
)

if (nrow(trait_effect) == 1L) {
  cat(
    "Trait-distance coefficient:",
    fmt_num(trait_effect$estimate[[1]], 3),
    "\n"
  )

  cat(
    "95% CI:",
    fmt_ci(
      trait_effect$ci_low[[1]],
      trait_effect$ci_high[[1]],
      3
    ),
    "\n"
  )

  cat(
    "Satterthwaite p:",
    fmt_p(trait_effect$p_value[[1]]),
    "\n"
  )
}

cat(
  "Trait-effect LRT p:",
  fmt_p(trait_lrt$p_value[[1]]),
  "\n"
)

cat(
  "Interaction LRT p:",
  fmt_p(interaction_p),
  "\n"
)

cat(
  "Descriptive pooled rho:",
  fmt_num(pooled_rho, 3),
  "\n"
)

cat(
  "\nOutputs written to:\n",
  "- ", csv_dir, "\n",
  "- ", plots_dir, "\n",
  "- ", txt_dir, "\n",
  sep = ""
)

# End of script

