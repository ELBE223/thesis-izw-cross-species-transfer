# =============================================================================
# 02_H3_distance_proxy_mixed_models_final.R
# =============================================================================
# Author : Lucas Beseler
#
# Purpose:
# - Compare five biological-distance specifications for H3.
# - Apply the same dyadic mixed-effects structure used in the primary analysis.
# - Report standardized coefficients that are comparable across proxies.
# - Retain Spearman correlations as descriptive secondary analyses.
#
# Random-effects structures:
# - Primary: unordered pair + directed pair + source species + target species.
# - Directional sensitivity: directed pair + source species + target species.
# - Pair-only sensitivity: directed pair.
#
# The additive dyadic model remains primary for each distance proxy.
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
  "emmeans",
  "ape"
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

core_trait_filename <- "species_traits_core_h3_final.csv"
extended_trait_filename <- "species_traits_extended_sensitivity_full_final.csv"

confidence_level <- 0.95
singularity_tolerance <- 1e-4

run_interaction_models <- TRUE
run_directional_sensitivity <- TRUE
run_pair_only_sensitivity <- TRUE
run_loso_mixed_models <- TRUE

# Show beta and 95% CI directly in the main forest plot.
forest_show_value_labels <- TRUE
forest_label_size <- 3.1

# Black-and-white emphasis for the manuscript figure.
forest_role_levels <- c(
  "Primary H3 distance",
  "Sensitivity proxy"
)

forest_line_colors <- c(
  "Primary H3 distance" = "black",
  "Sensitivity proxy"   = "grey35"
)

forest_fill_colors <- c(
  "Primary H3 distance" = "black",
  "Sensitivity proxy"   = "grey72"
)

forest_point_sizes <- c(
  "Primary H3 distance" = 4.2,
  "Sensitivity proxy"   = 3.7
)

plot_dpi <- 600

# Color-blind-safe palette (Okabe-Ito), shared with the primary H3 script.
# Used only for the classifier-specific supplementary figure.
model_colors <- c(
  CNN         = "#0072B2",
  ResNet      = "#D55E00",
  HYDRA       = "#009E73",
  MultiRocket = "#CC79A7",
  RF          = "#E69F00",
  LGBM        = "#56B4E9"
)

axis_label_beta <- expression(
  "Standardized " * beta *
    " (Macro-F1 per 1 SD increase in distance)"
)

axis_label_distance <- "Biological distance"
axis_label_performance <- "Mean cross-species Macro-F1"


# ── 3. Paths and output structure ─────────────────────────────────────────────

models_dir <- file.path(base_dir, "Models")

# H3_DISTANCE_INPUT_DIR can point to an external data directory. Otherwise,
# use the first existing project-relative directory name.
input_dir_override <- Sys.getenv("H3_DISTANCE_INPUT_DIR", unset = "")

input_dir_candidates <- c(
  input_dir_override,
  file.path(base_dir, "TimeTree_Elton"),
  file.path(base_dir, "TimeTree-Elton"),
  file.path(base_dir, "TimeTree:Elton")
)

input_dir_candidates <- unique(
  input_dir_candidates[nzchar(input_dir_candidates)]
)

input_dir <- input_dir_candidates[
  match(TRUE, dir.exists(input_dir_candidates), nomatch = 0L)
]

if (length(input_dir) == 0L) {
  input_dir <- input_dir_candidates[[1]]
}

timetree_file <- file.path(input_dir, "timetree_species_list.nwk")
elton_file <- file.path(input_dir, "MamFuncDat.txt")

output_r_root <- file.path(base_dir, "Output_R")

out_dir <- file.path(
  output_r_root,
  "H3_distance_proxy_mixed_models"
)

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


# ── 4. Constants ──────────────────────────────────────────────────────────────

distance_specs <- tribble(
  ~proxy_id, ~proxy_label, ~proxy_role, ~expected_direction,
  "functional_core_gower",
  "Functional-biomechanical traits (core)",
  "Primary H3 reference",
  "Negative",
  "functional_extended_gower",
  "Functional-biomechanical traits (extended)",
  "Expert-coded robustness specification",
  "Negative",
  "elton_core_gower",
  "EltonTraits (core)",
  "Alternative ecological proxy",
  "Negative",
  "elton_extended_gower",
  "EltonTraits (extended)",
  "Alternative ecological proxy",
  "Negative",
  "timetree_divergence_mya",
  "Phylogenetic distance",
  "Alternative phylogenetic proxy",
  "Negative"
)

required_pairwise_columns <- c(
  "pair_id",
  "train_dataset",
  "test_dataset",
  "accuracy",
  "macro_recall",
  "macro_precision",
  "macro_f1"
)

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

required_extended_trait_columns <- c(
  required_core_trait_columns,
  "activity_pattern",
  "vertical_use_motion_scope",
  "postural_compactness_body_profile_reduction",
  "defensive_posture_specialization"
)

elton_core_columns <- c(
  "BodyMass-Value",
  "ForStrat-Value",
  "Activity-Nocturnal",
  "Activity-Crepuscular",
  "Activity-Diurnal"
)

elton_diet_columns <- c(
  "Diet-Inv",
  "Diet-Vend",
  "Diet-Vect",
  "Diet-Vfish",
  "Diet-Vunk",
  "Diet-Scav",
  "Diet-Fruit",
  "Diet-Nect",
  "Diet-Seed",
  "Diet-PlantO"
)

ordinal_levels <- c("low", "moderate", "high", "very_high")

# Ordinal and nominal expert-coded traits, used for type conversion below.
core_ordinal_traits <- c(
  "expected_trunk_motion_amplitude",
  "expected_directional_change_tendency",
  "expected_head_neck_contribution_to_acceleration_relevant_motion"
)

core_nominal_traits <- c(
  "dominant_locomotor_mode",
  "body_configuration_type",
  "dominant_feeding_foraging_mode"
)

extended_ordinal_traits <- c(
  core_ordinal_traits,
  "postural_compactness_body_profile_reduction",
  "defensive_posture_specialization"
)

extended_nominal_traits <- c(
  core_nominal_traits,
  "activity_pattern",
  "vertical_use_motion_scope"
)


# ── 5. General helpers ────────────────────────────────────────────────────────

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

print_table <- function(x) {
  print(
    as_tibble(x),
    n = Inf,
    width = Inf
  )
  invisible(x)
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

first_existing <- function(paths) {
  hits <- paths[file.exists(paths)]
  
  if (length(hits) == 0L) {
    return(NA_character_)
  }
  
  hits[[1]]
}

resolve_metric_file <- function(directory, candidates) {
  first_existing(
    file.path(directory, candidates)
  )
}

pick_column <- function(data, candidates) {
  hit <- intersect(candidates, names(data))
  
  if (length(hit) == 0L) {
    stop(
      "None of these columns were found: ",
      paste(candidates, collapse = ", "),
      call. = FALSE
    )
  }
  
  hit[[1]]
}

clean_species <- function(x) {
  x %>%
    as.character() %>%
    str_replace_all("_", " ") %>%
    str_replace_all("\\*", "") %>%
    str_squish()
}

# "Canis lupus familiaris" -> "C. lupus familiaris" for compact axis labels.
short_species <- function(x) {
  parts <- str_split(as.character(x), " ")
  
  vapply(
    parts,
    function(part) {
      if (length(part) >= 2L) {
        paste0(
          substr(part[[1]], 1, 1),
          ". ",
          paste(part[-1], collapse = " ")
        )
      } else {
        part[[1]]
      }
    },
    FUN.VALUE = character(1)
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


# ── 6. Plot helpers ───────────────────────────────────────────────────────────

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

save_plot_png <- function(plot_object, filename, width, height) {
  ggsave(
    filename = file.path(plots_dir, filename),
    plot = plot_object,
    width = width,
    height = height,
    dpi = plot_dpi,
    bg = "white"
  )
}

# ── 7. Statistical helpers ────────────────────────────────────────────────────

safe_spearman <- function(
    x,
    y,
    conf_level = confidence_level
) {
  valid <- is.finite(x) & is.finite(y)
  
  x <- x[valid]
  y <- y[valid]
  
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
    z_value <- atanh(
      pmin(pmax(rho, -0.999999), 0.999999)
    )
    
    standard_error_z <- 1 / sqrt(n_obs - 3)
    critical_value <- qnorm((1 + conf_level) / 2)
    
    ci_low <- tanh(z_value - critical_value * standard_error_z)
    ci_high <- tanh(z_value + critical_value * standard_error_z)
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

# Placeholder for models switched off by the run flags.
skipped_model <- function(label) {
  list(
    label = label,
    fit = NULL,
    error = "Not requested",
    warnings = character()
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

extract_fixed_table <- function(
    fit,
    proxy_id,
    model_name,
    conf_level = confidence_level
) {
  if (is.null(fit)) {
    return(
      tibble(
        proxy_id = proxy_id,
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
    rownames_to_column("term")
  
  df_column <- intersect("df", names(coefficients))
  p_column <- intersect("Pr(>|t|)", names(coefficients))
  
  coefficients %>%
    transmute(
      proxy_id = proxy_id,
      model_name = model_name,
      term = term,
      estimate = .data[["Estimate"]],
      std_error = .data[["Std. Error"]],
      df = if (length(df_column) > 0L) {
        .data[[df_column[[1]]]]
      } else {
        NA_real_
      },
      statistic = .data[["t value"]],
      p_value = if (length(p_column) > 0L) {
        .data[[p_column[[1]]]]
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

extract_random_table <- function(
    fit,
    proxy_id,
    model_name
) {
  if (is.null(fit)) {
    return(
      tibble(
        proxy_id = proxy_id,
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
      proxy_id = proxy_id,
      model_name = model_name,
      group = grp,
      term = var1,
      variance = vcov,
      std_dev = sdcor
    ) %>%
    as_tibble()
}

extract_diagnostics <- function(model_object, proxy_id) {
  fit <- model_object$fit
  
  if (is.null(fit)) {
    return(
      tibble(
        proxy_id = proxy_id,
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
    proxy_id = proxy_id,
    model_name = model_object$label,
    reml = isREML(fit),
    n_observations = nobs(fit),
    n_groups_unordered_pair = count_groups("unordered_species_pair"),
    n_groups_ordered_pair = count_groups("ordered_species_pair"),
    n_groups_train_species = count_groups("train_species"),
    n_groups_test_species = count_groups("test_species"),
    singular = isSingular(fit, tol = singularity_tolerance),
    optimizer_convergence_code = get_optimizer_convergence_code(fit),
    convergence_messages = get_convergence_messages(fit),
    warnings = paste(model_object$warnings, collapse = " | "),
    error = model_object$error,
    log_likelihood = as.numeric(logLik(fit)),
    aic = AIC(fit),
    bic = BIC(fit)
  )
}

compare_nested_models <- function(
    reduced_fit,
    full_fit,
    proxy_id,
    comparison_name
) {
  if (is.null(reduced_fit) || is.null(full_fit)) {
    return(
      tibble(
        proxy_id = proxy_id,
        comparison = comparison_name,
        df_difference = NA_real_,
        likelihood_ratio_chisq = NA_real_,
        p_value = NA_real_,
        aic_reduced = NA_real_,
        aic_full = NA_real_
      )
    )
  }
  
  anova_table <- suppressMessages(
    stats::anova(reduced_fit, full_fit)
  ) %>%
    as.data.frame()
  
  required_columns <- c("Df", "Chisq", "Pr(>Chisq)")
  
  missing_columns <- setdiff(
    required_columns,
    names(anova_table)
  )
  
  if (length(missing_columns) > 0L) {
    stop(
      "Nested-model comparison is missing: ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }
  
  comparison_row <- nrow(anova_table)
  
  tibble(
    proxy_id = proxy_id,
    comparison = comparison_name,
    df_difference = as.numeric(anova_table[comparison_row, "Df"]),
    likelihood_ratio_chisq = as.numeric(anova_table[comparison_row, "Chisq"]),
    p_value = as.numeric(anova_table[comparison_row, "Pr(>Chisq)"]),
    aic_reduced = AIC(reduced_fit),
    aic_full = AIC(full_fit)
  )
}

extract_r2 <- function(fit, proxy_id, model_name) {
  empty_r2 <- tibble(
    proxy_id = proxy_id,
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
    proxy_id = proxy_id,
    model_name = model_name,
    marginal_r2 = as.numeric(r2_result$R2_marginal),
    conditional_r2 = as.numeric(r2_result$R2_conditional)
  )
}

extract_interaction_slopes <- function(fit, proxy_id) {
  empty_result <- tibble(
    proxy_id = character(),
    model = character(),
    trend = numeric(),
    std_error = numeric(),
    df = numeric(),
    ci_low = numeric(),
    ci_high = numeric(),
    statistic = numeric(),
    p_value = numeric()
  )
  
  if (is.null(fit)) {
    return(empty_result)
  }
  
  tryCatch(
    {
      trend_object <- emmeans::emtrends(
        fit,
        specs = "model",
        var = "z_distance"
      )
      
      summary(
        trend_object,
        infer = c(TRUE, TRUE),
        level = confidence_level,
        adjust = "holm"
      ) %>%
        as.data.frame() %>%
        as_tibble() %>%
        transmute(
          proxy_id = proxy_id,
          model = as.character(model),
          trend = z_distance.trend,
          std_error = SE,
          df = df,
          ci_low = lower.CL,
          ci_high = upper.CL,
          statistic = t.ratio,
          p_value = p.value
        )
    },
    error = function(error_condition) empty_result
  )
}


# ── 8. Trait and distance helpers ─────────────────────────────────────────────

read_trait_file <- function(file, required_columns) {
  data <- read_csv(file, show_col_types = FALSE) %>%
    mutate(across(where(is.character), str_trim)) %>%
    filter(
      !if_all(everything(), ~ is.na(.x) | .x == "")
    )
  
  missing_columns <- setdiff(required_columns, names(data))
  
  if (length(missing_columns) > 0L) {
    stop(
      basename(file),
      " is missing: ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }
  
  data <- data %>%
    select(all_of(required_columns))
  
  if (anyDuplicated(data$species) > 0L) {
    stop(
      "Duplicated species in ",
      basename(file),
      ".",
      call. = FALSE
    )
  }
  
  data
}

# Shared type conversion for the expert-coded core and extended trait tables.
type_expert_traits <- function(
    data,
    ordinal_columns,
    nominal_columns,
    label
) {
  output <- data %>%
    mutate(
      log_body_mass_g = as.numeric(log_body_mass_g),
      across(
        all_of(ordinal_columns),
        ~ ordered(str_to_lower(.x), levels = ordinal_levels)
      ),
      across(
        all_of(nominal_columns),
        as.factor
      )
    )
  
  if (any(!complete.cases(output))) {
    stop(
      label,
      " trait table contains invalid values after conversion.",
      call. = FALSE
    )
  }
  
  output
}

make_gower_long <- function(trait_data, proxy_id) {
  distance_matrix <- trait_data %>%
    select(-species) %>%
    daisy(metric = "gower") %>%
    as.matrix()
  
  rownames(distance_matrix) <- trait_data$species
  colnames(distance_matrix) <- trait_data$species
  
  as.data.frame(distance_matrix) %>%
    rownames_to_column("train_species") %>%
    pivot_longer(
      -train_species,
      names_to = "test_species",
      values_to = "distance_raw"
    ) %>%
    mutate(proxy_id = proxy_id, .before = 1)
}


# ── 9. EltonTraits helpers ────────────────────────────────────────────────────

read_elton_traits <- function(file) {
  elton_raw <- read_delim(
    file,
    delim = "\t",
    na = c("", "NA", "NaN"),
    trim_ws = TRUE,
    show_col_types = FALSE
  )
  
  id_column <- pick_column(
    elton_raw,
    c("MSW3_ID", "SW3_ID", "MSW3.ID", "SW3.ID")
  )
  
  species_column <- pick_column(
    elton_raw,
    c("Scientific", "scientificNameStd", "scientificName", "Species")
  )
  
  required_columns <- c(
    id_column,
    species_column,
    elton_core_columns,
    elton_diet_columns
  )
  
  missing_columns <- setdiff(required_columns, names(elton_raw))
  
  if (length(missing_columns) > 0L) {
    stop(
      "MamFuncDat.txt is missing: ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }
  
  numeric_columns <- c(
    "bodymass_g",
    "activity_nocturnal",
    "activity_crepuscular",
    "activity_diurnal",
    "diet_inv",
    "diet_vend",
    "diet_vect",
    "diet_vfish",
    "diet_vunk",
    "diet_scav",
    "diet_fruit",
    "diet_nect",
    "diet_seed",
    "diet_planto"
  )
  
  elton_raw %>%
    select(
      elton_id = all_of(id_column),
      elton_scientific = all_of(species_column),
      forstrat_raw = "ForStrat-Value",
      bodymass_g = "BodyMass-Value",
      activity_nocturnal = "Activity-Nocturnal",
      activity_crepuscular = "Activity-Crepuscular",
      activity_diurnal = "Activity-Diurnal",
      diet_inv = "Diet-Inv",
      diet_vend = "Diet-Vend",
      diet_vect = "Diet-Vect",
      diet_vfish = "Diet-Vfish",
      diet_vunk = "Diet-Vunk",
      diet_scav = "Diet-Scav",
      diet_fruit = "Diet-Fruit",
      diet_nect = "Diet-Nect",
      diet_seed = "Diet-Seed",
      diet_planto = "Diet-PlantO"
    ) %>%
    mutate(
      elton_scientific = clean_species(elton_scientific),
      forstrat_raw = as.character(forstrat_raw),
      across(
        all_of(numeric_columns),
        ~ parse_number(as.character(.x))
      ),
      forstrat_num = parse_number(forstrat_raw),
      log_bodymass_g = log(bodymass_g)
    ) %>%
    filter(!is.na(elton_scientific))
}

build_elton_candidates <- function(study_species_vec) {
  exact <- tibble(
    study_species = study_species_vec,
    candidate = study_species_vec,
    priority = 1L,
    match_source = "exact"
  )
  
  fallbacks <- tribble(
    ~study_species, ~candidate, ~priority, ~match_source,
    "Canis lupus familiaris",
    "Canis lupus",
    2L,
    "manual_fallback_parent_species",
    "Equus ferus przewalskii",
    "Equus ferus",
    2L,
    "manual_fallback_parent_species",
    "Equus ferus przewalskii",
    "Equus caballus",
    3L,
    "manual_fallback_close_species",
    "Giraffa camelopardalis",
    "Giraffa reticulata",
    2L,
    "manual_fallback_close_species"
  ) %>%
    filter(.data$study_species %in% study_species_vec)
  
  bind_rows(exact, fallbacks) %>%
    mutate(candidate_clean = clean_species(candidate))
}

match_elton_traits <- function(file, study_species_vec) {
  elton_traits <- read_elton_traits(file)
  
  diet_columns <- c(
    "diet_inv",
    "diet_vend",
    "diet_vect",
    "diet_vfish",
    "diet_vunk",
    "diet_scav",
    "diet_fruit",
    "diet_nect",
    "diet_seed",
    "diet_planto"
  )
  
  matched <- build_elton_candidates(study_species_vec) %>%
    left_join(
      elton_traits,
      by = c("candidate_clean" = "elton_scientific")
    ) %>%
    mutate(
      has_core = is.finite(log_bodymass_g) &
        !is.na(forstrat_raw) &
        is.finite(activity_nocturnal) &
        is.finite(activity_crepuscular) &
        is.finite(activity_diurnal),
      has_extended = has_core &
        if_all(all_of(diet_columns), is.finite)
    ) %>%
    arrange(study_species, desc(has_core), priority) %>%
    group_by(study_species) %>%
    slice(1) %>%
    ungroup() %>%
    mutate(
      match_status = case_when(
        has_core & match_source == "exact" ~ "exact",
        has_core & match_source != "exact" ~ match_source,
        TRUE ~ "not_found_or_incomplete"
      ),
      matched_name = candidate
    )
  
  incomplete <- matched %>%
    filter(!has_core | !has_extended)
  
  if (nrow(incomplete) > 0L) {
    write_csv(
      matched,
      file.path(csv_dir, "error_elton_matching_attempts.csv")
    )
    
    stop(
      "At least one study species lacks complete EltonTraits data. ",
      "See error_elton_matching_attempts.csv.",
      call. = FALSE
    )
  }
  
  matched
}


# ── 10. TimeTree helpers ──────────────────────────────────────────────────────

build_timetree_candidates <- function(study_species_vec) {
  exact <- tibble(
    study_species = study_species_vec,
    timetree_candidate = study_species_vec,
    priority = 1L,
    match_source = "exact"
  )
  
  replacements <- tribble(
    ~study_species, ~timetree_candidate, ~priority, ~match_source,
    "Canis lupus familiaris",
    "Canis lupus",
    1L,
    "manual_replacement_parent_species",
    "Equus ferus przewalskii",
    "Equus ferus",
    1L,
    "manual_replacement_parent_species",
    "Giraffa camelopardalis",
    "Giraffa reticulata",
    2L,
    "manual_fallback_close_species"
  ) %>%
    filter(.data$study_species %in% study_species_vec)
  
  bind_rows(exact, replacements) %>%
    mutate(candidate_clean = clean_species(timetree_candidate)) %>%
    arrange(study_species, priority)
}

read_timetree_distances <- function(file, study_species_vec) {
  tree <- ape::read.tree(file)
  
  tree$tip.label <- clean_species(tree$tip.label)
  
  if (anyDuplicated(tree$tip.label) > 0L) {
    stop(
      "TimeTree has duplicated tip labels after cleaning.",
      call. = FALSE
    )
  }
  
  tip_table <- tibble(timetree_tip = tree$tip.label)
  
  resolved <- build_timetree_candidates(study_species_vec) %>%
    left_join(
      tip_table,
      by = c("candidate_clean" = "timetree_tip")
    ) %>%
    mutate(found = candidate_clean %in% tree$tip.label) %>%
    arrange(study_species, desc(found), priority) %>%
    group_by(study_species) %>%
    slice(1) %>%
    ungroup() %>%
    mutate(
      match_status = ifelse(found, match_source, "not_found"),
      timetree_label = ifelse(found, candidate_clean, NA_character_)
    )
  
  if (any(!resolved$found)) {
    write_csv(
      resolved,
      file.path(csv_dir, "error_timetree_matching_attempts.csv")
    )
    
    stop(
      "At least one study species has no TimeTree match. ",
      "See error_timetree_matching_attempts.csv.",
      call. = FALSE
    )
  }
  
  duplicated_mapping <- resolved %>%
    count(timetree_label) %>%
    filter(n > 1L)
  
  if (nrow(duplicated_mapping) > 0L) {
    write_csv(
      resolved,
      file.path(csv_dir, "error_timetree_duplicate_mapping.csv")
    )
    
    stop(
      "Multiple study species map to the same TimeTree tip.",
      call. = FALSE
    )
  }
  
  tree_use <- ape::keep.tip(tree, resolved$timetree_label)
  
  rename_map <- setNames(
    resolved$study_species,
    resolved$timetree_label
  )
  
  tree_use$tip.label <- unname(rename_map[tree_use$tip.label])
  
  node_depth <- ape::node.depth.edgelength(tree_use)
  
  tip_depth <- node_depth[seq_along(tree_use$tip.label)]
  
  tree_height <- max(tip_depth, na.rm = TRUE)
  
  # Divergence time = tree height minus the depth of the most recent
  # common ancestor.
  get_divergence_age <- function(species_1, species_2) {
    if (species_1 == species_2) {
      return(0)
    }
    
    mrca <- ape::getMRCA(tree_use, c(species_1, species_2))
    
    tree_height - node_depth[mrca]
  }
  
  divergence_long <- expand_grid(
    train_species = tree_use$tip.label,
    test_species = tree_use$tip.label
  ) %>%
    mutate(
      distance_raw = map2_dbl(
        train_species,
        test_species,
        get_divergence_age
      ),
      proxy_id = "timetree_divergence_mya",
      .before = 1
    )
  
  list(
    tree = tree_use,
    matching = resolved,
    distances = divergence_long,
    tree_height_mya = tree_height
  )
}


# ── 11. Resolve inputs ────────────────────────────────────────────────────────

print_section("1. RESOLVE INPUTS")

must_exist(
  c(
    base_dir,
    models_dir,
    input_dir,
    timetree_file,
    elton_file
  )
)

trait_directories <- unique(
  c(
    getwd(),
    file.path(base_dir, "R_analysis", "H3"),
    file.path(base_dir, "R_scripts", "H3")
  )
)

core_trait_file <- first_existing(
  file.path(trait_directories, core_trait_filename)
)

extended_trait_file <- first_existing(
  file.path(trait_directories, extended_trait_filename)
)

if (is.na(core_trait_file) || is.na(extended_trait_file)) {
  stop(
    "Core or extended expert-coded trait file not found.",
    call. = FALSE
  )
}

model_registry <- tibble(
  model = model_order,
  pairwise_directory = file.path(
    models_dir,
    model_order,
    paste0("Pairwise_", model_order),
    "statistics"
  )
) %>%
  mutate(
    pairwise_file = map_chr(
      pairwise_directory,
      resolve_metric_file,
      candidates = c(
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
    "At least one pairwise model file is missing: ",
    paste(
      model_registry$model[!model_registry$pairwise_exists],
      collapse = ", "
    ),
    call. = FALSE
  )
}

input_registry <- tibble(
  input = c(
    "Core expert-coded traits",
    "Extended expert-coded traits",
    "EltonTraits",
    "TimeTree"
  ),
  file = c(
    core_trait_file,
    extended_trait_file,
    elton_file,
    timetree_file
  )
)

write_csv(
  input_registry,
  file.path(csv_dir, "01_input_file_registry.csv")
)

cat("Output directory:", out_dir, "\n")


# ── 12. Load pairwise performance data ────────────────────────────────────────

print_section("2. LOAD PAIRWISE PERFORMANCE DATA")

read_pairwise_model <- function(model, pairwise_file) {
  data <- read_csv(pairwise_file, show_col_types = FALSE)
  
  missing_columns <- setdiff(required_pairwise_columns, names(data))
  
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

unmapped <- bind_rows(
  pairwise_mapped %>%
    filter(is.na(train_species)) %>%
    distinct(role = "train", dataset = train_dataset),
  pairwise_mapped %>%
    filter(is.na(test_species)) %>%
    distinct(role = "test", dataset = test_dataset)
)

if (nrow(unmapped) > 0L) {
  write_csv(
    unmapped,
    file.path(csv_dir, "error_unmapped_datasets.csv")
  )
  
  stop("Unmapped dataset names found.", call. = FALSE)
}

if (any(!is.finite(pairwise_mapped$macro_f1))) {
  stop("Non-finite Macro-F1 values found.", call. = FALSE)
}

study_species <- sort(
  unique(
    c(
      pairwise_mapped$train_species,
      pairwise_mapped$test_species
    )
  )
)

performance_species_pair <- pairwise_mapped %>%
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
    )
  )

if (nrow(performance_species_pair) != 432L) {
  stop(
    "Expected 432 model-specific species-pair rows, found ",
    nrow(performance_species_pair),
    ".",
    call. = FALSE
  )
}


# ── 13. Expert-coded distances ────────────────────────────────────────────────

print_section("3. BUILD EXPERT-CODED DISTANCES")

core_traits_raw <- read_trait_file(
  core_trait_file,
  required_core_trait_columns
)

extended_traits_raw <- read_trait_file(
  extended_trait_file,
  required_extended_trait_columns
)

core_traits <- type_expert_traits(
  core_traits_raw,
  core_ordinal_traits,
  core_nominal_traits,
  "Core"
)

extended_traits <- type_expert_traits(
  extended_traits_raw,
  extended_ordinal_traits,
  extended_nominal_traits,
  "Extended"
)

missing_expert_species <- unique(
  c(
    setdiff(study_species, core_traits$species),
    setdiff(study_species, extended_traits$species)
  )
)

if (length(missing_expert_species) > 0L) {
  stop(
    "Missing from the expert-coded trait files: ",
    paste(missing_expert_species, collapse = ", "),
    call. = FALSE
  )
}

functional_core_long <- make_gower_long(
  core_traits,
  "functional_core_gower"
)

functional_extended_long <- make_gower_long(
  extended_traits,
  "functional_extended_gower"
)

write_csv(
  core_traits_raw,
  file.path(csv_dir, "02_functional_core_traits.csv")
)

write_csv(
  extended_traits_raw,
  file.path(csv_dir, "03_functional_extended_traits.csv")
)


# ── 14. EltonTraits distances ─────────────────────────────────────────────────

print_section("4. BUILD ELTONTRAITS DISTANCES")

elton_matches <- match_elton_traits(elton_file, study_species)

# Foraging stratum is numeric where possible, otherwise treated as nominal.
elton_matches <- elton_matches %>%
  mutate(
    forstrat_value = if (all(is.finite(forstrat_num))) {
      as.numeric(forstrat_num)
    } else {
      as.factor(forstrat_raw)
    }
  )

elton_core_traits <- elton_matches %>%
  transmute(
    species = study_species,
    log_bodymass_g,
    forstrat_value,
    activity_nocturnal,
    activity_crepuscular,
    activity_diurnal
  )

elton_extended_traits <- elton_matches %>%
  transmute(
    species = study_species,
    log_bodymass_g,
    forstrat_value,
    activity_nocturnal,
    activity_crepuscular,
    activity_diurnal,
    diet_inv,
    diet_vend,
    diet_vect,
    diet_vfish,
    diet_vunk,
    diet_scav,
    diet_fruit,
    diet_nect,
    diet_seed,
    diet_planto
  )

if (
  any(!complete.cases(elton_core_traits)) ||
  any(!complete.cases(elton_extended_traits))
) {
  stop("EltonTraits tables contain missing values.", call. = FALSE)
}

elton_core_long <- make_gower_long(
  elton_core_traits,
  "elton_core_gower"
)

elton_extended_long <- make_gower_long(
  elton_extended_traits,
  "elton_extended_gower"
)

write_csv(
  elton_matches,
  file.path(csv_dir, "04_eltontraits_species_matching.csv")
)

write_csv(
  elton_core_traits,
  file.path(csv_dir, "05_eltontraits_core_traits.csv")
)

write_csv(
  elton_extended_traits,
  file.path(csv_dir, "06_eltontraits_extended_traits.csv")
)


# ── 15. TimeTree distances ────────────────────────────────────────────────────

print_section("5. BUILD TIMETREE DISTANCES")

timetree_object <- read_timetree_distances(timetree_file, study_species)

timetree_long <- timetree_object$distances

write_csv(
  timetree_object$matching,
  file.path(csv_dir, "07_timetree_species_matching.csv")
)

write_csv(
  tibble(tree_height_mya = timetree_object$tree_height_mya),
  file.path(csv_dir, "08_timetree_metadata.csv")
)


# ── 16. Combine and validate distances ────────────────────────────────────────

print_section("6. COMBINE DISTANCE PROXIES")

distance_long <- bind_rows(
  functional_core_long,
  functional_extended_long,
  elton_core_long,
  elton_extended_long,
  timetree_long
) %>%
  left_join(distance_specs, by = "proxy_id") %>%
  mutate(same_species = train_species == test_species)

distance_validation <- distance_long %>%
  group_by(proxy_id, proxy_label) %>%
  summarise(
    n_all_pairs = n(),
    n_cross_species_pairs = sum(!same_species),
    n_missing = sum(!is.finite(distance_raw)),
    minimum = min(distance_raw, na.rm = TRUE),
    maximum = max(distance_raw, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(
  distance_long,
  file.path(csv_dir, "09_all_distance_matrices_long.csv")
)

write_csv(
  distance_validation,
  file.path(csv_dir, "10_distance_validation.csv")
)

if (
  any(distance_validation$n_all_pairs != 81L) ||
  any(distance_validation$n_cross_species_pairs != 72L) ||
  any(distance_validation$n_missing > 0L)
) {
  stop(
    "Distance validation failed. See 10_distance_validation.csv.",
    call. = FALSE
  )
}

# Standardize within proxy so coefficients are comparable across proxies.
pair_distances_scaled <- distance_long %>%
  filter(!same_species) %>%
  group_by(proxy_id) %>%
  mutate(
    distance_mean = mean(distance_raw),
    distance_sd = sd(distance_raw),
    z_distance = (distance_raw - distance_mean) / distance_sd
  ) %>%
  ungroup()

h3_long <- performance_species_pair %>%
  left_join(
    pair_distances_scaled %>%
      select(
        proxy_id,
        proxy_label,
        proxy_role,
        expected_direction,
        train_species,
        test_species,
        distance_raw,
        distance_mean,
        distance_sd,
        z_distance
      ),
    by = c("train_species", "test_species")
  ) %>%
  mutate(
    model = factor(model, levels = model_order),
    ordered_species_pair = factor(ordered_species_pair),
    unordered_species_pair = factor(unordered_species_pair),
    train_species = factor(train_species),
    test_species = factor(test_species),
    proxy_id = factor(
      proxy_id,
      levels = distance_specs$proxy_id
    ),
    proxy_label = factor(
      proxy_label,
      levels = distance_specs$proxy_label
    )
  )

h3_structure <- h3_long %>%
  group_by(proxy_id, proxy_label) %>%
  summarise(
    n_model_observations = n(),
    n_models = n_distinct(model),
    n_ordered_pairs = n_distinct(ordered_species_pair),
    n_species = n_distinct(
      c(
        as.character(train_species),
        as.character(test_species)
      )
    ),
    n_missing = sum(
      !is.finite(mean_macro_f1) | !is.finite(z_distance)
    ),
    .groups = "drop"
  )

write_csv(
  h3_long,
  file.path(csv_dir, "11_h3_all_proxy_mixed_model_data.csv")
)

write_csv(
  h3_structure,
  file.path(csv_dir, "12_h3_proxy_data_structure.csv")
)

if (
  any(h3_structure$n_model_observations != 432L) ||
  any(h3_structure$n_models != 6L) ||
  any(h3_structure$n_ordered_pairs != 72L) ||
  any(h3_structure$n_species != 9L) ||
  any(h3_structure$n_missing > 0L)
) {
  stop(
    "H3 proxy data structure is invalid. See 12_h3_proxy_data_structure.csv.",
    call. = FALSE
  )
}


# ── 17. Descriptive Spearman correlations ─────────────────────────────────────

print_section("7. DESCRIPTIVE SPEARMAN CORRELATIONS")

descriptive_spearman <- map_dfr(
  distance_specs$proxy_id,
  function(current_proxy) {
    current_data <- h3_long %>%
      filter(as.character(proxy_id) == current_proxy)
    
    pooled <- safe_spearman(
      current_data$distance_raw,
      current_data$mean_macro_f1
    ) %>%
      mutate(
        proxy_id = current_proxy,
        model = "All models pooled",
        analysis = "pooled_descriptive",
        n_unique_ordered_pairs = n_distinct(
          current_data$ordered_species_pair
        ),
        .before = 1
      )
    
    by_model <- current_data %>%
      group_by(model) %>%
      group_modify(
        ~ safe_spearman(
          .x$distance_raw,
          .x$mean_macro_f1
        )
      ) %>%
      ungroup() %>%
      mutate(
        proxy_id = current_proxy,
        model = as.character(model),
        analysis = "model_specific_descriptive",
        n_unique_ordered_pairs = n,
        .before = 1
      )
    
    bind_rows(pooled, by_model)
  }
) %>%
  left_join(distance_specs, by = "proxy_id") %>%
  group_by(analysis) %>%
  mutate(
    p_value_holm = p.adjust(p_value_naive, method = "holm")
  ) %>%
  ungroup()

write_csv(
  descriptive_spearman,
  file.path(csv_dir, "13_descriptive_spearman_correlations.csv")
)

print_table(
  descriptive_spearman %>%
    filter(analysis == "pooled_descriptive") %>%
    select(
      proxy_label,
      n,
      rho,
      ci_low_naive,
      ci_high_naive,
      p_value_naive
    )
)


# ── 18. Fit mixed-effects models for every proxy ──────────────────────────────

print_section("8. FIT MIXED-EFFECTS MODELS")

# Primary dyadic structure used in script 01.
formula_full_null <- mean_macro_f1 ~ model +
  (1 | unordered_species_pair) +
  (1 | ordered_species_pair) +
  (1 | train_species) +
  (1 | test_species)

formula_full_additive <- mean_macro_f1 ~ z_distance + model +
  (1 | unordered_species_pair) +
  (1 | ordered_species_pair) +
  (1 | train_species) +
  (1 | test_species)

formula_full_interaction <- mean_macro_f1 ~ z_distance * model +
  (1 | unordered_species_pair) +
  (1 | ordered_species_pair) +
  (1 | train_species) +
  (1 | test_species)

# Sensitivity without the reciprocal-dyad intercept.
formula_directional_null <- mean_macro_f1 ~ model +
  (1 | ordered_species_pair) +
  (1 | train_species) +
  (1 | test_species)

formula_directional_additive <- mean_macro_f1 ~ z_distance + model +
  (1 | ordered_species_pair) +
  (1 | train_species) +
  (1 | test_species)

# Reduced pair-only sensitivity.
formula_pair_only_null <- mean_macro_f1 ~ model +
  (1 | ordered_species_pair)

formula_pair_only_additive <- mean_macro_f1 ~ z_distance + model +
  (1 | ordered_species_pair)

get_proxy_data <- function(current_proxy) {
  h3_long %>%
    filter(as.character(proxy_id) == current_proxy) %>%
    droplevels()
}

# ML fits are used for LRTs; REML fits are used for coefficients.
fit_proxy_models <- function(current_proxy) {
  cat("Fitting:", current_proxy, "\n")
  
  current_data <- get_proxy_data(current_proxy)
  
  fits <- list(
    full_null_ml = safe_lmer(
      formula_full_null,
      current_data,
      FALSE,
      "full_null_ml"
    ),
    full_additive_ml = safe_lmer(
      formula_full_additive,
      current_data,
      FALSE,
      "full_additive_ml"
    ),
    full_additive_reml = safe_lmer(
      formula_full_additive,
      current_data,
      TRUE,
      "full_additive_reml"
    )
  )
  
  fits$full_interaction_ml <- if (run_interaction_models) {
    safe_lmer(
      formula_full_interaction,
      current_data,
      FALSE,
      "full_interaction_ml"
    )
  } else {
    skipped_model("full_interaction_ml")
  }
  
  fits$full_interaction_reml <- if (run_interaction_models) {
    safe_lmer(
      formula_full_interaction,
      current_data,
      TRUE,
      "full_interaction_reml"
    )
  } else {
    skipped_model("full_interaction_reml")
  }
  
  fits$directional_null_ml <- if (run_directional_sensitivity) {
    safe_lmer(
      formula_directional_null,
      current_data,
      FALSE,
      "directional_null_ml"
    )
  } else {
    skipped_model("directional_null_ml")
  }
  
  fits$directional_additive_ml <- if (run_directional_sensitivity) {
    safe_lmer(
      formula_directional_additive,
      current_data,
      FALSE,
      "directional_additive_ml"
    )
  } else {
    skipped_model("directional_additive_ml")
  }
  
  fits$directional_additive_reml <- if (run_directional_sensitivity) {
    safe_lmer(
      formula_directional_additive,
      current_data,
      TRUE,
      "directional_additive_reml"
    )
  } else {
    skipped_model("directional_additive_reml")
  }
  
  fits$pair_only_null_ml <- if (run_pair_only_sensitivity) {
    safe_lmer(
      formula_pair_only_null,
      current_data,
      FALSE,
      "pair_only_null_ml"
    )
  } else {
    skipped_model("pair_only_null_ml")
  }
  
  fits$pair_only_additive_ml <- if (run_pair_only_sensitivity) {
    safe_lmer(
      formula_pair_only_additive,
      current_data,
      FALSE,
      "pair_only_additive_ml"
    )
  } else {
    skipped_model("pair_only_additive_ml")
  }
  
  fits$pair_only_additive_reml <- if (run_pair_only_sensitivity) {
    safe_lmer(
      formula_pair_only_additive,
      current_data,
      TRUE,
      "pair_only_additive_reml"
    )
  } else {
    skipped_model("pair_only_additive_reml")
  }
  
  fits
}

model_store <- set_names(
  map(distance_specs$proxy_id, fit_proxy_models),
  distance_specs$proxy_id
)

get_fit <- function(current_proxy, label) {
  model_store[[current_proxy]][[label]]$fit
}

proxy_ids <- distance_specs$proxy_id


# ── 19. Collect model results ─────────────────────────────────────────────────

print_section("9. COLLECT MODEL RESULTS")

fixed_effects_all <- map_dfr(
  proxy_ids,
  ~ extract_fixed_table(
    get_fit(.x, "full_additive_reml"),
    .x,
    "full_additive_reml"
  )
)

random_effects_all <- map_dfr(
  proxy_ids,
  ~ extract_random_table(
    get_fit(.x, "full_additive_reml"),
    .x,
    "full_additive_reml"
  )
)

model_diagnostics <- map_dfr(
  proxy_ids,
  function(current_proxy) {
    map_dfr(
      model_store[[current_proxy]],
      extract_diagnostics,
      proxy_id = current_proxy
    )
  }
)

model_comparisons <- map_dfr(
  proxy_ids,
  function(current_proxy) {
    bind_rows(
      compare_nested_models(
        get_fit(current_proxy, "full_null_ml"),
        get_fit(current_proxy, "full_additive_ml"),
        current_proxy,
        "full_null_vs_full_additive"
      ),
      compare_nested_models(
        get_fit(current_proxy, "full_additive_ml"),
        get_fit(current_proxy, "full_interaction_ml"),
        current_proxy,
        "full_additive_vs_full_interaction"
      ),
      compare_nested_models(
        get_fit(current_proxy, "directional_null_ml"),
        get_fit(current_proxy, "directional_additive_ml"),
        current_proxy,
        "directional_null_vs_directional_additive"
      ),
      compare_nested_models(
        get_fit(current_proxy, "pair_only_null_ml"),
        get_fit(current_proxy, "pair_only_additive_ml"),
        current_proxy,
        "pair_only_null_vs_pair_only_additive"
      )
    )
  }
) %>%
  group_by(comparison) %>%
  mutate(
    p_value_holm = p.adjust(p_value, method = "holm")
  ) %>%
  ungroup()

r2_target_models <- c(
  "full_null_ml",
  "full_additive_ml",
  "full_interaction_ml",
  "directional_null_ml",
  "directional_additive_ml",
  "pair_only_null_ml",
  "pair_only_additive_ml"
)

model_r2 <- map_dfr(
  proxy_ids,
  function(current_proxy) {
    map_dfr(
      r2_target_models,
      ~ extract_r2(
        get_fit(current_proxy, .x),
        current_proxy,
        .x
      )
    )
  }
)

interaction_slopes <- map_dfr(
  proxy_ids,
  ~ extract_interaction_slopes(
    get_fit(.x, "full_interaction_reml"),
    .x
  )
)

directional_effects <- map_dfr(
  proxy_ids,
  ~ extract_fixed_table(
    get_fit(.x, "directional_additive_reml"),
    .x,
    "directional_additive_reml"
  ) %>%
    filter(term == "z_distance")
)

pair_only_effects <- map_dfr(
  proxy_ids,
  ~ extract_fixed_table(
    get_fit(.x, "pair_only_additive_reml"),
    .x,
    "pair_only_additive_reml"
  ) %>%
    filter(term == "z_distance")
)

primary_effects <- fixed_effects_all %>%
  filter(term == "z_distance") %>%
  left_join(distance_specs, by = "proxy_id") %>%
  left_join(
    model_diagnostics %>%
      filter(model_name == "full_additive_reml") %>%
      select(
        proxy_id,
        singular,
        optimizer_convergence_code,
        convergence_messages,
        warnings,
        error
      ),
    by = "proxy_id"
  ) %>%
  mutate(
    converged =
      (is.na(optimizer_convergence_code) |
         optimizer_convergence_code == 0L) &
      (is.na(error) | error == "") &
      (is.na(convergence_messages) | convergence_messages == "") &
      (is.na(warnings) | warnings == ""),
    valid_primary_model = converged & !singular,
    ci_excludes_zero = is.finite(ci_low) &
      is.finite(ci_high) &
      (ci_low > 0 | ci_high < 0),
    p_value_holm = p.adjust(p_value, method = "holm")
  )

# Added fixed-effect contribution as change in marginal Nakagawa R2.
r2_change <- model_r2 %>%
  select(proxy_id, model_name, marginal_r2, conditional_r2) %>%
  pivot_wider(
    names_from = model_name,
    values_from = c(marginal_r2, conditional_r2)
  ) %>%
  transmute(
    proxy_id,
    delta_marginal_r2_full =
      marginal_r2_full_additive_ml - marginal_r2_full_null_ml,
    delta_marginal_r2_interaction =
      marginal_r2_full_interaction_ml - marginal_r2_full_additive_ml,
    delta_marginal_r2_directional =
      marginal_r2_directional_additive_ml - marginal_r2_directional_null_ml,
    delta_marginal_r2_pair_only =
      marginal_r2_pair_only_additive_ml - marginal_r2_pair_only_null_ml
  ) %>%
  left_join(distance_specs, by = "proxy_id")

write_csv(
  fixed_effects_all,
  file.path(csv_dir, "14_full_additive_fixed_effects_all_proxies.csv")
)

write_csv(
  random_effects_all,
  file.path(csv_dir, "15_full_additive_random_effects_all_proxies.csv")
)

write_csv(
  model_diagnostics,
  file.path(csv_dir, "16_model_diagnostics_all_proxies.csv")
)

write_csv(
  model_comparisons,
  file.path(csv_dir, "17_likelihood_ratio_model_comparisons.csv")
)

write_csv(
  model_r2,
  file.path(csv_dir, "18_nakagawa_r2_all_models.csv")
)

write_csv(
  r2_change,
  file.path(csv_dir, "19_added_distance_delta_marginal_r2.csv")
)

write_csv(
  interaction_slopes,
  file.path(csv_dir, "20_interaction_model_specific_slopes.csv")
)

write_csv(
  directional_effects,
  file.path(csv_dir, "21_directional_pair_sensitivity_effects.csv")
)

write_csv(
  pair_only_effects,
  file.path(csv_dir, "22_pair_only_sensitivity_effects.csv")
)

write_csv(
  primary_effects,
  file.path(csv_dir, "23_primary_mixed_effect_summary.csv")
)

print_table(
  primary_effects %>%
    select(
      proxy_label,
      estimate,
      std_error,
      ci_low,
      ci_high,
      p_value,
      p_value_holm,
      singular,
      converged
    )
)


# ── 20. Leave-one-species-out mixed models ────────────────────────────────────

print_section("10. LEAVE-ONE-SPECIES-OUT MIXED MODELS")

# Refit the additive model with all pairs involving one species removed.
fit_loso_model <- function(proxy_id, dropped_species) {
  current_proxy <- proxy_id
  current_species <- dropped_species
  
  loso_data <- get_proxy_data(current_proxy) %>%
    filter(
      as.character(train_species) != current_species,
      as.character(test_species) != current_species
    ) %>%
    droplevels()
  
  model_label <- paste0("loso_", current_species)
  
  loso_fit <- safe_lmer(
    formula_full_additive,
    loso_data,
    TRUE,
    model_label
  )
  
  loso_diagnostic <- extract_diagnostics(loso_fit, current_proxy)
  
  loso_fixed <- extract_fixed_table(
    loso_fit$fit,
    current_proxy,
    model_label
  ) %>%
    filter(term == "z_distance")
  
  if (nrow(loso_fixed) == 0L) {
    loso_fixed <- tibble(
      proxy_id = current_proxy,
      model_name = model_label,
      term = "z_distance",
      estimate = NA_real_,
      std_error = NA_real_,
      df = NA_real_,
      statistic = NA_real_,
      p_value = NA_real_,
      ci_low = NA_real_,
      ci_high = NA_real_
    )
  }
  
  loso_fixed %>%
    transmute(
      proxy_id,
      dropped_species = current_species,
      n_observations = nrow(loso_data),
      n_unordered_pairs = n_distinct(loso_data$unordered_species_pair),
      n_ordered_pairs = n_distinct(loso_data$ordered_species_pair),
      estimate,
      std_error,
      df,
      statistic,
      p_value,
      ci_low,
      ci_high,
      singular = loso_diagnostic$singular[[1]],
      optimizer_convergence_code =
        loso_diagnostic$optimizer_convergence_code[[1]],
      convergence_messages = loso_diagnostic$convergence_messages[[1]],
      warnings = loso_diagnostic$warnings[[1]],
      error = loso_diagnostic$error[[1]]
    )
}

loso_results <- tibble()
loso_summary <- tibble()

if (run_loso_mixed_models) {
  loso_results <- expand_grid(
    proxy_id = proxy_ids,
    dropped_species = study_species
  ) %>%
    pmap_dfr(fit_loso_model) %>%
    left_join(
      primary_effects %>%
        select(
          proxy_id,
          full_estimate = estimate,
          full_ci_low = ci_low,
          full_ci_high = ci_high
        ),
      by = "proxy_id"
    ) %>%
    left_join(distance_specs, by = "proxy_id") %>%
    mutate(
      delta_estimate = estimate - full_estimate,
      abs_delta_estimate = abs(delta_estimate),
      sign_flip = is.finite(estimate) &
        is.finite(full_estimate) &
        sign(estimate) != sign(full_estimate),
      ci_excludes_zero = is.finite(ci_low) &
        is.finite(ci_high) &
        (ci_low > 0 | ci_high < 0)
    )
  
  loso_summary <- loso_results %>%
    group_by(proxy_id, proxy_label, proxy_role) %>%
    summarise(
      full_estimate = first(full_estimate),
      minimum_estimate = min(estimate, na.rm = TRUE),
      maximum_estimate = max(estimate, na.rm = TRUE),
      maximum_absolute_change = max(abs_delta_estimate, na.rm = TRUE),
      n_sign_flips = sum(sign_flip, na.rm = TRUE),
      n_singular = sum(singular, na.rm = TRUE),
      most_influential_species = dropped_species[
        which.max(abs_delta_estimate)
      ],
      estimate_without_most_influential = estimate[
        which.max(abs_delta_estimate)
      ],
      .groups = "drop"
    )
}

write_csv(
  loso_results,
  file.path(csv_dir, "24_loso_mixed_effect_results.csv")
)

write_csv(
  loso_summary,
  file.path(csv_dir, "25_loso_mixed_effect_summary.csv")
)


# ── 21. Figures ───────────────────────────────────────────────────────────────

print_section("11. FIGURES")

forest_plot_data <- primary_effects %>%
  left_join(
    model_comparisons %>%
      filter(comparison == "full_null_vs_full_additive") %>%
      select(
        proxy_id,
        lrt_chisq = likelihood_ratio_chisq,
        lrt_p_value = p_value,
        lrt_p_value_holm = p_value_holm,
        additive_aic = aic_full,
        null_aic = aic_reduced
      ),
    by = "proxy_id"
  ) %>%
  left_join(
    r2_change %>%
      select(proxy_id, delta_marginal_r2_full),
    by = "proxy_id"
  ) %>%
  mutate(
    plot_role = if_else(
      as.character(proxy_id) == "functional_core_gower",
      "Primary H3 distance",
      "Sensitivity proxy"
    ),
    plot_role = factor(
      plot_role,
      levels = forest_role_levels
    ),
    display_label = factor(
      proxy_label,
      levels = rev(distance_specs$proxy_label)
    ),
    value_label = paste0(
      fmt_num(estimate, 3),
      " ",
      fmt_ci(ci_low, ci_high, 3)
    ),
    value_fontface = if_else(
      as.character(proxy_id) == "functional_core_gower",
      "bold",
      "plain"
    )
  )

write_csv(
  forest_plot_data,
  file.path(csv_dir, "26_forest_plot_data.csv")
)

forest_data_range <- range(
  c(
    forest_plot_data$ci_low,
    forest_plot_data$ci_high,
    0
  ),
  na.rm = TRUE
)

forest_data_span <- diff(forest_data_range)

forest_breaks <- pretty(forest_data_range, n = 5)

forest_label_x <- forest_data_range[[2]] + 0.08 * forest_data_span

forest_x_limits <- c(
  forest_data_range[[1]] - 0.06 * forest_data_span,
  forest_label_x + 0.65 * forest_data_span
)

# Main forest plot: standardized coefficient by distance proxy.
# The primary core distance is highlighted in black; all sensitivity proxies
# are shown in gray. Direction is communicated by position relative to zero,
# not by color or significance categories.
plot_forest <- ggplot(
  forest_plot_data,
  aes(
    x = estimate,
    y = display_label
  )
) +
  geom_vline(
    xintercept = 0,
    linetype = 2,
    colour = "grey45",
    linewidth = 0.45
  ) +
  geom_errorbarh(
    aes(
      xmin = ci_low,
      xmax = ci_high,
      colour = plot_role
    ),
    height = 0.12,
    linewidth = 0.65,
    show.legend = FALSE
  ) +
  geom_point(
    aes(
      fill = plot_role,
      colour = plot_role,
      size = plot_role
    ),
    shape = 21,
    stroke = 0.55,
    show.legend = FALSE
  ) +
  scale_colour_manual(
    values = forest_line_colors,
    guide = "none"
  ) +
  scale_fill_manual(
    values = forest_fill_colors,
    guide = "none"
  ) +
  scale_size_manual(
    values = forest_point_sizes,
    guide = "none"
  ) +
  scale_x_continuous(
    breaks = forest_breaks,
    limits = if (forest_show_value_labels) {
      forest_x_limits
    } else {
      NULL
    },
    expand = expansion(
      mult = if (forest_show_value_labels) {
        c(0.01, 0.01)
      } else {
        c(0.10, 0.10)
      }
    )
  ) +
  labs(
    x = axis_label_beta,
    y = NULL
  ) +
  theme_h3(base_size = 12) +
  theme(
    panel.background = element_rect(
      fill = "white",
      colour = NA
    ),
    plot.background = element_rect(
      fill = "white",
      colour = NA
    ),
    panel.grid.major.x = element_line(
      colour = "grey92",
      linewidth = 0.3
    ),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "none"
  )

if (forest_show_value_labels) {
  plot_forest <- plot_forest +
    geom_text(
      aes(
        x = forest_label_x,
        y = display_label,
        label = value_label,
        fontface = value_fontface
      ),
      hjust = 0,
      vjust = 0.5,
      size = forest_label_size,
      colour = "grey15",
      show.legend = FALSE
    )
}

save_plot_png(
  plot_forest,
  "01_distance_proxy_mixed_effect_forest_plot.png",
  width = 9.2,
  height = 4.2
)

descriptive_pair_mean <- h3_long %>%
  group_by(
    proxy_id,
    proxy_label,
    train_species,
    test_species,
    distance_raw
  ) %>%
  summarise(
    mean_macro_f1_across_models = mean(mean_macro_f1),
    .groups = "drop"
  ) %>%
  mutate(
    proxy_label = factor(
      proxy_label,
      levels = distance_specs$proxy_label
    )
  )

write_csv(
  descriptive_pair_mean,
  file.path(csv_dir, "27_descriptive_scatter_plot_data.csv")
)

# Descriptive distance-performance pattern by proxy.
plot_descriptive <- ggplot(
  descriptive_pair_mean,
  aes(
    x = distance_raw,
    y = mean_macro_f1_across_models
  )
) +
  geom_smooth(
    method = "lm",
    formula = y ~ x,
    se = TRUE,
    colour = "grey25",
    fill = "grey80",
    linewidth = 0.7,
    alpha = 0.5
  ) +
  geom_point(
    shape = 21,
    colour = "grey15",
    fill = "grey55",
    stroke = 0.3,
    size = 1.8,
    alpha = 0.8
  ) +
  facet_wrap(
    ~ proxy_label,
    scales = "free_x",
    ncol = 2
  ) +
  guides(colour = "none") +
  scale_x_continuous(expand = expansion(mult = 0.04)) +
  scale_y_continuous(expand = expansion(mult = 0.06)) +
  labs(
    x = axis_label_distance,
    y = axis_label_performance
  ) +
  theme_h3() +
  theme(
    panel.spacing.x = grid::unit(1.3, "lines"),
    panel.spacing.y = grid::unit(1.0, "lines")
  )

save_plot_png(
  plot_descriptive,
  "02_descriptive_distance_performance_scatter.png",
  width = 10,
  height = 8
)

# Leave-one-species-out sensitivity; dashed lines show full-data estimates.
if (run_loso_mixed_models && nrow(loso_results) > 0L) {
  loso_plot_data <- loso_results %>%
    mutate(
      dropped_species_short = short_species(dropped_species),
      proxy_label = factor(
        proxy_label,
        levels = distance_specs$proxy_label
      )
    )
  
  write_csv(
    loso_plot_data,
    file.path(csv_dir, "28_loso_plot_data.csv")
  )
  
  loso_reference <- primary_effects %>%
    select(
      proxy_id,
      proxy_label,
      full_estimate = estimate
    ) %>%
    mutate(
      proxy_label = factor(
        proxy_label,
        levels = distance_specs$proxy_label
      )
    )
  
  plot_loso <- ggplot(
    loso_plot_data,
    aes(
      x = estimate,
      y = reorder(dropped_species_short, estimate)
    )
  ) +
    geom_vline(
      xintercept = 0,
      colour = "grey40",
      linewidth = 0.4
    ) +
    geom_vline(
      data = loso_reference,
      aes(xintercept = full_estimate),
      linetype = "dashed",
      colour = "#D55E00",
      linewidth = 0.5
    ) +
    geom_errorbarh(
      aes(
        xmin = ci_low,
        xmax = ci_high
      ),
      height = 0.12,
      linewidth = 0.5,
      colour = "grey30"
    ) +
    geom_point(
      size = 1.9,
      colour = "grey15"
    ) +
    facet_wrap(
      ~ proxy_label,
      scales = "free_y",
      ncol = 2
    ) +
    labs(
      x = axis_label_beta,
      y = "Dropped species"
    ) +
    theme_h3() +
    theme(
      axis.text.y = element_text(size = 8)
    )
  
  save_plot_png(
    plot_loso,
    "03_loso_mixed_effect_sensitivity.png",
    width = 10,
    height = 9
  )
}

# Classifier-specific slopes from the interaction models.
if (nrow(interaction_slopes) > 0L) {
  interaction_plot_data <- interaction_slopes %>%
    left_join(distance_specs, by = "proxy_id") %>%
    mutate(
      model = factor(model, levels = rev(model_order)),
      proxy_label = factor(
        proxy_label,
        levels = distance_specs$proxy_label
      )
    )
  
  plot_interaction_slopes <- ggplot(
    interaction_plot_data,
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
      linewidth = 0.5,
      size = 0.35
    ) +
    facet_wrap(~ proxy_label, ncol = 2) +
    scale_colour_manual(
      values = model_colors,
      guide = "none"
    ) +
    labs(
      x = axis_label_beta,
      y = NULL
    ) +
    theme_h3()
  
  save_plot_png(
    plot_interaction_slopes,
    "04_interaction_model_specific_slopes.png",
    width = 10,
    height = 7
  )
}


# ── 22. Reproducibility exports ───────────────────────────────────────────────

print_section("12. REPRODUCIBILITY EXPORTS")

n_valid_models <- sum(
  forest_plot_data$valid_primary_model,
  na.rm = TRUE
)

n_ci_excludes_zero <- sum(
  forest_plot_data$valid_primary_model &
    forest_plot_data$ci_excludes_zero,
  na.rm = TRUE
)

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
  file.path(csv_dir, "29_model_formula_registry.csv")
)

run_manifest <- tibble(
  item = c(
    "script",
    "run_time",
    "base_dir",
    "input_dir",
    "output_dir",
    "core_trait_file",
    "extended_trait_file",
    "elton_file",
    "timetree_file",
    "random_seed",
    "confidence_level",
    "singularity_tolerance",
    "n_distance_proxies",
    "run_interaction_models",
    "run_directional_sensitivity",
    "run_pair_only_sensitivity",
    "run_loso_mixed_models",
    "n_valid_primary_models",
    "n_models_with_ci_excluding_zero"
  ),
  value = c(
    "02_H3_distance_proxy_mixed_models_final.R",
    as.character(Sys.time()),
    base_dir,
    input_dir,
    out_dir,
    core_trait_file,
    extended_trait_file,
    elton_file,
    timetree_file,
    "42",
    as.character(confidence_level),
    as.character(singularity_tolerance),
    as.character(nrow(distance_specs)),
    as.character(run_interaction_models),
    as.character(run_directional_sensitivity),
    as.character(run_pair_only_sensitivity),
    as.character(run_loso_mixed_models),
    as.character(n_valid_models),
    as.character(n_ci_excludes_zero)
  )
)

write_csv(
  run_manifest,
  file.path(csv_dir, "30_run_manifest.csv")
)

model_summary_lines <- c(
  "H3 DISTANCE-PROXY MIXED MODEL SUMMARIES",
  ""
)

for (current_proxy in proxy_ids) {
  model_summary_lines <- c(
    model_summary_lines,
    strrep("=", 72),
    current_proxy,
    strrep("=", 72),
    "",
    "PRIMARY DYADIC ADDITIVE REML MODEL",
    capture.output(summary(get_fit(current_proxy, "full_additive_reml"))),
    "",
    "PRIMARY DYADIC INTERACTION REML MODEL",
    capture.output(summary(get_fit(current_proxy, "full_interaction_reml"))),
    "",
    "DIRECTIONAL-PAIR ADDITIVE REML SENSITIVITY MODEL",
    capture.output(summary(get_fit(current_proxy, "directional_additive_reml"))),
    "",
    "PAIR-ONLY ADDITIVE REML SENSITIVITY MODEL",
    capture.output(summary(get_fit(current_proxy, "pair_only_additive_reml"))),
    ""
  )
}

model_summary_lines <- c(
  model_summary_lines,
  strrep("=", 72),
  "ML MODEL COMPARISONS",
  strrep("=", 72),
  capture.output(print_table(model_comparisons))
)

write_txt(
  file.path(txt_dir, "01_model_summaries.txt"),
  model_summary_lines
)

write_txt(
  file.path(txt_dir, "02_session_info.txt"),
  capture.output(sessionInfo())
)

# Written last so the index covers all other files.
output_index <- tibble(
  folder = c(
    rep("csv", length(list.files(csv_dir))),
    rep("plots", length(list.files(plots_dir))),
    rep("txt", length(list.files(txt_dir)))
  ),
  file = c(
    list.files(csv_dir),
    list.files(plots_dir),
    list.files(txt_dir)
  )
)

write_csv(
  output_index,
  file.path(csv_dir, "99_output_file_index.csv")
)


# ── 23. Console summary ───────────────────────────────────────────────────────

print_section("13. FINAL CONSOLE SUMMARY")

print_table(
  forest_plot_data %>%
    select(
      proxy_label,
      estimate,
      ci_low,
      ci_high,
      p_value,
      p_value_holm,
      lrt_p_value,
      lrt_p_value_holm,
      delta_marginal_r2_full,
      singular,
      converged
    )
)

cat(
  "\nValid primary dyadic models:",
  n_valid_models,
  "of",
  nrow(forest_plot_data),
  "\n"
)

cat(
  "Primary models with 95% CI excluding zero:",
  n_ci_excludes_zero,
  "\n"
)

if (nrow(loso_summary) > 0L) {
  cat(
    "LOSO sign flips across proxies:",
    sum(loso_summary$n_sign_flips, na.rm = TRUE),
    "\n"
  )
}

cat(
  "\nOutputs written to:\n",
  "- ", csv_dir, "\n",
  "- ", plots_dir, "\n",
  "- ", txt_dir, "\n",
  sep = ""
)

# End of script
