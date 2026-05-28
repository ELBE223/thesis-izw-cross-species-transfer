# =============================================================================
# H3 sensitivity analysis: TimeTree and EltonTraits distances
# =============================================================================
# Author : Lucas Beseler
# Date   : 2026-05-17
#
# Purpose:
# - Reuse the pairwise H3 transfer outputs from the model folders.
# - Build alternative distance matrices for H3 sensitivity checks:
#   1) EltonTraits minimal Gower distance:
#      log body mass + foraging stratum + activity pattern.
#   2) EltonTraits extended Gower distance:
#      minimal Elton traits + diet variables.
#   3) TimeTree divergence-time distance from a Newick tree.
# - Relate these distances to pairwise cross-species transfer performance.
#
# Notes:
# - Main H3 should remain the expert-coded functional-biomechanical distance.
# - This script is a sensitivity check only.
# - Canis lupus familiaris is matched to Canis lupus where needed.
# - Giraffa camelopardalis is matched exactly if present in the final Newick tree.
# - Giraffa reticulata is kept only as a documented fallback if the exported
#   TimeTree file does not contain Giraffa camelopardalis.
# - Equus ferus przewalskii is mapped to Equus ferus for TimeTree.
#
# Output:
# - The output folder is named after this script:
#   Output_R/H3_timetree_elton_sensitivity/
#
# =============================================================================

# ── 1. Setup ──────────────────────────────────────────────────────────────────
library(readr)
library(dplyr)
library(tidyr)
library(tibble)
library(purrr)
library(stringr)
library(cluster)
library(ggplot2)
library(ape)
library(scales)

# ── 2. User settings ──────────────────────────────────────────────────────────
show_p_values <- TRUE
plot_styles <- c("bw", "color")

# ── 3. Paths ──────────────────────────────────────────────────────────────────
script_name   <- "H3_timetree_elton_sensitivity.R"
script_stub   <- tools::file_path_sans_ext(script_name)

base_dir      <- "/Volumes/Z Slim/11_05_2026_Data_Analysis"
models_dir    <- file.path(base_dir, "Models")
input_dir     <- file.path(base_dir, "TimeTree:Elton")
output_r_root <- file.path(base_dir, "Output_R")
out_dir       <- file.path(output_r_root, script_stub)

timetree_file <- file.path(input_dir, "timetree_species_list.nwk")
elton_file    <- file.path(input_dir, "MamFuncDat.txt")

csv_dir   <- file.path(out_dir, "csv")
plots_dir <- file.path(out_dir, "plots")
txt_dir   <- file.path(out_dir, "txt")

for (d in c(output_r_root, out_dir, csv_dir, plots_dir, txt_dir)) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

# ── 4. Constants ──────────────────────────────────────────────────────────────
model_order <- c("CNN", "ResNet", "HYDRA", "MultiRocket", "RF", "LGBM")
metrics <- c("accuracy", "macro_recall", "macro_precision", "macro_f1")
required_pairwise_cols <- c("pair_id", "train_dataset", "test_dataset", metrics)

elton_minimal_cols <- c(
  "BodyMass-Value",
  "ForStrat-Value",
  "Activity-Nocturnal",
  "Activity-Crepuscular",
  "Activity-Diurnal"
)

elton_diet_cols <- c(
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

# ── 5. Model input registry ───────────────────────────────────────────────────
model_files <- tribble(
  ~model,         ~pairwise_dir,
  "RF",          file.path(models_dir, "RF",          "Pairwise_RF",          "statistics"),
  "LGBM",        file.path(models_dir, "LGBM",        "Pairwise_LGBM",        "statistics"),
  "CNN",         file.path(models_dir, "CNN",         "Pairwise_CNN",         "statistics"),
  "ResNet",      file.path(models_dir, "ResNet",      "Pairwise_ResNet",      "statistics"),
  "HYDRA",       file.path(models_dir, "HYDRA",       "Pairwise_HYDRA",       "statistics"),
  "MultiRocket", file.path(models_dir, "MultiRocket", "Pairwise_MultiRocket", "statistics")
)

# ── 6. Helpers ────────────────────────────────────────────────────────────────
must_exist <- function(x) {
  missing <- x[!file.exists(x)]
  if (length(missing) > 0) {
    stop("Missing files:\n", paste(missing, collapse = "\n"))
  }
}

resolve_existing_file <- function(dir_path, candidates) {
  hits <- file.path(dir_path, candidates)
  hit <- hits[file.exists(hits)][1]
  if (length(hit) == 0 || is.na(hit)) return(hits[1])
  hit
}

pick_col <- function(df, candidates) {
  hit <- intersect(candidates, names(df))
  if (length(hit) == 0) {
    stop("None of these columns found: ", paste(candidates, collapse = ", "))
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

fmt_num <- function(x, digits = 3) {
  ifelse(is.na(x), "NA", formatC(x, format = "f", digits = digits))
}

fmt_p <- function(x) {
  ifelse(is.na(x), "NA", format.pval(x, digits = 3, eps = 0.001))
}

fmt_ci <- function(low, high, digits = 3) {
  if (is.na(low) || is.na(high)) return("[NA, NA]")
  paste0("[", fmt_num(low, digits), ", ", fmt_num(high, digits), "]")
}

append_section <- function(lines, title, obj = NULL) {
  header <- paste0(strrep("=", 18), " ", title, " ", strrep("=", 18))
  if (is.null(obj)) return(c(lines, "", header, ""))
  c(lines, "", header, capture.output(print(obj, n = Inf)))
}

write_txt_report <- function(file, lines) {
  writeLines(enc2utf8(lines), con = file, useBytes = TRUE)
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

make_gower_long <- function(mat, value_name) {
  as.data.frame(mat) %>%
    rownames_to_column("train_species") %>%
    pivot_longer(-train_species, names_to = "test_species", values_to = value_name)
}

short_species <- function(x) {
  parts <- str_split(as.character(x), " ")
  vapply(parts, function(p) {
    if (length(p) >= 2) paste0(substr(p[1], 1, 1), ". ", paste(p[-1], collapse = " ")) else p[1]
  }, character(1))
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

scale_01 <- function(x) {
  rng <- range(x[is.finite(x)], na.rm = TRUE)
  if (!is.finite(rng[1]) || diff(rng) == 0) return(rep(NA_real_, length(x)))
  (x - rng[1]) / diff(rng)
}

# ── 7. Resolve model files ────────────────────────────────────────────────────
must_exist(c(models_dir, input_dir, timetree_file, elton_file))

model_files <- model_files %>%
  mutate(
    pairwise_file = map_chr(
      pairwise_dir,
      ~ resolve_existing_file(.x, c(
        "metrics_all.csv",
        "pairwise_metrics_all.csv",
        "pairwise_summary_metrics.csv"
      ))
    ),
    pairwise_exists = file.exists(pairwise_file),
    include_model = pairwise_exists,
    missing_reason = ifelse(pairwise_exists, "available", "missing pairwise file")
  )

model_files_registry <- model_files

model_files <- model_files_registry %>%
  filter(include_model) %>%
  select(model, pairwise_file)

missing_model_files <- model_files_registry %>%
  filter(!include_model)

if (nrow(model_files) == 0) {
  stop("No pairwise model inputs found.")
}

# ── 8. Dataset and species mappings ───────────────────────────────────────────
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

species_documentation <- tribble(
  ~study_species,                ~timetree_note,                                                ~elton_note,
  "Bison bonasus",               "TimeTree exact name expected.",                                "Elton exact name expected.",
  "Bos taurus",                  "TimeTree exact name expected.",                                "Elton exact name expected.",
  "Canis lupus familiaris",      "Mapped to Canis lupus for TimeTree.",                          "Fallback to Canis lupus if domestic dog is absent.",
  "Equus caballus",              "TimeTree exact name expected.",                                "Elton exact name expected.",
  "Equus ferus przewalskii",     "Mapped to Equus ferus for TimeTree.",                          "Fallback to Equus ferus, then Equus caballus if needed.",
  "Erinaceus europaeus",         "TimeTree exact name expected.",                                "Elton exact name expected.",
  "Giraffa camelopardalis",      "Final Newick matched Giraffa camelopardalis directly; Giraffa reticulata is only a fallback.", "Elton exact name first; Giraffa reticulata as fallback.",
  "Procyon lotor",               "TimeTree exact name expected.",                                "Elton exact name expected.",
  "Vulpes vulpes",               "TimeTree exact name expected.",                                "Elton exact name expected."
)

# ── 9. Load pairwise transfer metrics ─────────────────────────────────────────
pairwise <- pmap_dfr(model_files[, c("model", "pairwise_file")], read_pairwise_model) %>%
  mutate(
    train_species = recode(train_dataset, !!!dataset_to_species, .default = NA_character_),
    test_species = recode(test_dataset, !!!dataset_to_species, .default = NA_character_),
    same_species = train_species == test_species,
    model = factor(model, levels = model_order)
  )

unmapped <- bind_rows(
  pairwise %>%
    filter(is.na(train_species)) %>%
    distinct(dataset = train_dataset) %>%
    mutate(role = "train"),
  pairwise %>%
    filter(is.na(test_species)) %>%
    distinct(dataset = test_dataset) %>%
    mutate(role = "test")
)

if (nrow(unmapped) > 0) {
  write_csv(unmapped, file.path(csv_dir, "error_unmapped_datasets.csv"))
  stop("Some dataset names are not mapped to species. See error_unmapped_datasets.csv.")
}

study_species <- sort(unique(c(pairwise$train_species, pairwise$test_species)))

# ── 10. EltonTraits matching and distances ────────────────────────────────────
read_elton_traits <- function(elton_file) {
  elton_raw <- read_delim(
    elton_file,
    delim = "\t",
    na = c("", "NA", "NaN"),
    trim_ws = TRUE,
    show_col_types = FALSE
  )
  
  id_col <- pick_col(elton_raw, c("MSW3_ID", "SW3_ID", "MSW3.ID", "SW3.ID"))
  species_col <- pick_col(elton_raw, c("Scientific", "scientificNameStd", "scientificName", "Species"))
  
  required_elton <- c(id_col, species_col, elton_minimal_cols, elton_diet_cols)
  missing_elton <- setdiff(required_elton, names(elton_raw))
  if (length(missing_elton) > 0) {
    stop("MamFuncDat.txt is missing columns: ", paste(missing_elton, collapse = ", "))
  }
  
  elton_raw %>%
    transmute(
      elton_id = .data[[id_col]],
      elton_scientific = clean_species(.data[[species_col]]),
      bodymass_g = parse_number(as.character(.data[["BodyMass-Value"]])),
      log_bodymass_g = log(bodymass_g),
      forstrat_raw = as.character(.data[["ForStrat-Value"]]),
      forstrat_num = parse_number(as.character(.data[["ForStrat-Value"]])),
      activity_nocturnal = parse_number(as.character(.data[["Activity-Nocturnal"]])),
      activity_crepuscular = parse_number(as.character(.data[["Activity-Crepuscular"]])),
      activity_diurnal = parse_number(as.character(.data[["Activity-Diurnal"]])),
      diet_inv = parse_number(as.character(.data[["Diet-Inv"]])),
      diet_vend = parse_number(as.character(.data[["Diet-Vend"]])),
      diet_vect = parse_number(as.character(.data[["Diet-Vect"]])),
      diet_vfish = parse_number(as.character(.data[["Diet-Vfish"]])),
      diet_vunk = parse_number(as.character(.data[["Diet-Vunk"]])),
      diet_scav = parse_number(as.character(.data[["Diet-Scav"]])),
      diet_fruit = parse_number(as.character(.data[["Diet-Fruit"]])),
      diet_nect = parse_number(as.character(.data[["Diet-Nect"]])),
      diet_seed = parse_number(as.character(.data[["Diet-Seed"]])),
      diet_planto = parse_number(as.character(.data[["Diet-PlantO"]]))
    ) %>%
    filter(!is.na(elton_scientific))
}

build_elton_candidates <- function(study_species) {
  exact <- tibble(
    study_species = study_species,
    candidate = study_species,
    priority = 1L,
    match_source = "exact"
  )
  
  fallbacks <- tribble(
    ~study_species,                ~candidate,              ~priority, ~match_source,
    "Canis lupus familiaris",      "Canis lupus",           2L,        "manual_fallback_parent_species",
    "Equus ferus przewalskii",     "Equus ferus",           2L,        "manual_fallback_parent_species",
    "Equus ferus przewalskii",     "Equus caballus",        3L,        "manual_fallback_close_species",
    "Giraffa camelopardalis",      "Giraffa reticulata",    2L,        "manual_fallback_timetree_replacement"
  ) %>%
    filter(study_species %in% study_species)
  
  bind_rows(exact, fallbacks) %>%
    mutate(candidate_clean = clean_species(candidate))
}

match_elton_traits <- function(elton_file, study_species) {
  elton_traits <- read_elton_traits(elton_file)
  
  matched <- build_elton_candidates(study_species) %>%
    left_join(elton_traits, by = c("candidate_clean" = "elton_scientific")) %>%
    mutate(
      has_minimal = is.finite(log_bodymass_g) &
        !is.na(forstrat_raw) &
        is.finite(activity_nocturnal) &
        is.finite(activity_crepuscular) &
        is.finite(activity_diurnal),
      has_extended = has_minimal &
        is.finite(diet_inv) &
        is.finite(diet_vend) &
        is.finite(diet_vect) &
        is.finite(diet_vfish) &
        is.finite(diet_vunk) &
        is.finite(diet_scav) &
        is.finite(diet_fruit) &
        is.finite(diet_nect) &
        is.finite(diet_seed) &
        is.finite(diet_planto)
    ) %>%
    arrange(study_species, desc(has_minimal), priority) %>%
    group_by(study_species) %>%
    slice(1) %>%
    ungroup() %>%
    mutate(
      match_status = case_when(
        has_minimal & match_source == "exact" ~ "exact",
        has_minimal & match_source != "exact" ~ match_source,
        TRUE ~ "not_found_or_incomplete"
      ),
      matched_name = candidate
    )
  
  unmatched <- matched %>% filter(!has_minimal)
  if (nrow(unmatched) > 0) {
    write_csv(matched, file.path(csv_dir, "error_elton_matching_attempts.csv"))
    stop("Some study species have no complete minimal EltonTraits match. See error_elton_matching_attempts.csv.")
  }
  
  matched
}

elton_matches <- match_elton_traits(elton_file, study_species)

# Keep ForStrat numeric if it is numeric for all matched species. Otherwise use factor.
if (all(is.finite(elton_matches$forstrat_num))) {
  elton_matches <- elton_matches %>% mutate(forstrat_value = as.numeric(forstrat_num))
} else {
  elton_matches <- elton_matches %>% mutate(forstrat_value = as.factor(forstrat_raw))
}

elton_minimal_traits <- elton_matches %>%
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

if (any(!complete.cases(elton_minimal_traits))) {
  stop("Elton minimal trait table contains missing values.")
}
if (any(!complete.cases(elton_extended_traits))) {
  message("Warning: Elton extended trait table has missing values. Extended distance will be computed only if complete.")
}

elton_minimal_gower <- daisy(elton_minimal_traits %>% select(-species), metric = "gower")
elton_minimal_mat <- as.matrix(elton_minimal_gower)
rownames(elton_minimal_mat) <- elton_minimal_traits$species
colnames(elton_minimal_mat) <- elton_minimal_traits$species

elton_extended_gower <- daisy(elton_extended_traits %>% select(-species), metric = "gower")
elton_extended_mat <- as.matrix(elton_extended_gower)
rownames(elton_extended_mat) <- elton_extended_traits$species
colnames(elton_extended_mat) <- elton_extended_traits$species

elton_minimal_long <- make_gower_long(elton_minimal_mat, "elton_minimal_gower")
elton_extended_long <- make_gower_long(elton_extended_mat, "elton_extended_gower")

# ── 11. TimeTree matching and distances ───────────────────────────────────────
build_timetree_candidates <- function(study_species) {
  exact <- tibble(
    study_species = study_species,
    timetree_candidate = study_species,
    priority = 1L,
    match_source = "exact"
  )
  
  replacements <- tribble(
    ~study_species,                ~timetree_candidate,      ~priority, ~match_source,
    "Canis lupus familiaris",      "Canis lupus",            1L,        "manual_replacement_parent_species",
    "Equus ferus przewalskii",     "Equus ferus",            1L,        "manual_replacement_parent_species",
    "Giraffa camelopardalis",      "Giraffa reticulata",     2L,        "manual_fallback_timetree_replacement"
  ) %>%
    filter(study_species %in% study_species)
  
  bind_rows(exact, replacements) %>%
    mutate(candidate_clean = clean_species(timetree_candidate)) %>%
    arrange(study_species, priority)
}

read_timetree_distances <- function(timetree_file, study_species) {
  tree <- read.tree(timetree_file)
  tree$tip.label <- clean_species(tree$tip.label)
  
  if (anyDuplicated(tree$tip.label) > 0) {
    stop("TimeTree has duplicate tip labels after cleaning.")
  }
  
  tip_tbl <- tibble(timetree_tip = tree$tip.label)
  
  resolved <- build_timetree_candidates(study_species) %>%
    left_join(tip_tbl, by = c("candidate_clean" = "timetree_tip")) %>%
    mutate(found = !is.na(candidate_clean) & candidate_clean %in% tree$tip.label) %>%
    arrange(study_species, desc(found), priority) %>%
    group_by(study_species) %>%
    slice(1) %>%
    ungroup() %>%
    mutate(
      match_status = ifelse(found, match_source, "not_found"),
      timetree_label = ifelse(found, candidate_clean, NA_character_)
    )
  
  missing <- resolved %>% filter(!found)
  if (nrow(missing) > 0) {
    write_csv(resolved, file.path(csv_dir, "error_timetree_matching_attempts.csv"))
    stop("Some study species have no TimeTree tip match. See error_timetree_matching_attempts.csv.")
  }
  
  duplicated_tips <- resolved %>%
    count(timetree_label) %>%
    filter(n > 1)
  if (nrow(duplicated_tips) > 0) {
    write_csv(resolved, file.path(csv_dir, "error_timetree_duplicate_tip_mapping.csv"))
    stop("Two or more study species map to the same TimeTree tip. See error_timetree_duplicate_tip_mapping.csv.")
  }
  
  tree_use <- keep.tip(tree, resolved$timetree_label)
  
  rename_map <- setNames(resolved$study_species, resolved$timetree_label)
  tree_use$tip.label <- unname(rename_map[tree_use$tip.label])
  
  node_depth <- node.depth.edgelength(tree_use)
  tip_depth <- node_depth[seq_along(tree_use$tip.label)]
  tree_height <- max(tip_depth, na.rm = TRUE)
  
  species_grid <- expand_grid(
    train_species = tree_use$tip.label,
    test_species = tree_use$tip.label
  )
  
  get_divergence_age <- function(sp1, sp2) {
    if (sp1 == sp2) return(0)
    mrca <- getMRCA(tree_use, c(sp1, sp2))
    tree_height - node_depth[mrca]
  }
  
  divergence_long <- species_grid %>%
    mutate(
      timetree_divergence_mya = map2_dbl(train_species, test_species, get_divergence_age)
    )
  
  patristic_mat <- cophenetic.phylo(tree_use)
  patristic_long <- make_gower_long(patristic_mat, "timetree_patristic_mya")
  
  out <- divergence_long %>%
    left_join(patristic_long, by = c("train_species", "test_species")) %>%
    mutate(
      timetree_scaled_0_1 = scale_01(timetree_divergence_mya)
    )
  
  list(
    tree = tree_use,
    resolved = resolved,
    distances = out,
    tree_height_mya = tree_height
  )
}

timetree_obj <- read_timetree_distances(timetree_file, study_species)
timetree_long <- timetree_obj$distances
timetree_matches <- timetree_obj$resolved

# ── 12. Merge distances with transfer metrics ─────────────────────────────────
analysis_dt <- pairwise %>%
  left_join(elton_minimal_long, by = c("train_species", "test_species")) %>%
  left_join(elton_extended_long, by = c("train_species", "test_species")) %>%
  left_join(timetree_long, by = c("train_species", "test_species")) %>%
  mutate(model = factor(model, levels = model_order))

missing_distances <- analysis_dt %>%
  filter(
    is.na(elton_minimal_gower) |
      is.na(elton_extended_gower) |
      is.na(timetree_divergence_mya)
  ) %>%
  distinct(train_species, test_species, elton_minimal_gower, elton_extended_gower, timetree_divergence_mya)

if (nrow(missing_distances) > 0) {
  write_csv(missing_distances, file.path(csv_dir, "error_missing_distances_after_merge.csv"))
  stop("Some pairwise rows have missing distances. See error_missing_distances_after_merge.csv.")
}

species_level_dt <- analysis_dt %>%
  group_by(
    model,
    train_species,
    test_species,
    same_species,
    elton_minimal_gower,
    elton_extended_gower,
    timetree_divergence_mya,
    timetree_patristic_mya,
    timetree_scaled_0_1
  ) %>%
  summarise(
    mean_macro_f1 = mean(macro_f1, na.rm = TRUE),
    sd_macro_f1 = sd(macro_f1, na.rm = TRUE),
    n_rows = n(),
    .groups = "drop"
  )

# ── 13. Correlation analyses ─────────────────────────────────────────────────
distance_specs <- tribble(
  ~distance_version,          ~distance_col,              ~x_label,
  "elton_minimal_gower",      "elton_minimal_gower",      "EltonTraits minimal distance (Gower)",
  "elton_extended_gower",     "elton_extended_gower",     "EltonTraits extended distance (Gower)",
  "timetree_divergence_mya",  "timetree_divergence_mya",  "TimeTree divergence time (MYA)"
)

calc_correlations <- function(df, x_col, y_col, distance_version, level, scope) {
  bind_rows(
    safe_spearman(df[[x_col]], df[[y_col]]) %>%
      mutate(model = "overall", .before = 1),
    df %>%
      group_by(model) %>%
      group_modify(~ safe_spearman(.x[[x_col]], .x[[y_col]])) %>%
      ungroup() %>%
      mutate(model = as.character(model))
  ) %>%
    mutate(
      distance_version = distance_version,
      level = level,
      scope = scope,
      .before = 1
    )
}

correlation_summary <- pmap_dfr(distance_specs, function(distance_version, distance_col, x_label) {
  bind_rows(
    calc_correlations(
      analysis_dt,
      distance_col,
      "macro_f1",
      distance_version,
      "row",
      "all_pairs"
    ),
    calc_correlations(
      analysis_dt %>% filter(!same_species),
      distance_col,
      "macro_f1",
      distance_version,
      "row",
      "different_species_only"
    ),
    calc_correlations(
      species_level_dt,
      distance_col,
      "mean_macro_f1",
      distance_version,
      "species_pair",
      "all_pairs"
    ),
    calc_correlations(
      species_level_dt %>% filter(!same_species),
      distance_col,
      "mean_macro_f1",
      distance_version,
      "species_pair",
      "different_species_only"
    )
  )
}) %>%
  select(distance_version, level, scope, model, n, rho, p_value, ci_low, ci_high)

primary_summary <- correlation_summary %>%
  filter(level == "species_pair", scope == "different_species_only", model == "overall") %>%
  arrange(factor(distance_version, levels = distance_specs$distance_version))

distance_compare <- species_level_dt %>%
  distinct(train_species, test_species, same_species, elton_minimal_gower, elton_extended_gower, timetree_divergence_mya) %>%
  filter(!same_species) %>%
  summarise(
    rho_elton_minimal_vs_extended = cor(elton_minimal_gower, elton_extended_gower, method = "spearman"),
    rho_elton_minimal_vs_timetree = cor(elton_minimal_gower, timetree_divergence_mya, method = "spearman"),
    rho_elton_extended_vs_timetree = cor(elton_extended_gower, timetree_divergence_mya, method = "spearman")
  )

# ── 14. Plots ─────────────────────────────────────────────────────────────────
get_plot_style <- function(plot_style = c("bw", "color")) {
  plot_style <- match.arg(plot_style)
  
  if (plot_style == "color") {
    return(list(
      point_fill = "#4A4A4A",
      smooth_colour = "grey30",
      smooth_fill = "grey80",
      heatmap_low = "#FFF4C2",
      heatmap_high = "#B33000",
      label_fill = "white"
    ))
  }
  
  list(
    point_fill = "grey70",
    smooth_colour = "grey30",
    smooth_fill = "grey80",
    heatmap_low = "white",
    heatmap_high = "grey25",
    label_fill = "white"
  )
}

theme_clean <- function(plot_style = c("bw", "color")) {
  plot_style <- match.arg(plot_style)
  style <- get_plot_style(plot_style)
  
  theme_bw(base_size = 10, base_family = "sans") +
    theme(
      plot.title = element_blank(),
      plot.subtitle = element_blank(),
      plot.caption = element_blank(),
      axis.title.x = element_text(size = 11, margin = margin(t = 8)),
      axis.title.y = element_text(size = 11, margin = margin(r = 8)),
      axis.text.x = element_text(size = 10, colour = "black"),
      axis.text.y = element_text(size = 10, colour = "black"),
      strip.background = element_rect(fill = "grey95", colour = "black", linewidth = 0.6),
      strip.text = element_text(face = "bold", size = 11),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(colour = "grey85", linewidth = 0.3, linetype = "dashed"),
      panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.8),
      legend.position = "none",
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA)
    )
}

theme_heatmap <- function(plot_style = c("bw", "color")) {
  theme_clean(plot_style) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
      legend.position = "right",
      panel.grid.major.y = element_blank()
    )
}

save_plot <- function(plot_obj, file, width_mm = 180, height_mm = 130) {
  ggsave(
    filename = file,
    plot = plot_obj,
    width = width_mm,
    height = height_mm,
    units = "mm",
    dpi = 600,
    bg = "white"
  )
}

build_scatter <- function(fig_dt, distance_col, x_label, plot_style = c("bw", "color")) {
  plot_style <- match.arg(plot_style)
  style <- get_plot_style(plot_style)
  
  stats <- fig_dt %>%
    group_by(model) %>%
    group_modify(~ safe_spearman(.x[[distance_col]], .x$mean_macro_f1)) %>%
    ungroup() %>%
    mutate(
      stat_label = paste0(
        "rho = ", sprintf("%.3f", round(rho, 3)), "\n",
        "p = ", fmt_p(p_value)
      )
    )
  
  ggplot(fig_dt, aes(x = .data[[distance_col]], y = mean_macro_f1)) +
    geom_smooth(
      method = "lm",
      formula = y ~ x,
      se = TRUE,
      level = 0.95,
      linewidth = 0.7,
      colour = style$smooth_colour,
      fill = style$smooth_fill,
      alpha = 0.40
    ) +
    geom_point(
      shape = 21,
      size = 2.2,
      fill = style$point_fill,
      colour = "black",
      stroke = 0.35,
      alpha = 0.90
    ) +
    {
      if (isTRUE(show_p_values)) {
        geom_label(
          data = stats,
          aes(x = Inf, y = Inf, label = stat_label),
          inherit.aes = FALSE,
          hjust = 1.05,
          vjust = 1.15,
          size = 2.8,
          fontface = "bold",
          label.size = 0.35,
          fill = style$label_fill,
          colour = "black",
          lineheight = 0.95
        )
      }
    } +
    facet_wrap(~ model, ncol = 3, drop = TRUE) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25), labels = number_format(accuracy = 0.01)) +
    labs(x = x_label, y = "Mean macro-F1") +
    theme_clean(plot_style)
}

build_heatmap <- function(distance_long, distance_col, legend_title, plot_style = c("bw", "color")) {
  plot_style <- match.arg(plot_style)
  style <- get_plot_style(plot_style)
  
  sp_levels <- sort(unique(distance_long$train_species))
  
  distance_long %>%
    mutate(
      train_species = factor(train_species, levels = sp_levels),
      test_species = factor(test_species, levels = sp_levels)
    ) %>%
    ggplot(aes(x = test_species, y = train_species, fill = .data[[distance_col]])) +
    geom_tile(colour = "white", linewidth = 0.6) +
    geom_text(aes(label = sprintf("%.2f", .data[[distance_col]])), size = 2.6, colour = "black") +
    scale_fill_gradient(low = style$heatmap_low, high = style$heatmap_high) +
    scale_x_discrete(labels = short_species) +
    scale_y_discrete(labels = short_species) +
    labs(x = "Target species", y = "Source species", fill = legend_title) +
    theme_heatmap(plot_style)
}

plot_index <- tibble(file = character(), description = character())

fig_dt <- species_level_dt %>%
  filter(!same_species) %>%
  mutate(model = factor(model, levels = model_order))

for (plot_style in plot_styles) {
  suffix <- ifelse(plot_style == "bw", "bw", "color")
  
  p_heat_elton_min <- build_heatmap(
    elton_minimal_long,
    "elton_minimal_gower",
    "Gower",
    plot_style
  )
  heat_elton_min_file <- paste0("01_heatmap_elton_minimal_gower_", suffix, ".png")
  save_plot(p_heat_elton_min, file.path(plots_dir, heat_elton_min_file), width_mm = 190, height_mm = 150)
  
  p_heat_elton_ext <- build_heatmap(
    elton_extended_long,
    "elton_extended_gower",
    "Gower",
    plot_style
  )
  heat_elton_ext_file <- paste0("02_heatmap_elton_extended_gower_", suffix, ".png")
  save_plot(p_heat_elton_ext, file.path(plots_dir, heat_elton_ext_file), width_mm = 190, height_mm = 150)
  
  p_heat_timetree <- build_heatmap(
    timetree_long,
    "timetree_divergence_mya",
    "MYA",
    plot_style
  )
  heat_timetree_file <- paste0("03_heatmap_timetree_divergence_mya_", suffix, ".png")
  save_plot(p_heat_timetree, file.path(plots_dir, heat_timetree_file), width_mm = 190, height_mm = 150)
  
  plot_index <- bind_rows(
    plot_index,
    tibble(
      file = c(heat_elton_min_file, heat_elton_ext_file, heat_timetree_file),
      description = c(
        paste0("Elton minimal Gower heatmap (", suffix, ")"),
        paste0("Elton extended Gower heatmap (", suffix, ")"),
        paste0("TimeTree divergence-time heatmap (", suffix, ")")
      )
    )
  )
  
  for (i in seq_len(nrow(distance_specs))) {
    dist_name <- distance_specs$distance_version[[i]]
    dist_col <- distance_specs$distance_col[[i]]
    x_label <- distance_specs$x_label[[i]]
    
    p_scatter <- build_scatter(fig_dt, dist_col, x_label, plot_style)
    scatter_file <- paste0(
      sprintf("%02d", 3 + i),
      "_scatter_transfer_vs_",
      dist_name,
      "_",
      suffix,
      ".png"
    )
    save_plot(p_scatter, file.path(plots_dir, scatter_file), width_mm = 270, height_mm = 180)
    
    plot_index <- bind_rows(
      plot_index,
      tibble(
        file = scatter_file,
        description = paste0("Species-pair transfer vs ", dist_name, " (", suffix, ")")
      )
    )
  }
}

# ── 15. Export CSV ────────────────────────────────────────────────────────────
write_csv(model_files_registry, file.path(csv_dir, "00_model_file_registry.csv"))
write_csv(tibble(dataset = names(dataset_to_species), species = unname(dataset_to_species)),
          file.path(csv_dir, "01_dataset_to_species_mapping.csv"))
write_csv(species_documentation, file.path(csv_dir, "02_species_name_documentation.csv"))
write_csv(elton_matches, file.path(csv_dir, "03_elton_trait_matches.csv"))
write_csv(timetree_matches, file.path(csv_dir, "04_timetree_tip_matches.csv"))
write_csv(elton_minimal_traits, file.path(csv_dir, "05_elton_minimal_traits_used.csv"))
write_csv(elton_extended_traits, file.path(csv_dir, "06_elton_extended_traits_used.csv"))
write_csv(elton_minimal_long, file.path(csv_dir, "07_elton_minimal_distance_long.csv"))
write_csv(elton_extended_long, file.path(csv_dir, "08_elton_extended_distance_long.csv"))
write_csv(timetree_long, file.path(csv_dir, "09_timetree_distance_long.csv"))
write_csv(analysis_dt, file.path(csv_dir, "10_h3_pairwise_with_timetree_elton_distances.csv"))
write_csv(species_level_dt, file.path(csv_dir, "11_h3_species_pair_with_timetree_elton_distances.csv"))
write_csv(correlation_summary, file.path(csv_dir, "12_h3_timetree_elton_correlation_summary.csv"))
write_csv(primary_summary, file.path(csv_dir, "13_h3_primary_sensitivity_summary.csv"))
write_csv(distance_compare, file.path(csv_dir, "14_distance_matrix_rank_correlations.csv"))
write_csv(plot_index, file.path(csv_dir, "15_plot_index.csv"))

# ── 16. Export TXT reports ────────────────────────────────────────────────────
summary_lines <- c(
  "# ===================================================",
  "# H3 sensitivity summary: TimeTree and EltonTraits",
  "# ===================================================",
  "",
  paste0("Date: ", Sys.time()),
  paste0("Script: ", script_name),
  paste0("Base folder: ", base_dir),
  paste0("Model folder: ", models_dir),
  paste0("Input folder: ", input_dir),
  paste0("TimeTree file: ", timetree_file),
  paste0("EltonTraits file: ", elton_file),
  paste0("Output folder: ", out_dir),
  "",
  "# ----- Scope -----",
  "This script is a sensitivity analysis for H3.",
  "The main H3 analysis should remain the expert-coded functional-biomechanical trait distance.",
  "Distances tested here:",
  "- EltonTraits minimal Gower: log body mass, foraging stratum, activity pattern.",
  "- EltonTraits extended Gower: minimal Elton traits plus diet variables.",
  "- TimeTree divergence time in MYA from the Newick tree.",
  "",
  "# ----- Name replacements and matching decisions -----",
  "- Canis lupus familiaris is mapped to Canis lupus where needed.",
  "- Equus ferus przewalskii is mapped to Equus ferus for TimeTree.",
  "- Initial TimeTree matching can use Giraffa reticulata as a fallback, but the final Newick is matched exactly to Giraffa camelopardalis if that tip is present.",
  "- Elton matching tries exact names first and then documented fallbacks.",
  "",
  "# ----- Primary sensitivity result level -----",
  "Use species-pair level and different-species pairs as the main sensitivity interpretation.",
  ""
)

for (i in seq_len(nrow(primary_summary))) {
  row <- primary_summary[i, ]
  summary_lines <- c(
    summary_lines,
    paste0(
      row$distance_version,
      ": rho = ", fmt_num(row$rho, 3),
      ", 95% CI = ", fmt_ci(row$ci_low, row$ci_high, 3),
      ", p = ", fmt_p(row$p_value),
      ", n = ", row$n
    )
  )
}

summary_lines <- c(
  summary_lines,
  "",
  "# ----- Interpretation guide -----",
  "If the expert-coded H3 distance remains strongest, this supports the argument that functional-biomechanical traits are more relevant than broad ecological or phylogenetic distance.",
  "If TimeTree is also negative, it can be reported as phylogenetic support, but not as a replacement for the expert-coded biomechanical framework.",
  "If EltonTraits is weak, that is not a problem: EltonTraits are broader ecological traits and less targeted to accelerometer signal mechanics."
)

write_txt_report(file.path(txt_dir, "01_h3_timetree_elton_summary.txt"), summary_lines)

report_lines <- c(
  "# ===================================================",
  "# H3 TimeTree and EltonTraits sensitivity report",
  "# ===================================================",
  "",
  paste0("Script: ", script_name),
  paste0("Created on: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  paste0("Output folder: ", out_dir),
  paste0("Tree height from TimeTree Newick: ", fmt_num(timetree_obj$tree_height_mya, 3), " MYA"),
  ""
)

report_lines <- append_section(report_lines, "MODEL FILE REGISTRY", model_files_registry)
report_lines <- append_section(report_lines, "SPECIES NAME DOCUMENTATION", species_documentation)
report_lines <- append_section(report_lines, "ELTON MATCHES", elton_matches)
report_lines <- append_section(report_lines, "TIMETREE MATCHES", timetree_matches)
report_lines <- append_section(report_lines, "PRIMARY SENSITIVITY SUMMARY", primary_summary)
report_lines <- append_section(report_lines, "FULL CORRELATION SUMMARY", correlation_summary)
report_lines <- append_section(report_lines, "DISTANCE MATRIX RANK CORRELATIONS", distance_compare)
report_lines <- append_section(report_lines, "PLOT INDEX", plot_index)

write_txt_report(file.path(txt_dir, "02_h3_timetree_elton_full_report.txt"), report_lines)

files_lines <- c(
  paste0("Script: ", script_name),
  paste0("Output folder: ", out_dir),
  paste0("CSV folder: ", csv_dir),
  paste0("Plots folder: ", plots_dir),
  paste0("TXT folder: ", txt_dir),
  "",
  "Written CSV files:",
  "- 00_model_file_registry.csv",
  "- 01_dataset_to_species_mapping.csv",
  "- 02_species_name_documentation.csv",
  "- 03_elton_trait_matches.csv",
  "- 04_timetree_tip_matches.csv",
  "- 05_elton_minimal_traits_used.csv",
  "- 06_elton_extended_traits_used.csv",
  "- 07_elton_minimal_distance_long.csv",
  "- 08_elton_extended_distance_long.csv",
  "- 09_timetree_distance_long.csv",
  "- 10_h3_pairwise_with_timetree_elton_distances.csv",
  "- 11_h3_species_pair_with_timetree_elton_distances.csv",
  "- 12_h3_timetree_elton_correlation_summary.csv",
  "- 13_h3_primary_sensitivity_summary.csv",
  "- 14_distance_matrix_rank_correlations.csv",
  "- 15_plot_index.csv",
  "",
  "Written TXT files:",
  "- 01_h3_timetree_elton_summary.txt",
  "- 02_h3_timetree_elton_full_report.txt",
  "- 03_h3_timetree_elton_files_written.txt",
  "",
  "Written plot files:",
  paste0("- ", plot_index$file)
)

write_txt_report(file.path(txt_dir, "03_h3_timetree_elton_files_written.txt"), files_lines)

# ── 17. Console summary ───────────────────────────────────────────────────────
cat("\nH3 TimeTree + EltonTraits sensitivity complete.\n")
cat("Output folder:\n", out_dir, "\n\n")
cat("Primary sensitivity results:\n")
print(primary_summary, n = Inf)
