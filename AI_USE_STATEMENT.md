# AI Use Statement and Prompt Documentation

## Declaration on the use of AI tools

The conceptual design of this thesis, including the research question, hypotheses, methodological framework, dataset selection, preprocessing decisions, model selection, statistical analyses, interpretation of results, and final scientific conclusions, was developed by the author based on independent reasoning and the scientific literature.

AI-based tools, including ChatGPT, Claude, and Gemini, were used as supportive technical and editorial assistants during the preparation and review of analysis code, repository documentation, and script-level comments. Their use was limited to tasks such as identifying possible coding errors, debugging R and Python scripts, checking code consistency, improving code readability, adding concise English comments, suggesting clearer script headers, and reviewing whether the implemented code structure matched the author-defined analytical workflow.

AI tools were not used to autonomously generate empirical results, fabricate data, fabricate references, make final methodological decisions, or replace the author’s interpretation of the findings. Any AI-assisted suggestions were critically reviewed, tested, and, where appropriate, modified by the author before inclusion. AI-assisted code review was conducted at the level of scripts, documentation, and workflow structure; non-public raw datasets were not provided to AI tools. The author takes full responsibility for the final code, analyses, results, interpretation, and written thesis content.

## Use of AI tools in code preparation

During the preparation of the analysis repository, AI tools were used as technical assistants to improve the clarity, reproducibility, and maintainability of the code.

AI-assisted support included:

- detecting and fixing syntax or runtime errors in R and Python scripts;
- checking whether file paths, input registries, and output folders were handled consistently;
- reviewing metric calculations and export routines for consistency;
- improving code readability without changing the analytical logic;
- adding concise English comments and publication-ready script headers;
- checking whether R analysis scripts were consistent with the author-defined hypotheses H1, H2, and H3;
- reviewing Python model scripts for reproducibility, random seeds, output structure, and possible edge cases;
- checking possible data leakage risks in within-dataset, cross-dataset, pairwise transfer, and pooled multi-dataset evaluation scripts.

The final implementation decisions, testing, interpretation, and inclusion of all code changes were performed by the author.

## Note on AI-assisted code editing

Some scripts in this repository were reviewed with the support of AI tools to improve readability, detect possible bugs, harmonize comments, and standardize script headers.

The analytical workflow, model choices, statistical tests, interpretation, and final code decisions were made by the author. AI-generated suggestions were not accepted automatically but were checked, tested, and adapted before inclusion.

## Representative prompts used for technical code assistance

The following prompts represent the type of AI-assisted support used during the preparation and review of the R and Python analysis scripts. They are included to document the role of AI tools transparently. The prompts were used for technical assistance and consistency checks, not to replace the author’s scientific reasoning, methodological decisions, or interpretation of results.

### Prompt 1: R script header and documentation

Please review the following R script and make it publication-ready without changing the analytical logic or executable code structure. Add a clear header with author, date, purpose, and hypothesis. Add only short English section comments where helpful. Do not change the statistical workflow, input files, output files, object names, or analysis logic.

### Prompt 2: R bug check

Please check this R script for possible bugs, missing packages, inconsistent column names, path problems, or object names that may cause runtime errors. Do not rewrite the whole script. Only point out likely issues and suggest minimal fixes while preserving the current structure and analysis logic.

### Prompt 3: R statistical consistency check for H1

Please check whether this R script matches the intended hypothesis test. For H1, the script should compare paired within-dataset and cross-dataset Macro-F1 scores using descriptive summaries and paired Wilcoxon tests. Please identify inconsistencies, but do not change the analysis design.

### Prompt 4: R statistical consistency check for H2

Please check whether this R script matches the intended hypothesis test. For H2, the script should compare class-specific F1 scores for foraging, locomotion, and resting across models using Friedman tests and paired post-hoc Wilcoxon comparisons where appropriate. Please identify possible inconsistencies, but do not change the analysis design.

### Prompt 5: R statistical consistency check for H3

Please check whether this R script matches the intended hypothesis test. For H3, the script should test whether cross-species transfer performance declines with increasing functional-biomechanical trait distance between source and target species. Please check whether Gower distances, species-pair identifiers, model-specific correlations, and exported summary tables are handled consistently.

### Prompt 6: Python model script review

Please review this Python model script for bugs, reproducibility issues, and possible edge cases. The script trains and evaluates behavior classification models using tri-axial accelerometer windows. Do not change the scientific workflow or model logic. Focus on syntax errors, missing imports, random seed handling, data leakage risks, file I/O consistency, and robust metric export.

### Prompt 7: Python preprocessing script review

Please check this Python preprocessing script for consistency and possible bugs. The script should harmonize accelerometer datasets into a comparable window-level format. Please focus on timestamp handling, axis naming, resampling, segmentation, behavior-label mapping, missing values, file I/O, and output consistency. Do not change the preprocessing logic unless a minimal bug fix is required.

### Prompt 8: Data leakage check

Please review this Python script for possible data leakage. Focus on whether training and test data are separated correctly, whether scaling or feature selection is fitted only on the training data where required, and whether dataset, individual, subject, or source-file identifiers are used correctly during splitting. Suggest only minimal fixes.

### Prompt 9: Metric calculation check

Please check whether this Python script calculates classification metrics correctly and consistently. The script should export accuracy, precision, recall, F1 scores, Macro-F1, and class-specific F1 scores where required. Please check label ordering, missing classes, zero-division handling, and consistency with downstream R scripts. Do not change the analysis logic.

### Prompt 10: Random seed and reproducibility check

Please review this Python script for reproducibility. Check whether random seeds are set consistently for Python, NumPy, machine-learning libraries, and deep-learning frameworks where applicable. Please also check whether parallel processing, train/test splits, and exported results are handled in a reproducible way. Do not change the scientific workflow.

## Summary of AI tool use

**Tools used:** ChatGPT, Claude, Gemini

**Purpose of use:**  
AI tools were used for technical support during code preparation, including debugging, code review, readability improvements, concise English comments, script headers, reproducibility checks, and consistency checks across R and Python scripts.

**Type of scripts reviewed:**  

- R scripts for H1, H2, H3, H3 sensitivity analyses, and H3 bias-control analyses.
- Python scripts for preprocessing, feature extraction, model training, within-dataset evaluation, cross-dataset transfer, pairwise transfer, pooled multi-dataset evaluation, and metric export.

**Nature of AI assistance:**  

- Syntax and runtime bug checks.
- Review of possible data leakage risks.
- Review of input/output consistency.
- Suggestions for clearer code comments and documentation.
- Suggestions for publication-ready script headers.
- Checks of whether code structure matched the intended analysis.
- Review of reproducibility settings such as random seeds and output folders.

**Author responsibility:**  
All AI-generated suggestions were reviewed, tested, and adapted by the author. The final code, results, analyses, interpretation, and thesis content remain the responsibility of the author.
