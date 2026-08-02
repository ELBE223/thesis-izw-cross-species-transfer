# AI Use Statement and Prompt Documentation

## Declaration on the use of AI tools

The conceptual design of this thesis, including the research question, hypotheses, methodological framework, dataset selection, preprocessing decisions, model selection, statistical analyses, interpretation of results, and final scientific conclusions, was developed by the author based on independent reasoning and the scientific literature.

AI-based tools, including ChatGPT, Claude, and Gemini, were used as supportive technical and editorial assistants during the preparation and review of analysis code, repository documentation, script-level comments, and selected passages of the written thesis. Their use was limited to tasks such as identifying possible coding errors, debugging R and Python scripts, checking code consistency, improving code readability, adding concise English comments, suggesting clearer script headers, reviewing whether the implemented code structure matched the author-defined analytical workflow, checking possible data leakage risks, and reviewing selected text passages for language, clarity, and academic style.

AI tools were not used to autonomously generate empirical results, fabricate data or references, make final methodological decisions, or replace the author’s scientific interpretation of the findings. Any AI-assisted suggestions were critically reviewed, tested, and, where appropriate, modified by the author before inclusion. AI-assisted code review was conducted at the level of scripts, documentation, and workflow structure; non-public raw datasets were not provided to AI tools. The author takes full responsibility for the final code, analyses, results, interpretation, and written thesis content.

## Use of AI tools in code preparation

During the preparation of the analysis repository, AI tools were used as technical assistants to improve the clarity, reproducibility, and maintainability of the code.

AI-assisted support included:

* detecting and fixing syntax or runtime errors in R and Python scripts;
* checking whether file paths, input registries, and output folders were handled consistently;
* reviewing metric calculations and export routines for consistency;
* improving code readability without changing the analytical logic;
* adding concise English comments and publication-ready script headers;
* checking whether R analysis scripts were consistent with the author-defined hypotheses H1, H2, and H3;
* reviewing Python model scripts for reproducibility, random-seed handling, output structure, and possible edge cases;
* checking possible data leakage risks in within-dataset, cross-dataset, pairwise-transfer, and pooled multi-dataset evaluation scripts.

The final implementation decisions, testing, interpretation, and inclusion of all code changes were performed by the author.

## Representative prompts used 

### Prompt 1: R script header and documentation

Please review the following R script and make it publication-ready without changing the analytical logic or executable code structure. Add a clear header with the author, date, purpose, and relevant hypothesis. Add only short English section comments where helpful. Do not change the statistical workflow, input files, output files, object names, or analysis logic.

### Prompt 2: R bug check

Please check this R script for possible bugs, missing packages, inconsistent column names, path problems, or object names that may cause runtime errors. Do not rewrite the whole script. Only point out likely issues and suggest minimal fixes while preserving the current structure and analysis logic.

### Prompt 3: R statistical consistency check for H1

Please check whether this R script matches the intended hypothesis test. For H1, the script should compare paired within-dataset and cross-dataset Macro-F1 scores using descriptive summaries and paired Wilcoxon signed-rank tests. Please identify inconsistencies, but do not change the analysis design.

### Prompt 4: R statistical consistency check for H2

Please check whether this R script matches the intended hypothesis test. For H2, the script should compare class-specific F1 scores for resting, locomotion, and foraging across models using Friedman tests and paired post-hoc Wilcoxon signed-rank comparisons where appropriate. Please identify possible inconsistencies, but do not change the analysis design.

### Prompt 5: R statistical consistency check for H3

Please check whether this R script matches the intended hypothesis test. For H3, the script should test whether cross-species transfer performance declines with increasing functional-biomechanical trait distance between source and target species. Please check whether Gower distances, species-pair identifiers, dyadic dependence structures, model-specific effects, sensitivity analyses, and exported summary tables are handled consistently.

### Prompt 6: Python model script review

Please review this Python model script for bugs, reproducibility issues, and possible edge cases. The script trains and evaluates behavior-classification models using tri-axial accelerometer windows. Do not change the scientific workflow or model logic. Focus on syntax errors, missing imports, random-seed handling, data leakage risks, file input and output consistency, and robust metric export.

### Prompt 7: Python preprocessing script review

Please check this Python preprocessing script for consistency and possible bugs. The script should harmonize accelerometer datasets into a comparable window-level format. Please focus on timestamp handling, axis naming, resampling, segmentation, behavior-label mapping, missing values, file input and output, and output consistency. Do not change the preprocessing logic unless a minimal bug fix is required.

### Prompt 8: Data leakage check

Please review this Python script for possible data leakage. Focus on whether training and test data are separated correctly, whether scaling or feature selection is fitted only on the training data where required, and whether dataset, individual, subject, or source-file identifiers are used correctly during splitting. Suggest only minimal fixes.

### Prompt 9: Metric calculation check

Please check whether this Python script calculates classification metrics correctly and consistently. The script should export accuracy, precision, recall, F1 scores, Macro-F1, and class-specific F1 scores where required. Please check label ordering, missing classes, zero-division handling, and consistency with downstream R scripts. Do not change the analysis logic.

### Prompt 10: Random-seed and reproducibility check

Please review this Python script for reproducibility. Check whether random seeds are set consistently for Python, NumPy, machine-learning libraries, and deep-learning frameworks where applicable. Please also check whether parallel processing, training and test splits, and exported results are handled reproducibly. Do not change the scientific workflow.

### Prompt 11: Language and style review_thesis

Please review the following thesis passage for grammar, clarity, conciseness, and academic American English. Preserve the scientific meaning, argumentation, citations, numerical results, and level of certainty. Do not introduce new claims, references, interpretations, or methodological changes. Identify substantial issues separately and otherwise suggest only minimal language edits.

## Summary of AI tool use

**Tools used:** ChatGPT, Claude, and Gemini

**Purpose of use:**
AI tools were used for technical support during code preparation, including debugging, code review, readability improvements, concise English comments, script headers, reproducibility checks, and consistency checks across R and Python scripts.

**Types of scripts reviewed:**

* R scripts for H1, H2, and H3, including the H3 sensitivity analyses;
* Python scripts for preprocessing, feature extraction, model training, within-dataset evaluation, cross-dataset transfer, pairwise transfer, pooled multi-dataset evaluation, and metric export.
