# Portfolio  
**Name:** Kornel Gorski  
**Student Number:** 4880765  

**Individual GitHub Repository:** https://github.com/xemau/ADAI-individual-repository-2025  
**Group GitHub Repository:** https://github.com/FontysVenlo/grouprepository-group-2  

---

# Learning Outcomes  

## Learning Outcome 1: Evaluate machine learning and neural network concepts  

**Entry Level (Self-Assessment)**  
Before starting the project, I had some theoretical understanding of machine learning and neural networks, including basic architectures like feedforward and convolutional networks, but limited practical experience applying these concepts to real datasets.  

**Performed Activities**

| Week | Activities | Evidence |
|------|------------|----------|
| 3    | Implemented and trained a CNN model for skin lesion classification. Experimented with different hyperparameters to understand their effect on model performance. | [`src/train.py`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/src/train.py) demonstrates the model training pipeline including architecture definition, loss calculation, and optimization steps. |
| 3    | Explored the BCN20000 dataset annotations, including class labels and metadata. Applied preprocessing steps to handle missing or incorrect annotations. | [`notebooks/01_data_exploration.ipynb`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/notebooks/01_data_exploration.ipynb) explores dataset and annotation distribution. [`notebooks/02_preprocessing.ipynb`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/notebooks/02_preprocessing.ipynb) implements preprocessing and cleaning of annotated data. |
| 3    | Trained a baseline SimpleCNN on the BCN20000 skin lesion dataset with an end-to-end training loop (loss, optimizer, evaluation). Investigated the impact of basic hyperparameters such as learning rate, batch size, and number of epochs on model performance. | [`notebooks/03_model_training.ipynb`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/notebooks/03_model_training.ipynb) shows the full SimpleCNN training pipeline, including metrics and learning curves used as a baseline for later experiments. |
| 3    | Implemented transfer learning by fine-tuning a pretrained CNN on the skin lesion dataset. Compared training from scratch versus transfer learning to evaluate benefits. | [`notebooks/04_convolution_experiment.ipynb`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/notebooks/04_convolution_experiment.ipynb) experiments with convolution filter sizes and analyzes training curves and performance metrics. |
| 4    | Extended experiments: binary classification with medical metrics (accuracy, recall, AUROC) and TTA; checkpoint-based results reproduction; multi-class ResNet18 training and validation; side-by-side comparison of binary vs. multi-class. | [`notebooks/05_binary_classification.ipynb`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/notebooks/05_binary_classification.ipynb), [`notebooks/05_result_binary_classification.ipynb`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/notebooks/05_result_binary_classification.ipynb), [`notebooks/06_multi_class_classification.ipynb`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/notebooks/06_multi_class_classification.ipynb), [`notebooks/07_comparison_binary_multiclass.ipynb`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/notebooks/07_comparison_binary_multiclass.ipynb), plots in [artifacts/plots/](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/artifacts), metrics in [artifacts/binary_metrics.json](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/artifacts/binary_metrics.json) and [artifacts/multiclass_metrics.json](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/artifacts/multiclass_metrics.json) |
| 4    | Constructed benign/malignant mapping from diagnosis labels and validated class balance; selected screening-appropriate metrics; analyzed calibration and threshold effects to relate predictions back to annotation quality. | Mapping and metrics in [`notebooks/05_binary_classification.ipynb`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/notebooks/05_binary_classification.ipynb); calibration and threshold plots from [`notebooks/05_result_binary_classification.ipynb`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/notebooks/05_result_binary_classification.ipynb) saved to [artifacts/plots/](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/artifacts) |
| 4    | Fine-tuned pretrained ResNet18 for binary and multi-class tasks; applied validation-time TTA; compared transfer-learned models against earlier baseline. | [`notebooks/05_binary_classification.ipynb`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/notebooks/05_binary_classification.ipynb), [`notebooks/06_multi_class_classification.ipynb`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/notebooks/06_multi_class_classification.ipynb), comparison in [`notebooks/07_comparison_binary_multiclass.ipynb`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/notebooks/07_comparison_binary_multiclass.ipynb); model builder in [`src/utils/models_utils.py`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/src/utils/models_utils.py) |
| 5    | Performed dataset evaluation including age, diagnosis, localization, malignancy, and sex distribution analyses; created plots and summary tables. | [`evaluation.md`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/evaluation.md) with plots in [artifacts/plots/](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/artifacts) |


**Reflection (Weekly)**  

| Week | What went well | What was difficult | How to improve |
|------|----------------|--------------------|----------------|
| 3    | Gained hands-on experience training CNNs on real medical image data and saw the full pipeline from data loading to evaluation working end-to-end. | Understanding the effect of different hyperparameters and architectures on convergence and generalization was challenging and often unintuitive. | Design smaller, controlled experiments and track results more systematically to build intuition about which changes matter most. |
| 4    | Extended the approach to binary and multi-class setups and successfully used transfer learning and TTA to improve validation performance. | Balancing complexity (pretrained models, TTA, more metrics) with training time and reproducibility required careful orchestration of code and experiments. | Automate more of the experiment configuration, logging, and comparison so that results across models and runs are easier to interpret and reproduce. |
| 5    | Consolidated understanding of the dataset by linking model performance back to class balance, demographics, and localization distributions. | Interpreting how dataset biases and imbalances might impact model behavior in edge cases was non-trivial. | Incorporate dataset diagnostics earlier in the workflow and plan additional targeted experiments for underrepresented groups or classes. |

**Grading Level (Self-Assessment)**  
Undefined – You have yet to start addressing this Learning Outcome (not passed, 4)  
Orienting – You are beginning to address this Learning Outcome (5)  
Beginning – You have made some progress towards this Learning Outcome (6)  
**Proficient – You have made substantial progress and are competent in this Learning Outcome (8)**  
Advanced – You have fully mastered this Learning Outcome (10)  

---

## Learning Outcome 2: Apply and evaluate annotation strategies  

**Entry Level (Self-Assessment)**  
I was aware of the importance of data annotation in learning but had limited experience with annotation methods or evaluating their quality and impact on model performance.  

**Performed Activities**

| Week | Activities | Evidence |
|------|------------|----------|

**Reflection (Weekly)**  

| Week | What went well | What was difficult | How to improve |
|------|----------------|--------------------|----------------|

**Grading Level (Self-Assessment)**  
**Undefined – You have yet to start addressing this Learning Outcome (not passed, 4)** 
Orienting – You are beginning to address this Learning Outcome (5)  
Beginning – You have made some progress towards this Learning Outcome (6)  
Proficient – You have made substantial progress and are competent in this Learning Outcome (8)
Advanced – You have fully mastered this Learning Outcome (10)  

---

## Learning Outcome 3: Evaluate Large Language Model concepts  

**Entry Level (Self-Assessment)**  
I had basic knowledge of Large Language Models (LLMs) and their architectures but limited understanding of their training mechanisms and applications.  

**Performed Activities**

| Week | Activities | Evidence |
|------|------------|----------|
| 10    | Analysed the AMI case study to understand when Large Language Models are inappropriate for numerical time-series forecasting and why classical models are preferred as the core forecasting engine. Identified the need for a hybrid architecture where LLMs sit on top of time-series models. | [`03_business_case/main/artifacts/forecasting_report.md`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/03_business_case/main/artifacts/forecasting_report.md) documents the role and limitations of LLMs in the demand forecasting context. |
| 11    | Implemented Prophet and SARIMAX time-series models as a forecasting backbone and used their outputs (forecasts, confidence intervals, backtest metrics) as inputs for an LLM insight layer concept. Evaluated how LLMs can complement, but not replace, the statistical models. | [`03_business_case/main/notebooks/02_prophet_forecasting.ipynb`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/03_business_case/main/notebooks/02_prophet_forecasting.ipynb), [`03_business_case/main/notebooks/03_sarimax_forecasting.ipynb`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/03_business_case/main/notebooks/03_sarimax_forecasting.ipynb), and metrics/forecasts in [`03_business_case/main/artifacts`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/03_business_case/main/artifacts) underpin the comparison. |
| 13   | Designed and documented a hybrid forecasting architecture where LLMs generate natural-language explanations and recommendations based on time-series forecasts and uncertainties. Summarised recommendations for the company, including when and how to deploy LLMs safely. | [`03_business_case/main/artifacts/forecasting_report.md`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/03_business_case/main/artifacts/forecasting_report.md) includes the final narrative and recommendations about LLM usage in the AMI forecasting solution. |

**Reflection (Weekly)**  

| Week | What went well | What was difficult | How to improve |
|------|----------------|--------------------|----------------|
| 10   | Our group aligned on how to interpret the AMI case and split the work so that each person could focus on specific aspects (models, report structure, LLM role). Individually, I gained a clear understanding of why LLMs are not suitable as primary time-series forecasters. | Working only from a written assignment without any direct stakeholder interaction made it harder to judge how much detail and technical depth was actually needed. Translating the case into concrete technical requirements required a lot of reading and thought. | In future, a short clarification session with stakeholders (if available) or, at minimum, agree as a group on a concise “problem statement” before diving into implementation and analysis. |
| 11   | I implemented Prophet and SARIMAX independently, but the group exchanged ideas and sanity-checked assumptions informally. This helped to keep the modelling direction roughly aligned even though we worked mostly individually. | Coordinating modelling decisions without a strict shared design was challenging. Mostly due to communication and holidays. | Introduce a minimal “modelling spec” at the start (what horizon, which metrics, how to split train/test) so that individually developed models remain comparable and easier to combine in a group report. |
| 13   | I contributed to a written proposal where we recommended a hybrid architecture (time-series models + LLM insight layer) instead of trying to build an app in three weeks. The focus on proposed solutions rather than incomplete implementation made the outcome more realistic. | Justifying recommendations without an actual running system or stakeholder feedback felt abstract at times. It was not always clear how our proposed solution would be received in a real organisational context. | In similar short assignments, I would still try to create at least a very small technical demo or experiment (even if not production-ready).. |


**Grading Level (Self-Assessment)**  
Undefined – You have yet to start addressing this Learning Outcome (not passed, 4)  
Orienting – You are beginning to address this Learning Outcome (5)  
Beginning – You have made some progress towards this Learning Outcome (6)
Proficient – You have made substantial progress and are competent in this Learning Outcome (8)
**Advanced – You have fully mastered this Learning Outcome (10)**

---

## Learning Outcome 4: Evaluate transfer learning principles  

**Entry Level (Self-Assessment)**  

**Performed Activities**

| Week | Activities | Evidence |
|------|------------|----------|

**Reflection**  
- What went well:
- What was difficult: 

**Grading Level (Self-Assessment)**  
**Undefined – You have yet to start addressing this Learning Outcome (not passed, 4)**  
Orienting – You are beginning to address this Learning Outcome (5)  
Beginning – You have made some progress towards this Learning Outcome (6)  
Proficient – You have made substantial progress and are competent in this Learning Outcome (8) 
Advanced – You have fully mastered this Learning Outcome (10)  

---

## Learning Outcome 5: Show professional skills  

**Entry Level (Self-Assessment)**  
I had strong prior experience with version control (Git/GitHub), but limited practice in professional reporting.  

**Performed Activities**

| Week | Activities | Evidence |
|------|------------|----------|
| 3    | Maintained a structured GitHub repository with clear commit messages and organized code. Produced comprehensive documentation and reports summarizing methodology, results, and reflections. | Project repository at https://github.com/xemau/ADAI-individual-repository-2025 demonstrates professional code management. This README and accompanying Jupyter notebooks provide clear communication of project outcomes. |
| 4    | Refactored shared code into reusable utilities; standardized artifact logging and plotting; produced reproducible evaluation notebooks for binary and multi-class; updated portfolio evidence. | Utilities in [`src/utils/`](src/utils/); notebooks [`05_binary_classification.ipynb`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/notebooks/05_binary_classification.ipynb), [`05_result_binary_classification.ipynb`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/notebooks/05_result_binary_classification.ipynb), [`06_multi_class_classification.ipynb`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/notebooks/06_multi_class_classification.ipynb), [`07_comparison_binary_multiclass.ipynb`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/notebooks/07_comparison_binary_multiclass.ipynb); logs in [artifacts/metrics_log.csv](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/artifacts/metrics_log.csv) and [artifacts/metrics_log_multiclass.csv](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/artifacts/metrics_log_multiclass.csv) |
| 5    | Documented dataset evaluation systematically in markdown with evidence links and summary table; maintained professional reporting standards. | [`evaluation.md`](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/evaluation.md), repository updates with plots in [artifacts/plots/](https://github.com/xemau/ADAI-individual-repository-2025/tree/main/01_cnn_assignment/artifacts) |

**Reflection (Weekly)**  

| Week | What went well | What was difficult | How to improve |
|------|----------------|--------------------|----------------|
| 3    | Maintained a clear repository structure with meaningful commits and kept notebooks and code in sync with the evolving experiments. | Ensuring that all important decisions and rationale were captured in the documentation while still moving the project forward was demanding. | Write shorter but more frequent notes during development and reserve time at the end of each week to clean up and consolidate documentation. |
| 4    | Refactored code into reusable utilities and standardized logging and plotting, which made later experiments easier to run and compare. | Deciding where to draw the line between “good enough” engineering and over-engineering in a time-boxed academic project was not always obvious. | Define simple internal standards for utilities, logging, and structure at the start of the project and stick to them unless there is a strong reason to deviate. |
| 5    | Produced structured evaluation notes and visual evidence (plots, tables) that clearly support the portfolio and presentation. | Summarizing many experiments and insights into concise, readable artifacts without losing important nuance took several iterations. | Start drafting evaluation notes in parallel with experiments, and incrementally refine them rather than writing everything at the end. |

**Grading Level (Self-Assessment)**  
Undefined – You have yet to start addressing this Learning Outcome (not passed, 4)  
Orienting – You are beginning to address this Learning Outcome (5)  
Beginning – You have made some progress towards this Learning Outcome (6)  
**Proficient – You have made substantial progress and are competent in this Learning Outcome (8)**  
Advanced – You have fully mastered this Learning Outcome (10)  