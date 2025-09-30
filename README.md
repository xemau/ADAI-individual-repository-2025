# Portfolio  
**Name:** Kornel Gorski  
**Student Number:** 4880765  

**Individual GitHub Repository:** https://github.com/xemau/ADAI-individual-repository-2025  
**Group GitHub Repository:** https://github.com/FontysVenlo/grouprepository-group-2  

---

## Case Study: Diagnosis of Skin Cancer  

**Situation**  
You work for an innovative medical technology company that focuses on improving the diagnosis of skin cancer. The company has established a project that utilises Convolutional Neural Networks (CNNs) to analyse skin images and classify suspicious lesions. The goal is to develop a model that supports doctors in the early detection of skin cancer, which is crucial for successful treatment outcomes.  

**Problem Statement**  
The team is facing several challenges in developing the CNN model. The accuracy of the classification is inconsistent, and there are concerns about the interpretation of the results. Additionally, there is a need for a deeper understanding of the limitations of the techniques used and the impact of the dataset on the model's performance.  

---

# Learning Outcomes  

## Learning Outcome 1: Evaluate machine learning and neural network concepts  

**Entry Level (Self-Assessment)**  
Before starting the project, I had some theoretical understanding of machine learning and neural networks, including basic architectures like feedforward and convolutional networks, but limited practical experience applying these concepts to real datasets.  

**Performed Activities**

| Week | Activities | Evidence |
|------|------------|----------|
| 3    | Implemented and trained a CNN model for skin lesion classification. Experimented with different hyperparameters to understand their effect on model performance. | [`src/train.py`](src/train.py) demonstrates the model training pipeline including architecture definition, loss calculation, and optimization steps. |
| 4    | Extended experiments: binary classification with medical metrics (accuracy, recall, AUROC) and TTA; checkpoint-based results reproduction; multi-class ResNet18 training and validation; side-by-side comparison of binary vs. multi-class. | [`notebooks/05_binary_classification.ipynb`](notebooks/05_binary_classification.ipynb), [`notebooks/05_result_binary_classification.ipynb`](notebooks/05_result_binary_classification.ipynb), [`notebooks/06_multi_class_classification.ipynb`](notebooks/06_multi_class_classification.ipynb), [`notebooks/07_comparison_binary_multiclass.ipynb`](notebooks/07_comparison_binary_multiclass.ipynb), plots in `artifacts/plots/`, metrics in `artifacts/binary_metrics.json` and `artifacts/multiclass_metrics.json` |
| 5    | Performed dataset evaluation including age, diagnosis, localization, malignancy, and sex distribution analyses; created plots and summary tables. | [`evaluation.md`](evaluation.md) with plots in `artifacts/plots/` |

**Reflection**  
- What went well: I developed a solid understanding of how neural network components interact and influence learning.  
- What was difficult: Grasping the impact of hyperparameter choices on convergence and generalization required iterative experimentation.  

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
| 3    | Explored the BCN20000 dataset annotations, including class labels and metadata. Applied preprocessing steps to handle missing or incorrect annotations. | [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) explores dataset and annotation distribution. [`notebooks/02_preprocessing.ipynb`](notebooks/02_preprocessing.ipynb) implements preprocessing and cleaning of annotated data. |
| 4    | Constructed benign/malignant mapping from diagnosis labels and validated class balance; selected screening-appropriate metrics; analyzed calibration and threshold effects to relate predictions back to annotation quality. | Mapping and metrics in [`notebooks/05_binary_classification.ipynb`](notebooks/05_binary_classification.ipynb); calibration and threshold plots from [`notebooks/05_result_binary_classification.ipynb`](notebooks/05_result_binary_classification.ipynb) saved to `artifacts/plots/` |
| 5    | Evaluated annotation balance and representativeness through dataset distribution analysis; assessed implications of class imbalance on annotation strategies. | [`evaluation.md`](evaluation.md), supporting plots in `artifacts/plots/` |

**Reflection**  
- What went well: I learned to critically assess annotation quality and its implications for model training.  
- What was difficult: Identifying and resolving annotation errors required careful data inspection and domain knowledge.  

**Grading Level (Self-Assessment)**  
Undefined – You have yet to start addressing this Learning Outcome (not passed, 4)  
Orienting – You are beginning to address this Learning Outcome (5)  
Beginning – You have made some progress towards this Learning Outcome (6)  
**Proficient – You have made substantial progress and are competent in this Learning Outcome (8)**
Advanced – You have fully mastered this Learning Outcome (10)  

---

## Learning Outcome 3: Evaluate Large Language Model concepts  

**Entry Level (Self-Assessment)**  
I had basic knowledge of Large Language Models (LLMs) and their architectures but limited understanding of their training mechanisms and applications.  

**Performed Activities**

| Week | Activities | Evidence |
|------|------------|----------|
| 3    | Studied foundational concepts and architectures of LLMs through literature review and tutorials. Compared LLM principles with CNNs used in image analysis to understand modality-specific model designs. | [`docs/LLM_concepts_summary.md`](docs/LLM_concepts_summary.md) summarizes key properties and challenges of LLMs. [`notebooks/03_model_training.ipynb`](notebooks/03_model_training.ipynb) demonstrates training a SimpleCNN and evaluating results, providing a comparison to LLM concepts. |

**Reflection**  
- What went well: I developed a conceptual framework for understanding LLM capabilities and limitations.  
- What was difficult: Relating LLM concepts to practical applications required synthesizing diverse sources of information.  

**Grading Level (Self-Assessment)**  
Undefined – You have yet to start addressing this Learning Outcome (not passed, 4)  
Orienting – You are beginning to address this Learning Outcome (5)  
**Beginning – You have made some progress towards this Learning Outcome (6)**
Proficient – You have made substantial progress and are competent in this Learning Outcome (8)  
Advanced – You have fully mastered this Learning Outcome (10)  

---

## Learning Outcome 4: Evaluate transfer learning principles  

**Entry Level (Self-Assessment)**  
I was aware that transfer learning can improve model performance but had limited hands-on experience applying pretrained models to new tasks.  

**Performed Activities**

| Week | Activities | Evidence |
|------|------------|----------|
| 3    | Implemented transfer learning by fine-tuning a pretrained CNN on the skin lesion dataset. Compared training from scratch versus transfer learning to evaluate benefits. | [`notebooks/04_convolution_experiment.ipynb`](notebooks/04_convolution_experiment.ipynb) experiments with convolution filter sizes and analyzes training curves and performance metrics. |
| 4    | Fine-tuned pretrained ResNet18 for binary and multi-class tasks; applied validation-time TTA; compared transfer-learned models against earlier baseline. | [`notebooks/05_binary_classification.ipynb`](notebooks/05_binary_classification.ipynb), [`notebooks/06_multi_class_classification.ipynb`](notebooks/06_multi_class_classification.ipynb), comparison in [`notebooks/07_comparison_binary_multiclass.ipynb`](notebooks/07_comparison_binary_multiclass.ipynb); model builder in [`src/utils/models_utils.py`](src/utils/models_utils.py) |

**Reflection**  
- What went well: Transfer learning significantly improved model convergence speed and accuracy.  
- What was difficult: Selecting which layers to freeze and adapt required experimentation and understanding of model internals.  

**Grading Level (Self-Assessment)**  
Undefined – You have yet to start addressing this Learning Outcome (not passed, 4)  
Orienting – You are beginning to address this Learning Outcome (5)  
Beginning – You have made some progress towards this Learning Outcome (6)  
**Proficient – You have made substantial progress and are competent in this Learning Outcome (8)**  
Advanced – You have fully mastered this Learning Outcome (10)  

---

## Learning Outcome 5: Show professional skills  

**Entry Level (Self-Assessment)**  
I had strong prior experience with version control (Git/GitHub), but limited practice in professional reporting.  

**Performed Activities**

| Week | Activities | Evidence |
|------|------------|----------|
| 3    | Maintained a structured GitHub repository with clear commit messages and organized code. Produced comprehensive documentation and reports summarizing methodology, results, and reflections. | Project repository at https://github.com/xemau/ADAI-individual-repository-2025 demonstrates professional code management. This README and accompanying Jupyter notebooks provide clear communication of project outcomes. |
| 4    | Refactored shared code into reusable utilities; standardized artifact logging and plotting; produced reproducible evaluation notebooks for binary and multi-class; updated portfolio evidence. | Utilities in [`src/utils/`](src/utils/); notebooks [`05_binary_classification.ipynb`](notebooks/05_binary_classification.ipynb), [`05_result_binary_classification.ipynb`](notebooks/05_result_binary_classification.ipynb), [`06_multi_class_classification.ipynb`](notebooks/06_multi_class_classification.ipynb), [`07_comparison_binary_multiclass.ipynb`](notebooks/07_comparison_binary_multiclass.ipynb); logs in `artifacts/metrics_log.csv` and `artifacts/metrics_log_multiclass.csv` |
| 5    | Documented dataset evaluation systematically in markdown with evidence links and summary table; maintained professional reporting standards. | [`evaluation.md`](evaluation.md), repository updates with plots in `artifacts/plots/` |

**Reflection**  
- What went well: Consistent documentation and code organization improved project reproducibility and clarity.  
- What was difficult: Balancing detail and conciseness in reporting required iterative refinement.  

**Grading Level (Self-Assessment)**  
Undefined – You have yet to start addressing this Learning Outcome (not passed, 4)  
Orienting – You are beginning to address this Learning Outcome (5)  
Beginning – You have made some progress towards this Learning Outcome (6)  
**Proficient – You have made substantial progress and are competent in this Learning Outcome (8)**  
Advanced – You have fully mastered this Learning Outcome (10)  
