# Dataset Evaluation – BCN20000

- **Age Distribution**
  - Majority between **40–80 years** (~68% of all cases).
  - Training set closely mirrors overall distribution.
  - ![Age distribution (all)](/artifacts/plots/bcn20k_all_age_pie.png)
  - ![Age distribution (train)](/artifacts/plots/bcn20k_train_age_pie.png)

- **Diagnosis Distribution**
  - Most common: **NV (nevi)** ~22% of cases.
  - Malignant groups (**MEL, BCC, SCC**) ~32% overall.
  - Training set is more balanced: ~49% malignant vs. 51% benign.
  - ![Diagnosis distribution (all)](/artifacts/plots/bcn20k_all_diagnosis_bar.png)
  - ![Diagnosis distribution (train)](/artifacts/plots/bcn20k_train_diagnosis_bar.png)

- **Localization**
  - Concentrated in:
    - **Anterior torso** (39–41%).
    - **Head/neck** (26%).
  - Smaller fractions:
    - Lower/upper extremities.
    - Palms/soles.
    - Oral/genital.
  - ![Localization distribution (all)](/artifacts/plots/bcn20k_all_localization_bar.png)
  - ![Localization distribution (train)](/artifacts/plots/bcn20k_train_localization_bar.png)

- **Benign vs. Malignant**
  - **All data:** 67.8% benign vs. 32.2% malignant.
  - **Train split:** 50.9% benign vs. 49.1% malignant (well-balanced).
  - ![Benign vs malignant (all)](/artifacts/plots/bcn20k_all_malignancy_pie.png)
  - ![Benign vs malignant (train)](/artifacts/plots/bcn20k_train_malignancy_pie.png)

- **Sex Distribution**
  - Nearly even:
    - Male: 51.9% overall (52.4% train).
    - Female: 47.4% overall (47.0% train).
  - ~0.6% unknown.
  - ![Sex distribution (all)](/artifacts/plots/bcn20k_all_sex_pie.png)
  - ![Sex distribution (train)](/artifacts/plots/bcn20k_train_sex_pie.png)

---

## Summary Table (all vs train)

| Category  | N (all=18,946) | %    | N (train=12,413) | %    |
|-----------|----------------|------|------------------|------|
| Benign    | 12,849         | 67.8 | 6,316            | 50.9 |
| Malignant | 6,097          | 32.2 | 6,097            | 49.1 |
| Male      | 9,840          | 51.9 | 6,499            | 52.4 |
| Female    | 8,989          | 47.4 | 5,840            | 47.0 |

---