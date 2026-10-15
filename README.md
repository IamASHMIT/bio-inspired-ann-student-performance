# Problem 14: Student Performance Prediction using an Artificial Neural Network
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1l0hCaphfXd9qwCbDTgk-Gi9wIR-72L7O?usp=sharing)

**Name:** Ashmit Bhadoria  
**Register No.:** RA2311003010286  
**Course:** Bio Inspired Computing  
**Algorithm:** Artificial Neural Network (ANN), with a Genetic Algorithm (GA) used to evolve its hyper-parameters

## Problem
Train an ANN to predict a student's performance category (**At-Risk / Average / High**) from academic and attendance data.

## Approach
1. **Data**: 1500 synthetic students (`students.csv`) with 6 features: attendance %, weekly study hours, previous CGPA, midterm, assignment and quiz scores. A non-linear attendance penalty is built in. To use real data, set `USE_OWN_CSV = True` in the notebook.
2. **EDA**: class balance, feature distributions per class, correlation heatmap, scatter plots.
3. **Pre-processing**: stratified 70/15/15 split, standardisation fitted on the training set only.
4. **ANN from scratch (NumPy)**: He initialisation, ReLU hidden layers, softmax output, cross-entropy + L2 loss, back-propagation, Adam optimiser, mini-batches, early stopping. Back-propagation is verified with a numerical gradient check.
5. **Evaluation**: accuracy, macro-F1, confusion matrices, ROC/AUC, per-class precision/recall/F1.
6. **Interpretation**: permutation feature importance, PCA view of errors, first-layer weight map.
7. **Original contribution**: a Genetic Algorithm (tournament selection, uniform crossover, Gaussian mutation, elitism) searches the network size, learning rate and L2 strength, compared against random search with an equal budget.
8. **Comparison and robustness**: Logistic Regression, Random Forest, scikit-learn MLP; 5-seed mean +/- std.

## Results (test set, 225 students)
| Model | Accuracy |
|---|---|
| Baseline ANN [32,16] | about 0.82 |
| GA-tuned ANN | about 0.83 |

Both ANNs land around 83% (+/- 1%) over 5 seeds. The synthetic labels contain noise, which limits how high any model can go. The GA's main benefit is finding a much smaller network with equal accuracy. Exact numbers are in the executed notebook.

## How to run
1. Open `Problem14_Student_Performance_ANN.ipynb` in Google Colab (File > Upload notebook).
2. Runtime > Run all (about 1 minute, CPU is enough). No extra installs needed.

## Files
- `Problem14_Student_Performance_ANN.ipynb`: full notebook with outputs and figures
- `students.csv`: generated dataset
- `README.md`: this file
