# Predicting Chromatin Accessibility Using Neural Networks

This repository contains the experiments and results of a project focused on predicting chromatin accessibility from genomic DNA sequences using neural networks. The work was carried out during my Masters education at Aarhus University (2022).



## 🎯 Aim of the Project

The objective of this project is to **predict DNase I hypersensitivity signal enrichment** — an indicator of chromatin accessibility — using **neural networks trained on genomic DNA sequences**.  
By learning sequence-specific regulatory patterns, these models can identify **open chromatin regions** and **active promoter sites**, providing a computational alternative to costly wet-lab assays.



## 📘 Methods

### Data
- **Genome:** Human GRCh38 / hg38  
- **Signal tracks:** DNase-seq data from the **ENCODE Project**  
- **Chromosome:** 22  
- **Cell line:** Adrenal gland (M22 H3K4me3)

### Data Pre-processing
1. **One-hot encoding** of DNA bases (A,C,G,T → binary vectors)  
2. **Sliding window segmentation** (default = 201 bp)  
3. **Filtering:**  
   - Removed windows with `N` or missing values  
   - Excluded hard-to-map regions using **Umap/Bismap**  
4. **Signal transformation:** Applied `arcsinh(x)` normalization  
5. **Data split:** Train (90%)  |  Validation (5%)  |  Test (5%)

### Model Implementation
- Framework: **PyTorch**
- Optimizer: **ADAM**
- Loss function: **Mean Squared Error (MSE)**
- Regularization:  
  - Batch Normalization  
  - Dropout (p = 0.25)  
  - Early Stopping (3 epochs patience)
- Hardware: **GenomeDK** HPC cluster (GPU accelerated)

### Models
1. **Fully Connected Neural Network (FCNN)**  
   - 1–2 hidden layers  
   - Hyperparameters tuned: learning rate, hidden units, weight decay, window size  
2. **Convolutional Neural Network (CNN)**  
   - 1 convolution + 1 pooling + 1 hidden layer  
   - Hyperparameters tuned: kernel size, filter count, stride, hidden units  


## 📊 Results Summary

| Model | Key Hyperparameters | Test MSE |
|:------|:--------------------|:--------:|
| **Baseline (Random Forest)** | Random params | 0.066 |
| **FCNN (1 Hidden Layer)** | lr = 2e-4, hidden = 2000, wd = 1e-5 | 0.066 |
| **FCNN (2 Hidden Layers)** | lr = 2e-4, hidden = 2000, wd = 1e-5 | 0.076 |
| **CNN (Optimized)** :trophy: | conv_out = 10, kernel = 40, hidden = 1000, lr = 2e-4, wd = 1e-5 | **0.047** |

### Promoter vs Non-Promoter Predictions using CNN
| Region Type | Mean Signal Enrichment (± SEM) |
|:-------------|:-------------------------------:|
| Promoter | **1.130 ± 0.001** |
| Non-Promoter | **0.476 ± 0.03** |

➡️ CNN accurately predicts higher chromatin accessibility in promoter regions, aligning with biological expectations.



## 🚀 Key Insights
- Even **simple architectures** capture sequence-level determinants of chromatin accessibility.  
- **CNN outperformed FCNN**, achieving the best MSE (0.047).  
- Careful **hyperparameter tuning** (learning rate, filter size) was critical.  
- Baseline models (Linear, Lasso, Ridge) underperformed — highlighting the nonlinear nature of the problem.  
- Results confirm that **DNA sequence alone** encodes significant information about chromatin state.



## 🔜 Future Work
- Extend training to additional chromosomes and cell lines.  
- Explore **deeper and multi-layer CNNs** for motif detection.  
- Implement **automated hyperparameter optimization** (Bayesian / Optuna).  
- Predict the effect of **mutations or SNPs** on chromatin accessibility.


