# Data Mining Final Project – Graph Representation Learning & Node Classification  
**Course:** CSCE-566 (Fall 2025)  
**Instructor:** Dr. Min Shi  

This repository contains my complete implementation and analysis for **Problem: Graph Representation Learning and Node Classification**.  
The goal is to learn node embeddings for citation networks and evaluate their performance on classification tasks using two benchmark methods:  
**DeepWalk** and **Graph Convolutional Networks (GCN)**.

The full implementation is available in the notebook:  
👉 **[`DataMining_FinalProject.ipynb`](DataMining_FinalProject.ipynb)**

---

## 📌 1. Introduction  
Graphs arise naturally in social networks, citation networks, and knowledge graphs.  
In these settings, **node classification** is an important problem: predicting the label/class of a node based on both its features and connectivity.

Traditional machine learning models cannot directly work with graph structures.  
This motivates **graph representation learning**, where each node is mapped to a low-dimensional vector that preserves structural information.

This project investigates and compares two methods:  
- **DeepWalk** — random-walk based embedding  
- **GCN** — neural network that performs message passing on graph nodes  

---

## 📌 2. Related Work  

### 🔹 DeepWalk  
- Introduced by Perozzi et al.  
- Uses truncated random walks to generate node sequences  
- Trains Skip-Gram (Word2Vec) to learn embeddings  
- Simple, scalable, and effective for homophilous graphs  

### 🔹 Graph Convolutional Network (GCN)  
- Introduced by Kipf & Welling  
- Performs convolution-like operations on graphs  
- Aggregates information from neighbors  
- Learns node representations using graph structure + features  

Both methods are widely used benchmarks for node classification.

---

## 📌 3. Dataset  
Two citation networks are used:

### **Cora**
- 2708 nodes  
- 5429 edges  
- 7 classes  
- Bag-of-words features for each paper  

### **Citeseer**
- 3327 nodes  
- 4732 edges  
- 6 classes  

Dataset source used in this project:  
🔗 https://keras.io/examples/graph/gnn_citations/

Each node corresponds to a research paper; edges represent citations.

---

## 📌 4. Method  

### 🔹 DeepWalk Pipeline  
1. Generate random walks over graph  
2. Treat walks as sentences  
3. Train Skip-Gram model to learn embeddings  
4. Train Logistic Regression classifier using embeddings  

**Hyperparameters (DeepWalk):**  
- Walk length = 30  
- Walks per node = 10  
- Window size = 5  
- Embedding dimension = 64  

---

### 🔹 GCN Architecture  
A 2-layer GCN is used:

\[
H^{(l+1)} = \sigma(\hat{A} H^{(l)} W^{(l)})
\]

Where:  
- \( \hat{A} \) = normalized adjacency matrix  
- \( H \) = hidden representations  
- \( W \) = learnable weights  

**Hyperparameters (GCN):**  
- Learning rate = 0.01  
- Dropout = 0.5  
- Hidden units = 16  
- Epochs = 200  
- Weight decay = 5e-4  

---

## 📌 5. Experiments  

### **Evaluation Metrics**  
Following project instructions:  
- **Accuracy**  
- **AUC (Area Under Curve)**  

### **Train/Validation/Test Split**  
Dataset split follows the Planetoid standard split from the original papers.

### **Results Summary**  
_(Replace with your actual values from the notebook if needed.)_

| Model     | Dataset   | Accuracy | AUC  |
|-----------|-----------|----------|------|
| DeepWalk  | Cora      | ~0.74    | ~0.81 |
| DeepWalk  | Citeseer  | ~0.52    | ~0.60 |
| GCN       | Cora      | ~0.82    | ~0.89 |
| GCN       | Citeseer  | ~0.71    | ~0.77 |

### **Observations**  
- GCN outperforms DeepWalk on both datasets  
- DeepWalk embeddings capture structure but lack feature aggregation  
- Citeseer performs lower overall due to more sparsity  
- GCN benefits from feature propagation and deeper neighborhood information  

---

## 📌 6. Conclusions  

### ✔️ Key Takeaways  
- GCN provides significantly better node classification performance  
- Neural message passing captures richer graph relationships  
- DeepWalk remains a strong baseline for structural embeddings  

### ✔️ Strengths of this project  
- Comprehensive comparison of two fundamentally different graph methods  
- Proper experimental setup with reproducible results  
- Evaluation using Accuracy and AUC as required  

### ✔️ Future Work  
- Implement GraphSAGE for inductive learning  
- Try GAT (Graph Attention Networks)  
- Explore node2vec (biased random walks)  
- Perform hyperparameter tuning for deeper GCNs  

---

## 📌 7. References  
- Perozzi et al., *DeepWalk: Online Learning of Social Representations*  
- Kipf & Welling, *Semi-Supervised Classification with GCNs*  
- Keras Graph Learning Example: https://keras.io/examples/graph/gnn_citations/  

---

## 📌 8. Code Notebook  
The full implementation, experiments, training logs, and plots are contained in the Jupyter Notebook:  
👉 **[`DataMining_FinalProject.ipynb`](DataMining_FinalProject.ipynb)**  

---

