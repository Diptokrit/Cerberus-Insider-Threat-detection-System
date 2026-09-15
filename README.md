# Cerberus – Insider Threat Detection System

Cerberus is a lightweight AI-based project designed to detect insider threats using the CERT r4.2 dataset. It includes both a baseline anomaly detection model and an advanced temporal deep-learning model.

---

##  What the Project Does

###  Baseline Model (Isolation Forest)
- Uses engineered behavioral and psycholinguistic features  
- Features include after-hours logons, device connections, sentiment, and total logons  
- Detects 2 out of 5 insiders (40% performance)

###  Advanced Model (Temporal Graph Network)
- Processes user activity as a time-ordered sequence  
- Combines graph neural networks (GCN) + GRU  
- Detects 3 out of 5 insiders (60% performance)

###  Dashboard
- A Streamlit dashboard for visualizing model performance and insider rankings

---

## 📂 Files Included
- `train_baseline_model.py` – baseline feature engineering + Isolation Forest  
- `prepare_temporal_data.py` – merges logs into temporal event sequences  
- `Advanced_Model_Training.ipynb` – training the Temporal Graph Network  
- `dashboard.py` – interactive Streamlit app  
- `model_results.csv` – baseline model output  
- `advanced_model_results.csv` – advanced model output

---

## 📊 Dataset
This project uses the **CERT Insider Threat Dataset r4.2**  
(a publicly available synthetic dataset for research).

---

## 🛠 How to Run
```bash
python train_baseline_model.py
python dashboard.py
