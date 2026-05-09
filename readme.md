# Hybrid Machine Learning Based Network Intrusion Detection System

A Hybrid Network Intrusion Detection System developed using Machine Learning, XGBoost, and Streamlit. This project detects malicious network traffic using both NSL KDD and TON IoT datasets.

The system supports multiple attack classifications, algorithm comparison, visualization dashboards, and real time packet monitoring.

---

## Project Overview

This project analyzes network traffic and predicts whether the traffic is normal or malicious. It supports both traditional and modern intrusion datasets for improved accuracy and better real world attack detection.

The project includes:

• NSL KDD dataset support  
• TON IoT dataset support  
• Hybrid intrusion detection  
• Multi attack classification  
• Machine learning model comparison  
• Streamlit visualization dashboard  
• Real time packet monitoring  
• Feature importance analysis  
• Confusion matrix generation  
• Downloadable prediction reports  

---

## Datasets Used

### 1. NSL KDD Dataset

Used for traditional network intrusion detection.

Download Link:  
https://www.unb.ca/cic/datasets/nsl.html

Files Used:

• KDDTrain+.txt  
• KDDTest+.txt  

---

### 2. TON IoT Dataset

Used for modern IoT and network attack detection.

Download Link:  
https://research.unsw.edu.au/projects/toniot-datasets

File Used:

• train_test_network.csv  

---

## Attack Categories

The system detects the following attack classes:

• Normal  
• DoS  
• Probe  
• R2L  
• U2R  

---

## Machine Learning Algorithms Used

The following algorithms are implemented and compared:

1. Random Forest  
2. Decision Tree  
3. K Nearest Neighbors  
4. Support Vector Machine  
5. XGBoost  

---

## Technologies Used

• Python  
• Pandas  
• NumPy  
• Scikit Learn  
• XGBoost  
• Streamlit  
• Matplotlib  
• Joblib  
• Scapy  

---

## Project Structure

```text
network-intrusion-detection-system/
│
├── dataset/
│   ├── KDDTrain+.txt
│   └── train_test_network.csv
│
├── model/
│   ├── nsl_model.pkl
│   ├── nsl_scaler.pkl
│   ├── nsl_features.pkl
│   ├── nsl_label_encoder.pkl
│   ├── ton_model.pkl
│   ├── ton_scaler.pkl
│   └── ton_features.pkl
│
├── app.py
├── train_model.py
├── train_nsl_model.py
├── requirements.txt
├── README.md
└── .gitignore
