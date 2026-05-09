# Network Intrusion Detection System

A Machine Learning based Network Intrusion Detection System developed using Python and Streamlit. This project detects malicious network traffic and classifies traffic as normal or attack using multiple machine learning algorithms.

## Project Overview

The system analyzes network traffic data and predicts whether the traffic is safe or malicious. Multiple machine learning algorithms are trained and compared to identify the best performing model.

The project includes:

• Data preprocessing  
• Machine learning model training  
• Algorithm comparison  
• Attack prediction  
• Streamlit dashboard  
• Visualization graphs  
• Real time packet sniffing support  

## Algorithms Used

The following machine learning algorithms are implemented:

1. Random Forest  
2. Decision Tree  
3. K Nearest Neighbors  
4. Support Vector Machine  
5. XGBoost  

## Dataset Used

Dataset: NSL KDD Dataset

Download Link:  
https://www.unb.ca/cic/datasets/nsl.html

Files used:

• KDDTrain+.txt  
• KDDTest+.txt  

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

## Project Structure

```text
NIDS_Project/
│
├── dataset/
│   └── KDDTrain+.txt
│
├── model/
│   ├── nids_model.pkl
│   └── scaler.pkl
│
├── app.py
├── train_model.py
├── live_detection.py
├── packet_sniffer.py
├── requirements.txt
└── README.md