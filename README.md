# Real-Time Network Anomaly Detection Through Predictive Analytics 🚨🔐

This project aims to detect and classify network anomalies in real-time using **machine learning techniques**, particularly the **Random Forest classifier**, with the **NSL-KDD dataset**. A Streamlit-based user interface provides an interactive environment for end users to input traffic parameters and receive real-time predictions on the type of network activity (normal or specific attack type).

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [System Architecture](#system-architecture)
- [Installation & Setup](#installation--setup)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [User Interface](#user-interface)
- [Results](#results)
- [Limitations](#limitations)
- [Future Scope](#future-scope)
- [Contributors](#contributors)

---

## ✅ Project Overview

Real-time anomaly detection plays a crucial role in modern cybersecurity. This project leverages **predictive data analytics** and **supervised learning** to classify network connections as normal or part of known attack categories such as:

- DoS (Denial of Service)
- Probe (Surveillance)
- R2L (Remote to Local)
- U2R (User to Root)

The model is trained on **NSL-KDD**, a benchmark intrusion detection dataset, and integrated with a front-end dashboard using **Streamlit**.

---

## 🌟 Key Features

- Real-time prediction of network anomalies
- Preprocessing pipeline: cleaning, encoding, normalization
- Feature selection and visualization
- Categorization of over 20+ attack types into 4 major classes
- Interactive web UI using Streamlit
- Visualization of traffic type distribution

---

## 💻 Technologies Used

- **Python 3.10+**
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **Scikit-learn** – Random Forest Classifier
- **Streamlit** – Web UI
- **VS Code** – Development Environment

---

## 🧠 System Architecture

