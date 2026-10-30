# Wireless Intrusion Detection System (WIDS)

A machine-learning-based Wireless Intrusion Detection System for detecting suspicious Wi-Fi traffic, with a focus on deauthentication attacks.

The system combines wireless packet analysis, feature engineering, machine-learning classification, real-time/PCAP traffic processing, automated alert generation, and a web-based monitoring dashboard.

--

## 1. Project Overview

Wireless networks are vulnerable to attacks that manipulate management frames and disrupt legitimate communication.

This project implements a prototype Wireless Intrusion Detection System (WIDS) capable of:

- Processing captured Wi-Fi traffic
- Extracting packet-level and temporal features
- Training a machine-learning classifier
- Detecting malicious wireless traffic
- Detecting deauthentication attacks
- Generating alerts when suspicious traffic is detected
- Processing both PCAP files and live wireless interfaces
- Displaying detected alerts through a web dashboard

The project uses a `HistGradientBoostingClassifier` as its primary machine-learning model.

> WIPS functionality is outside the scope of this project. The system detects and reports attacks but does not actively disconnect attackers or modify wireless infrastructure.

--

## 2. System Architecture

```text
                WPA3 Attack Dataset
                        |
                        v
              Dataset Preprocessing
                        |
                        v
                Feature Engineering
                        |
                        v
              Training / Validation
                        |
                        v
       HistGradientBoostingClassifier
                        |
                        v
                  Saved Model
                        |
                        v
        +---------------+---------------+
        |                               |
        v                               v
   PCAP Traffic                  Live Wi-Fi Traffic
        |                               |
        +---------------+---------------+
                        |
                        v
                     TShark
                        |
                        v
              Feature Extraction
                        |
                        v
                  ML Prediction
                        |
                        v
                  Alert Engine
                        |
                        v
                  alerts.csv
                        |
                        v
                 Flask Dashboard
