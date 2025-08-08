# MSSAFN: A multiscale structure-aware spatial fusion network for Alzheimer’s disease classification
## Overview
Computer-aided diagnosis for Alzheimer’s disease (AD) is advancing, yet deep learning models using both sMRI and FDG-PET struggle with inefficiency, boundary information loss, and underutilizing frequency-domain features. To address this, we propose the Multi-Scale Spatial Structure Aware Fusion Network (MSSAFN). Our model features a novel Structure-Aware Spatial (SAS) module that uses 3D Dynamic Padded Window Attention to preserve critical boundary data via adaptive padding. Simultaneously, it employs the Fast Fourier Transform (FFT) to analyze features in the frequency domain, enhancing subtle pathological details before returning them to the spatial domain. MSSAFN adaptively fuses these advanced spatial and frequency features from both modalities across multiple scales. Evaluated on the ADNI dataset, MSSAFN demonstrates superior classification of AD and Mild Cognitive Impairment (MCI).
![MSSAFN](https://github.com/user-attachments/assets/6262a606-39d4-4e29-9b2c-8087ca40cda0)

## Data Preparation
Prepare your dataset according to the format expected by the MSSAFN model. The data should be split into two distinct modalities as required by the model
