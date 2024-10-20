# **Machine Translation for Extremely Low-Resourced Languages**  

## **Project Overview**  

This project focuses on developing **machine translation (MT) models** specifically for the **Tulu language**, a low-resourced Dravidian language spoken by communities in southern India. The project addresses the challenges posed by **data scarcity, linguistic complexity, and dialectal variations**. Traditional MT models are ineffective due to the absence of parallel corpora, inconsistent grammar structures, and limited computational resources for such languages. Through this repository, we aim to build robust **translation models between Tulu and Kannada**, while also laying a foundation for future work on other underrepresented languages such as **Lambadi, Bhili, and Santali**.  

The project seeks to **improve the accessibility and usability** of Tulu through machine translation, preserving its cultural significance and enabling **seamless communication and integration**. The repository contains code, datasets, model configurations, and experiment results related to **Tulu language translation**, offering a comprehensive framework for building **low-resource MT models** using **Fairseq**.

---

## **Contents**

### **1. Introduction**
- **Project Objectives:** Develop a machine translation model for **Tulu**, ensuring accurate translations even with limited labeled data.  
- **Relevance:** Enabling effective communication in Tulu through translation models promotes **accessibility, cultural preservation**, and **language integration**.  

- **Importance of MT for Low-Resource Languages:**  
  - **Preserves cultural heritage** by enabling communication in native languages.
  - **Provides access to information** in underrepresented languages.
  - Promotes **national integration** and boosts **confidence among speakers** of low-resource languages.

---

### **2. Challenges and Limitations**
- **Specific Challenges Faced in Tulu:**
  1. **Data Scarcity:** Limited availability of parallel corpora for Tulu-English or Tulu-Kannada translations.
  2. **Dialectal Variations:** Regional differences in spoken Tulu affect translation accuracy.
  3. **Complex Grammar Structures:** Non-standard sentence structures complicate the tokenization and translation processes.
  4. **Resource Constraints:** Limited computational resources and lack of linguistic research for Tulu.

- **Environment Setup and Data Preparation Challenges:**
  - Preparing the **data pipeline** for Tulu from limited datasets.
  - Ensuring data **consistency through tokenization** and cleaning processes.  

---

### **3. Environment Setup**
- **Importing Libraries and Dependencies:**
  - **Libraries Used:** Numpy, Pandas for data handling, PyTorch for model training.
  - **Dependencies:** torch, torchvision, torchaudio for deep learning infrastructure.

- **Tools Installation:**
  1. **Fairseq Framework:** Cloned from GitHub and installed in editable mode for building and training models.
  2. **Tokenizer:** Custom `ilstokenizer` to normalize and tokenize Kannada-Tulu datasets.
  3. **Moses Decoder:** Cleaned and preprocessed the training corpus, filtering out noisy and inconsistent data.

- **GPU Setup:** Verified GPU availability (`nvidia-smi`) to leverage hardware acceleration during training.  

---

### **4. Data Preparation**
- **Tokenization and Cleaning:**
  - Tokenized input datasets using `ilstokenizer` and converted all text to lowercase.
  - Used **Moses Decoder** to clean datasets, ensuring consistency by filtering out data based on sentence length and alignment.

- **Subword Tokenization with BPE:**
  - Employed **Byte Pair Encoding (BPE)** through `subword-nmt` to learn subword units, handling **out-of-vocabulary (OOV) words** effectively.
  - Generated BPE codes from combined tokenized data (`train.all.tkn`) and applied them to training and validation datasets.

- **Fairseq Preprocessing:**
  - Created a **joint dictionary** for **English-Tulu** datasets with `preprocess.py`.
  - Filtered out tokens using threshold parameters (`--thresholdtgt 0 --thresholdsrc 0`) to optimize data quality.
  - Prepared training data (`data-bin/trial`) with 20 parallel workers for efficient data handling.

---

### **5. Model Training and Evaluation**
- **Training a Transformer Model:**
  - Used Fairseq’s **Transformer architecture** (`--arch transformer_iwslt_de_en`) to translate between **Kannada and Tulu**.
  - Configured key hyperparameters:
    - **Optimizer:** Adam
    - **Learning Rate:** 0.01
    - **Dropout:** 0.3 for regularization

- **Training Procedure:**
  - Saved trained models and checkpoints (`trained_models/checkpoint_best.pt`) for evaluation and deployment.

- **Model Evaluation:**
  - Generated translations for the **validation set** using Fairseq’s `interactive.py`.
  - Evaluated translation quality with the **BLEU score metric** (`sacrebleu`) by comparing generated translations against the reference set.

---

### **6. Experiments**
#### **Initial Experiments:**
- **Task:**  
  Translate between **Tulu and Kannada** in both directions.
  
- **Parameters Explored:**
  - Encoder and decoder layers, embedding dimensions, attention heads.
  - Learning rate adjustments and optimizer configurations.

- **Result:**  
  Achieved **baseline BLEU scores**, indicating a functional translation model with initial settings.

#### **Parameter Tuning:**
- Adjusted hyperparameters (embedding size, attention heads, encoder/decoder layers), but observed **consistent BLEU scores** across different configurations.  

#### **Additional Experiments:**
- **IndicTrans2 Model:**  
  Translated between **Kannada and English**.
  
- **English-Tulu Experiments:**  
  Explored **multilingual translation** involving **Tulu-English** and **English-Tulu**, but BLEU scores showed minimal changes, emphasizing the need for **diverse data** and **alternative architectures**.

---

### **7. Outcome and Summary**
#### **Achievements:**
- Successfully built and trained **Transformer-based MT models** for **Kannada-Tulu** translation.
- Implemented effective **preprocessing pipelines** with tokenization, subword encoding, and data cleaning.
- Conducted multiple **experiments and parameter tuning**, achieving consistent performance.

#### **Key Learnings:**
- **Model Capacity:** Increasing model parameters did not significantly improve translation quality.
- **Data Quality:** High-quality, diverse data plays a crucial role in achieving meaningful translation results.
- **Parameter Sensitivity:** BLEU scores remained consistent despite hyperparameter changes, indicating the **need for advanced architectures** or **more data**.

#### **Future Work:**
1. **Data Expansion:**
   - Collect more representative and diverse datasets for Tulu.
   - Improve data curation with the help of **linguistic experts**.

2. **Model Enhancement:**
   - Experiment with **RNNs, CNNs**, and **larger Transformers** for better translation performance.
   - Implement **transfer learning** and **curriculum learning** to address data scarcity.

3. **Comprehensive Evaluation:**
   - Use **multiple evaluation metrics** (beyond BLEU) and conduct **human assessments** for better insights.
   - Perform extensive **hyperparameter optimization** and explore **multilingual models**.

4. **Collaboration and Expertise:**
   - Collaborate with **linguistic experts** to curate datasets and validate translations.
   - Explore **resource-sharing initiatives** with other research groups.

---

### **8. Files and Directories**
- **Code:**  
  Contains scripts for **environment setup, data preparation, training, and evaluation**.
  
- **Data:**  
  Includes **training and validation datasets** for English, Tulu, and Kannada.

- **Figures:**  
  Visualizations such as **sentence length distribution**, **token distribution**, and **language patterns**.

---

## **Conclusion**
This project demonstrated the feasibility of building **machine translation models for the Tulu language**, overcoming challenges associated with **data scarcity and linguistic complexity**. The repository offers a comprehensive framework to develop **low-resource MT models** using **Fairseq**, providing practical insights for future work on **multilingual translation systems**.

By improving **access to Tulu through machine translation**, the project aims to **promote national integration** and **preserve the cultural heritage** of underrepresented communities.

For detailed information, refer to the code and datasets provided within this repository.


