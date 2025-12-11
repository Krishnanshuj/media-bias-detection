# media-bias-detection
📘 Media Bias Detection in News Headlines
📰 Classifying Headlines as Left-Leaning, Neutral, or Right-Leaning Using a Fine-Tuned Transformer

Media bias influences how news is interpreted, shaping public opinion and political narratives.
This project uses NLP + Transformers to classify political bias in news headlines into:

Left-Leaning

Neutral

Right-Leaning

The model is built using DistilBERT (HuggingFace) and fine-tuned on a curated political bias dataset.

🚀 Project Highlights

Fine-tuned transformer model (DistilBERT) for 3-class political bias detection

6,000 balanced samples (2,000 per class)

74% test accuracy

Clean, modular code with training, preprocessing, and prediction support

Built with Transformers, PyTorch, Datasets, KaggleHub, Scikit-Learn
🧠 Problem Definition

Goal:
Predict whether a given news headline exhibits left, neutral, or right-leaning bias.

Input:
"Government passes new healthcare bill in tight vote"

Output:
"Left-Leaning"

📊 Dataset

Dataset Source: Kaggle — Labelled Corpus Political Bias
Downloaded using:

import kagglehub
path = kagglehub.dataset_download("surajkarakulath/labelled-corpus-political-bias-hugging-face")


Dataset contains TXT files under:

Left Data/

Right Data/

Center Data/

Merged and processed into a final CSV:

political_bias_3class.csv
- headline (text)
- bias_3class (Left-Leaning, Neutral, Right-Leaning)


Each class: 2,000 headlines

🧼 Preprocessing Steps

Load raw .txt headlines

Remove empty or duplicate entries

Assign class labels based on folder name

Shuffle and balance the dataset

Save into CSV format

🤖 Model: Fine-Tuned DistilBERT

We use DistilBERT, a lighter BERT model:

40% smaller

60% faster

97% of BERT’s performance

Training code: train_transformer_multiclass.py

🏋️ Training Summary
Train samples: 4,800
Test samples:  1,200
Epochs:        3
Batch size:    16
Learning rate: 2e-5 → linear decay

🧪 Final Evaluation (Test Set)
Class	Precision	Recall	F1-Score
Left-Leaning	0.72	0.74	0.73
Neutral	0.77	0.75	0.76
Right-Leaning	0.74	0.73	0.74
✅ Overall Accuracy: 74%
🔮 Example Prediction

Input:

"Government announces new health policy for all citizens."


Output:

Predicted: Neutral (confidence = 0.396)

▶️ How to Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Prepare the dataset
python prepare_media_dataset.py

3️⃣ Train the transformer model
python train_transformer_multiclass.py

4️⃣ Use model for inference

(If you add predict.py, for example):

python predict.py "The president criticized corporate tax breaks"
