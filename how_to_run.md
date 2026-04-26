# How to Run the ABSA Customer Feedback Intelligence System

## 1. Project Overview

This project implements an Aspect-Based Sentiment Analysis pipeline using DistilBERT. It performs two main tasks:

1. **Aspect Term Extraction (ATE)**: identifies product or service aspects mentioned in a review sentence.
2. **Aspect Sentiment Classification (ASC)**: predicts the sentiment for each extracted aspect as `negative`, `neutral`, or `positive`.

The project uses the SemEval-2014 Laptop and Restaurant review datasets stored inside the `data/` folder.


## 2. Environment Setup

Create and activate a Python virtual environment.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages.

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn tqdm
```


## 3. Run Training and Evaluation

From inside the `satwik_absa` folder, run:

```bash
python main.py
```

The script will:

1. Load the SemEval-2014 Laptop and Restaurant datasets.
2. Generate exploratory data analysis graphs.
3. Build train, validation and test dataloaders.
4. Train the ATE DistilBERT token classification model, saving the best checkpoint (by validation span F1) to `models/ate/`.
5. Train the ASC DistilBERT sequence classification model, saving the best checkpoint (by validation macro-F1) to `models/asc/`.
6. Evaluate both models on the test set.
7. Save the result graphs inside the `graphs/` folder.

Models are saved automatically during training — no extra steps required.

## 4. Run Inference on a Single Sentence

After training, pass a sentence as a positional argument:

```bash
python inference.py "The laptop screen is excellent but the battery life is poor."
```


## 5. Run Inference on Multiple Sentences

To process a plain-text file with one sentence per line:

```bash
python inference.py --file reviews.txt
```

Each sentence is printed with its extracted aspects and predicted sentiments.

## 7. GPU Usage

The project automatically uses GPU if CUDA is available. Otherwise, it runs on CPU.

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```