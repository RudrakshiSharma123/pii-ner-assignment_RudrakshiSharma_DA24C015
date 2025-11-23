# PII NER on Noisy STT Transcripts

This repository contains a complete solution for **PII Named Entity Recognition (NER)** on noisy Speech-to-Text (STT) transcripts.  
The model detects PII entities and returns **character-level spans** together with a **pii: true/false** flag for each span.

---

## Model

**Model used:** `google/bert_uncased_L-4_H-256_A-4` (BERT-Mini)

Chosen because it provides:

- very **high PII precision**, and  
- **very low CPU latency**, meeting the p95 ≤ 20 ms requirement.

The trained model and tokenizer are saved in:

out_mini/

yaml
Copy code

---

## How to Run

### 1. Install requirements

```bash
pip install -r requirements.txt
2. Generate synthetic train & dev datasets
bash
Copy code
python create_synth_data.py
This creates:

data/train.jsonl — 600 synthetic examples

data/dev.jsonl — 150 synthetic examples

3. Train the model
bash
Copy code
python src/train.py \
  --model_name google/bert_uncased_L-4_H-256_A-4 \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out_mini
4. Run prediction
bash
Copy code
python src/predict.py \
  --model_dir out_mini \
  --input data/dev.jsonl \
  --output out_mini/dev_pred.json
5. Evaluate span-level performance
bash
Copy code
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out_mini/dev_pred.json
6. Measure latency
bash
Copy code
python src/measure_latency.py \
  --model_dir out_mini \
  --input data/dev.jsonl
Final Metrics (Dev Set)
Metric	Score
Entity-wise F1	1.00
PII Precision	1.00
PII Recall	1.00
PII F1	1.00
Macro F1	1.00

All metrics measured on the synthetic dev set generated using create_synth_data.py.

Latency (CPU)
Using BERT-Mini (google/bert_uncased_L-4_H-256_A-4) with max_length=64:

p50 = 3.29 ms

p95 = 4.28 ms

(Meets requirement: p95 ≤ 20 ms per utterance)

Approach Summary
Generated a noisy STT-style PII dataset with patterns like spaced digits and "at gmail dot com".

Trained a BERT-Mini token classification model using BIO tags for:
CREDIT_CARD, PHONE, EMAIL, PERSON_NAME, DATE, CITY, LOCATION.

Converted token predictions into character-level spans.

Added light post-processing to remove clearly invalid spans.

Achieved perfect detection on the synthetic dev set with very low latency.

Repository Structure
graphql
Copy code
src/                  # training, prediction, evaluation, latency scripts
data/                 # synthetic training & dev data
out_mini/             # trained BERT-Mini model + predictions
create_synth_data.py  # synthetic STT-style PII generator
requirements.txt
README.md
