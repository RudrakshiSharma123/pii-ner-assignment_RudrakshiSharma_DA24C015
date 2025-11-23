This repository contains a complete solution for PII Named Entity Recognition (NER) on noisy Speech-to-Text (STT) transcripts.
The model identifies PII entities and returns character-level spans along with PII flags.

Model Used

google/bert_uncased_L-4_H-256_A-4 (BERT-Mini)
Chosen because it offers high PII precision and very low CPU latency.

The trained model is saved in:

out_mini/

How to Run
1. Install requirements
pip install -r requirements.txt

2. Generate synthetic train & dev datasets
python create_synth_data.py

3. Train the model
python src/train.py \
  --model_name google/bert_uncased_L-4_H-256_A-4 \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out_mini

4. Run prediction
python src/predict.py \
  --model_dir out_mini \
  --input data/dev.jsonl \
  --output out_mini/dev_pred.json

5. Evaluate
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out_mini/dev_pred.json

6. Measure latency
python src/measure_latency.py \
  --model_dir out_mini \
  --input data/dev.jsonl

Final Metrics
Metric	Score
Entity-wise F1	1.00
PII Precision	1.00
PII Recall	1.00
PII F1	1.00
Macro F1	1.00

(All metrics measured on synthetic dev set.)

Latency (CPU)

Using BERT-Mini (google/bert_uncased_L-4_H-256_A-4) with max_length=64:

p50 = 3.29 ms

p95 = 4.28 ms
(meets the requirement: p95 ≤ 20 ms)

Approach Summary

Created synthetic noisy STT-style PII dataset (spelled-out digits, “at gmail dot com”, etc.).

Trained a BERT-Mini token classification model using BIO tagging.

Converted token predictions to character spans.

Added light filtering to improve precision on PII types.

Achieved perfect detection on the synthetic dev set with extremely low latency.

Repository Structure
src/                  # training, prediction, evaluation, latency scripts  
data/                 # synthetic training and development data  
out_mini/             # trained BERT-mini model + predictions  
create_synth_data.py  # data generator  
README.md
