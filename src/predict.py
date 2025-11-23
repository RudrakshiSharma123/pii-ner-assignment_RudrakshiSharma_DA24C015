import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import re
import os

# Very simple regexes just to catch clearly invalid PII (currently used very lightly)
CREDIT_CARD_RE = re.compile(r"(?:\d[ -]?){13,19}")
PHONE_RE = re.compile(r"(?:\+?\d[ -]?){7,15}")
EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


def bio_to_spans(text, offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        # skip special tokens like [CLS]/[SEP] which usually have (0, 0)
        if start == 0 and end == 0:
            continue

        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                # BIO violation: close previous span (if any) and start new one
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


def filter_spans(text, spans):
    """
    Helper to very lightly clean spans.
    We only drop clearly impossible PII, and keep it STT-friendly.
    Regex is *not* enforced as a hard requirement, since STT emails/phones/cards
    can be written in very noisy ways (e.g., 'at', 'dot', spaced digits).
    """
    filtered = []
    for start, end, lab in spans:
        substr = text[start:end].strip()
        if not substr:
            # empty or whitespace-only span – always drop
            continue

        # CREDIT_CARD: must contain at least some digits
        if lab == "CREDIT_CARD":
            digits = [c for c in substr if c.isdigit()]
            if len(digits) == 0:
                # no digits at all → extremely unlikely to be a card number
                continue
            # We do NOT require CREDIT_CARD_RE to match because STT may be messy

        # PHONE: must contain a few digits
        elif lab == "PHONE":
            digits = [c for c in substr if c.isdigit()]
            if len(digits) < 3:
                # too few digits to be a phone number
                continue
            # Again, we don't hard-require PHONE_RE

        # EMAIL: STT often gives 'name at gmail dot com' (no '@'),
        # so we only drop obviously garbage (very short strings).
        elif lab == "EMAIL":
            if len(substr) < 5:
                continue  # too short to be any kind of email phrase

        # For PERSON_NAME, DATE, CITY, LOCATION – trust the model completely.

        filtered.append((start, end, lab))

    return filtered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name
    )
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

            # BIO -> spans
            spans = bio_to_spans(text, offsets, pred_ids)
            # Apply light filtering to drop only clearly impossible spans
            spans = filter_spans(text, spans)

            ents = []
            for s, e, lab in spans:
                ents.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
