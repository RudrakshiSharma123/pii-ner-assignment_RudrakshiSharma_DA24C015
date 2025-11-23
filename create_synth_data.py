import json
import random
import os

random.seed(42)

N_TRAIN = 600
N_DEV = 150

NAMES = [
    "ramesh sharma", "sneha verma", "amit kumar", "anita patel",
    "rahul singh", "priya gupta", "vikas yadav", "neha joshi"
]

CITIES = [
    "mumbai", "delhi", "chennai", "bangalore", "kolkata",
    "pune", "hyderabad", "ahmedabad"
]

EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com"
]

def random_digits(n):
    return "".join(str(random.randint(0, 9)) for _ in range(n))

def spaced_digits(n):
    # e.g. "4 2 4 2 1 2 3 4 ..."
    return " ".join(random_digits(1) for _ in range(n))

def random_phone():
    # 10 digit phone, mix of spaced / chunked
    raw = random_digits(10)
    if random.random() < 0.5:
        return " ".join(raw)
    else:
        return raw[:5] + " " + raw[5:]

def random_credit_card():
    # 16 digit card, sometimes spaced in groups of 4
    raw = random_digits(16)
    if random.random() < 0.5:
        return " ".join(raw)  # fully spaced
    else:
        return " ".join([raw[i:i+4] for i in range(0, 16, 4)])

def stt_style_email(name):
    """
    Convert 'ramesh sharma' → 'ramesh dot sharma at gmail dot com'
    or similar STT-ish style.
    """
    base = name.replace(" ", " dot ")
    domain = random.choice(EMAIL_DOMAINS)
    # sometimes STT-style, sometimes normal email
    if random.random() < 0.6:
        # STT style with "at" and "dot"
        domain_parts = domain.split(".")
        domain_stt = " dot ".join(domain_parts)
        return f"{base} at {domain_stt}"
    else:
        # normal email format (for some variety)
        base_clean = name.replace(" ", ".")
        return f"{base_clean}@{domain}"

def make_example(idx):
    """
    Create a single synthetic example: text + entities with char offsets.
    """
    name = random.choice(NAMES)
    city = random.choice(CITIES)
    phone = random_phone()
    card = random_credit_card()
    email = stt_style_email(name)
    date = f"{random.randint(1,28)} january 2025"

    # Simple template, STT-ish, no punctuation
    templates = [
        "my name is {name} i live in {city} my phone number is {phone} my email is {email} and my credit card number is {card} and the date today is {date}",
        "this is {name} from {city} my contact number is {phone} and my email id is {email} i want to pay with card number {card} on {date}",
        "{name} speaking from {city} phone is {phone} email {email} card {card} date {date}"
    ]
    template = random.choice(templates)

    text = template.format(
        name=name,
        city=city,
        phone=phone,
        email=email,
        card=card,
        date=date,
    )

    entities = []

    def add_entity(substring, label):
        start = text.find(substring)
        if start == -1:
            return  # should not happen, but safety
        end = start + len(substring)
        entities.append({"start": start, "end": end, "label": label})

    # Add entities: PERSON_NAME, CITY, PHONE, EMAIL, CREDIT_CARD, DATE
    add_entity(name, "PERSON_NAME")
    add_entity(city, "CITY")
    add_entity(phone, "PHONE")
    add_entity(email, "EMAIL")
    add_entity(card, "CREDIT_CARD")
    add_entity(date, "DATE")

    example = {
        "id": f"utt_{idx:04d}",
        "text": text,
        "entities": entities,
    }
    return example


def write_jsonl(path, examples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    all_examples = [make_example(i) for i in range(N_TRAIN + N_DEV)]

    train_examples = all_examples[:N_TRAIN]
    dev_examples = all_examples[N_TRAIN:]

    write_jsonl("data/train.jsonl", train_examples)
    write_jsonl("data/dev.jsonl", dev_examples)

    print(f"Wrote {len(train_examples)} train examples to data/train.jsonl")
    print(f"Wrote {len(dev_examples)} dev examples to data/dev.jsonl")


if __name__ == "__main__":
    main()
