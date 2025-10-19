import re
import string

try:
    import spacy

    _nlp = spacy.load("en_core_web_sm")
except ImportError:
    _nlp = None
    print("spaCy not available — lemmatization will be skipped.")


def lemmatize(text):
    """Lemmatize text while preserving plural nouns."""
    if _nlp is None:
        raise ImportError
    doc = _nlp(text)
    lemmas = []
    for token in doc:
        if token.is_punct:
            continue
        # Preserve plural nouns
        if token.tag_ in ("NNS", "NNPS"):
            lemmas.append(token.text)
        # Preserve verbal adjectives (like 'cooking' in 'cooking gas')
        elif token.tag_ == "VBG" and token.dep_ == "amod":
            lemmas.append(token.text)
        else:
            lemmas.append(token.lemma_)
    return " ".join(lemmas)


def clean_whitespace(text):
    """Normalize whitespace and remove trailing punctuation."""
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[.!?]+$", "", text).strip()


def canonicalize_step(text):
    """
    Canonicalize a step description by:
    - Lowercasing
    - Lemmatizing
    - Normalizing whitespace and punctuation
    """
    text = text.lower()
    text = clean_whitespace(text)
    text = lemmatize(text)
    return clean_whitespace(text)


def generate_unique_token_labels(n, tokenizer):
    """Generate `n` unique string labels (A, B, ..., Z, AA, AB, ...) with distinct first tokens."""
    labels = []
    used_first_tokens = set()
    alphabet = string.ascii_uppercase
    i = 0

    while len(labels) < n:
        # Generate Excel-style label (A, B, ..., Z, AA, AB, ...)
        label = ""
        x = i
        while True:
            label = alphabet[x % 26] + label
            x = x // 26 - 1
            if x < 0:
                break

        tokens = tokenizer.encode(label, add_special_tokens=False)
        first_token = tokens[0] if tokens else None

        if first_token not in used_first_tokens:
            used_first_tokens.add(first_token)
            labels.append(label)

        i += 1

    return labels


def validate_step_lengths(texts):
    """Ensure all 'step' lists in `texts` have the same length."""
    iterator = iter(texts)
    expected_length = len(next(iterator)["step"])
    if not all(len(item["step"]) == expected_length for item in iterator):
        raise ValueError("Not all 'step' lists have the same length.")
    return expected_length


def format_questions(texts, tokenizer, question_template):
    """Format choice strings and instantiate questions from a list of step dicts."""
    step_count = validate_step_lengths(texts)
    token_labels = generate_unique_token_labels(step_count, tokenizer)
    for item in texts:
        steps = item["step"]
        choices_str = "\n".join(f"{label}. {step}" for label, step in zip(token_labels, steps))
        # item['step'] = choices_str
        item["choices"] = choices_str
    questions = [question_template.format(**fields) for fields in texts]
    return questions
