"""Answer quality: EM, F1 (or graded relevance)."""


def exact_match(pred: str, ref: str) -> float:
    """1.0 if normalized strings match."""
    return 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0


def f1_score(pred: str, ref: str) -> float:
    """Token-level F1 between pred and ref."""
    p_tokens = set(pred.lower().split())
    r_tokens = set(ref.lower().split())
    if not p_tokens or not r_tokens:
        return 0.0
    common = p_tokens & r_tokens
    prec = len(common) / len(p_tokens)
    rec = len(common) / len(r_tokens)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
