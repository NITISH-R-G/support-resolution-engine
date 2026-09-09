"""Leakage guards for the evaluation pipeline.

Every check here **raises**. A warning that scrolls past in a log is not a guard, and the
public field demonstrates the cost of not having one: two of the seven repositories audited
in ``docs/PUBLIC_REPO_COMPARISON.md`` published headline numbers produced by grading a model
against its own output, and neither had anything that could have caught it.

The checks fall into two groups:

* **Data leakage** — the evaluation set and the retrieval corpus must share no ids, no
  duplicate text, no near-duplicates, no question/answer pairs, and no overlapping time
  range.
* **Model leakage** — the model that pre-annotates gold, the model that judges, and the
  model under test must be genuinely independent. Same-family counts as the same model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from hiver_support.data.schema import SupportPair

DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.90


class LeakageError(AssertionError):
    """Raised when evaluation integrity is compromised. Never caught inside the pipeline."""


# --------------------------------------------------------------------------- text handling

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalise_for_comparison(text: str) -> str:
    """Aggressively normalise text for duplicate detection only.

    Deliberately lossier than the pipeline's own normalisation: casing, punctuation,
    mentions and URLs all carry signal for the classifier but are exactly the decorations
    that let the *same* message look like two different ones to a naive duplicate check.
    """
    lowered = text.lower()
    lowered = _URL_RE.sub(" ", lowered)
    lowered = _MENTION_RE.sub(" ", lowered)
    lowered = _NON_ALNUM_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", lowered).strip()


# ----------------------------------------------------------------------------- data guards


def assert_no_id_overlap(corpus: Sequence[SupportPair], golden: Sequence[SupportPair]) -> None:
    """No pair, conversation, or customer may appear in both splits.

    Customer overlap matters as much as pair overlap: the same person phrases their
    complaints the same way and often raises the same issue twice, so a shared customer
    leaks both style and content across the split.
    """
    for field in ("pair_id", "conversation_id"):
        overlap = {getattr(p, field) for p in corpus} & {getattr(p, field) for p in golden}
        if overlap:
            raise LeakageError(
                f"{field} overlap between retrieval corpus and golden set: "
                f"{sorted(overlap)[:5]} ({len(overlap)} total)"
            )

    corpus_customers = {p.customer_tweet.author_id for p in corpus}
    golden_customers = {p.customer_tweet.author_id for p in golden}
    shared_customers = corpus_customers & golden_customers
    if shared_customers:
        raise LeakageError(
            f"customer_id overlap between retrieval corpus and golden set: "
            f"{sorted(shared_customers)[:5]} ({len(shared_customers)} total)"
        )


def assert_no_text_duplicates(corpus: Sequence[SupportPair], golden: Sequence[SupportPair]) -> None:
    """No golden question may appear verbatim in the corpus once normalised."""
    corpus_texts = {normalise_for_comparison(p.customer_text) for p in corpus}
    corpus_texts.discard("")

    for pair in golden:
        normalised = normalise_for_comparison(pair.customer_text)
        if normalised and normalised in corpus_texts:
            raise LeakageError(
                f"duplicate customer message across splits (golden pair {pair.pair_id!r}): "
                f"{normalised[:80]!r}"
            )


def assert_no_near_duplicates(
    corpus: Sequence[SupportPair],
    golden: Sequence[SupportPair],
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> None:
    """No golden question may be a near-paraphrase of a corpus question.

    Character n-grams rather than words, because the near-duplicates that actually occur
    here are typos, truncations and small edits rather than genuine synonym substitution.
    """
    if not corpus or not golden:
        return

    # Imported lazily so the cheap guards stay usable without the ML stack loaded.
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus_texts = [normalise_for_comparison(p.customer_text) for p in corpus]
    golden_texts = [normalise_for_comparison(p.customer_text) for p in golden]

    if not any(corpus_texts) or not any(golden_texts):
        return

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    corpus_matrix = vectorizer.fit_transform(corpus_texts)
    golden_matrix = vectorizer.transform(golden_texts)

    # Both matrices are L2-normalised by TfidfVectorizer, so the dot product is cosine.
    similarities = (golden_matrix @ corpus_matrix.T).toarray()

    for golden_index, row in enumerate(similarities):
        best_index = int(row.argmax())
        score = float(row[best_index])
        if score >= threshold:
            raise LeakageError(
                f"near-duplicate across splits at cosine {score:.3f} >= {threshold}: "
                f"golden {golden[golden_index].pair_id!r} vs corpus "
                f"{corpus[best_index].pair_id!r}\n"
                f"  golden: {golden_texts[golden_index][:80]!r}\n"
                f"  corpus: {corpus_texts[best_index][:80]!r}"
            )


def assert_no_response_leakage(
    corpus: Sequence[SupportPair], golden: Sequence[SupportPair]
) -> None:
    """No golden question/answer *combination* may exist in the corpus.

    Matching on the reply alone would be wrong rather than merely strict: brands send the
    same canned reply ("please DM us") thousands of times, so a reply-only check fires
    constantly on legitimate data and would have to be disabled -- which is how guards die.
    Pairing the question with the answer catches genuine reposts and duplicated rows while
    staying silent on canned replies.
    """
    corpus_pairs = {
        (normalise_for_comparison(p.customer_text), normalise_for_comparison(p.support_text))
        for p in corpus
    }

    for pair in golden:
        key = (
            normalise_for_comparison(pair.customer_text),
            normalise_for_comparison(pair.support_text),
        )
        if all(key) and key in corpus_pairs:
            raise LeakageError(
                f"response leakage: golden pair {pair.pair_id!r} has its question and answer "
                f"already present in the retrieval corpus"
            )


def assert_temporal_split(corpus: Sequence[SupportPair], golden: Sequence[SupportPair]) -> None:
    """The whole corpus must predate the whole golden set.

    A random split lets the system retrieve a resolution written after the question it is
    answering, which inflates every downstream number and cannot happen in production.
    """
    if not corpus or not golden:
        return

    latest_corpus = max(p.customer_tweet.created_at for p in corpus)
    earliest_golden = min(p.customer_tweet.created_at for p in golden)

    if latest_corpus >= earliest_golden:
        raise LeakageError(
            f"temporal leakage: retrieval corpus extends to {latest_corpus.isoformat()}, "
            f"at or beyond the earliest golden example {earliest_golden.isoformat()}"
        )


# ---------------------------------------------------------------------------- model guards

_MODEL_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("anthropic", ("claude",)),
    ("openai", ("gpt", "o1-", "o3-", "davinci")),
    ("google", ("gemini", "gemma")),
    ("qwen", ("qwen",)),
    ("meta", ("llama",)),
    ("mistral", ("mistral", "mixtral")),
    ("deepseek", ("deepseek",)),
)


def model_family(model_name: str) -> str:
    """Map a model id to its family.

    An unrecognised model returns a value unique to that name rather than a shared
    "unknown" bucket. Collapsing unknowns together would make two genuinely different
    models compare equal and fire the guard spuriously; more dangerously, the reverse
    default would let two *related* unknown models pass as independent.
    """
    lowered = model_name.lower()
    for family, prefixes in _MODEL_FAMILIES:
        if any(prefix in lowered for prefix in prefixes):
            return family
    return f"unknown:{lowered}"


def assert_independent_models(
    *,
    generator: str,
    judge: str,
    pre_annotator: str,
    system_under_test: str,
) -> None:
    """Enforce methodological independence between model roles.

    Two distinct failures are blocked. A judge from the generator's family exhibits
    self-preference, scoring its own family's output higher. A pre-annotator that is the
    model under test produces gold labels that *are* the model's predictions, so the
    reported accuracy measures the annotator's edit rate -- the failure proved numerically
    in ``docs/PUBLIC_REPO_COMPARISON.md`` §2.2.
    """
    if model_family(judge) == model_family(generator):
        raise LeakageError(
            f"judge {judge!r} shares model family {model_family(judge)!r} with generator "
            f"{generator!r}; self-preference bias makes reply-quality scores unusable"
        )

    if model_family(pre_annotator) == model_family(system_under_test):
        raise LeakageError(
            f"pre-annotator {pre_annotator!r} shares model family "
            f"{model_family(pre_annotator)!r} with the system under test "
            f"{system_under_test!r}; gold labels would be the model's own predictions"
        )


# ------------------------------------------------------------------------------ aggregate


def run_all_checks(
    corpus: Sequence[SupportPair],
    golden: Sequence[SupportPair],
    *,
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> None:
    """Run every data guard and raise once, listing all violations.

    Aggregating rather than stopping at the first failure matters in practice: leakage
    usually arrives in clusters from one bad split, and fixing them one error message at a
    time wastes a full pipeline run per violation.
    """
    checks = (
        ("id overlap", lambda: assert_no_id_overlap(corpus, golden)),
        ("text duplicates", lambda: assert_no_text_duplicates(corpus, golden)),
        (
            "near duplicates",
            lambda: assert_no_near_duplicates(corpus, golden, threshold=near_duplicate_threshold),
        ),
        ("response leakage", lambda: assert_no_response_leakage(corpus, golden)),
        ("temporal split", lambda: assert_temporal_split(corpus, golden)),
    )

    failures: list[str] = []
    for name, check in checks:
        try:
            check()
        except LeakageError as error:
            failures.append(f"- [{name}] {error}")

    if failures:
        raise LeakageError(
            f"{len(failures)} leakage check(s) failed:\n" + "\n".join(failures)
        )
