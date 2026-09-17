# Evaluation boundary

Original evaluated system: commit `6772707`. Hardened system: commit `1aae20c`.
Gold set verified against `GOLDEN_LOCK.json` (`6d78823ac42c95c6...`).
Replay configuration: golden set v1, PII masker v2 (post-evaluation hardening: corrected masker).

Hardened replay: cache only, 0 network connection attempts, 6 cache misses (network block control: PASS). The original run recorded 6 generator failures; failed responses are never cached, so each is a cache miss here and escalates as `generator_failed` exactly as it did originally (the decision comparison below includes those rows).

| Comparison | Rows | Decision diffs | Routing diffs | Intent diffs | Reply diffs | Changed | Unchanged |
|---|---|---|---|---|---|---|---|
| original vs committed | 800 | 0 | 0 | 0 | 0 | 0 | 800 |
| committed vs hardened | 800 | 0 | 0 | 0 | 0 | 2 | 798 |
| original vs hardened | 800 | 0 | 0 | 0 | 0 | 0 | 800 |

Compared fields: escalate, reason, intent, message_sha256, reply_sha256, security_sensitive, context_sufficient, evidence, grounding_passed, derived_text_features, intent_confidence, retrieval_confidence, model_confidence. Score fields did not exist at the evaluation commit, so the original comparisons cover the remaining fields (not comparable: intent_confidence, retrieval_confidence, model_confidence). Latency and token usage are excluded: they are not outputs.

Negative control (one flipped decision is reported): FAIL.

Source changes since the evaluation commit:

```
scripts/evaluate_golden.py              | 48 ++++++++++++++++--
 src/hiver_support/agent/agent.py        | 45 +++++++++++++++--
 src/hiver_support/evaluation/metrics.py | 87 +++++++++++++++++++++++++++++++++
 3 files changed, 171 insertions(+), 9 deletions(-)
```
