# Milestone Log

Each milestone records its plan before implementation and its result after. A milestone is complete
only when its acceptance criteria pass, the full regression suite is green, and output has been
manually inspected.

---

## MILESTONE 1 — Project scaffold + conversation reconstruction

**Goal:** A tested project skeleton that turns raw TWCS tweet rows into reconstructed conversation
threads and per-brand customer→support pairs.

**SPEC:** `SPEC.md` §3.1 (pipeline: raw tweets → thread reconstruction → brand extraction →
customer/support pairing).

The TWCS schema is `tweet_id, author_id, inbound, created_at, text, response_tweet_id,
in_response_to_tweet_id`. Reconstruction must handle the corpus's real quirks: `response_tweet_id`
is a comma-separated list; `in_response_to_tweet_id` is frequently absent; replies can be orphaned
(pointing at a tweet not present in a subsample); ids appear as both ints and strings; and
malformed rows can form cycles.

**ACCEPTANCE CRITERIA:**
1. Raw rows parse into a typed `Tweet` model, with TWCS date format and multi-valued
   `response_tweet_id` handled.
2. Conversations are reconstructed by reply-linkage; every tweet belongs to exactly one
   conversation; tweets within a conversation are chronologically ordered.
3. Cyclic reply links terminate rather than hang.
4. Orphan replies (dangling `in_response_to`) become their own conversation root, not a crash.
5. Brand is identified per conversation from the non-inbound author; multi-brand and brand-less
   conversations are handled explicitly.
6. Customer→support pairs are extracted with preceding turns as context; a pair always has an
   inbound customer tweet and a following outbound brand reply.
7. Subsampling is deterministic under a fixed seed.

**TEST PLAN:** `tests/test_thread_reconstruction.py`, `tests/test_brand_filtering.py` — unit tests
per criterion plus adversarial fixtures for cycles, orphans, mixed id types, and empty input.

**FILES EXPECTED TO CHANGE:** `pyproject.toml`, `requirements.txt`, `src/hiver_support/__init__.py`,
`src/hiver_support/config.py`, `src/hiver_support/data/schema.py`,
`src/hiver_support/data/threads.py`, `tests/*`.

**RISKS:** Assuming a schema without seeing real rows — mitigated by validating against the real
Kaggle file before declaring the milestone complete.

**DONE WHEN:** all criteria pass, suite green, and reconstruction verified by eye on real rows.

**RESULT:** _pending_
