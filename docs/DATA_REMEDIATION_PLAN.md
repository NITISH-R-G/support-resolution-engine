# Dataset-text remediation plan

**Status (2026-09-17): EXECUTED on a local branch, with history rewrite prepared but NOT pushed.**
The plan below is kept as written for review. What was actually done, and every proof, is in
`docs/RELEASE_AUDIT.md` §19. Deviations from this plan, each decided by the owner or forced by a
measurement:

- **agent_llm replies are not kept** (owner decision). All replies, messages, evidence texts and
  judge and pre-annotator rationales are hashed, and rebuilt locally from the dataset and the
  local response cache.
- **The text-free candidates keep their path** (`data/golden/candidates.jsonl`), so history and
  HEAD share one format. v1 and v2 text hashes live in each record.
- **The codebook edit scripts were proven**: 29/29 examples rebuilt, reproducing taxonomy hash
  `613f5dfe…`.
- **The evaluation files were proven**: predictions, judge and partial judge rebuilt
  byte-identical to their original hashes.
- **A full-corpus (all brands) quote scan** found 10 edited or other-brand quotes the
  AppleSupport-only detector missed; they were replaced by hand. Two commit messages quoting
  dataset text were reworded.
- **Golden set v2** was created for the PII masker fix and the line-ending fix; v1 stays
  immutable.

## 1. Finding

- **Licence.** The Customer Support on Twitter corpus is CC BY-NC-SA 4.0, and its owner asks to
  be contacted about commercial or full-dataset use. *This licence statement was supplied by
  the project owner and has not been independently re-verified in this audit.*
- **Tree.** The repository tracks verbatim tweet text (customer messages and brand replies) in
  18 files at HEAD (`reports/privacy_scan.json`, `docs/RELEASE_AUDIT.md` §10).
- **History.** 9 further blobs hold text that exists only in history.
- **Visibility.** The repository is private with 0 forks, but the exposure travels with every
  clone.

Separately, the release audit found a PII-masker gap. `mask_pii` leaves 11-digit North American
numbers written with a leading country code (`N-NNN-NNN-NNNN`) unmasked. One gold message
contains one: a toll-free business number, quoted by the customer. Fixing the masker would
change stored gold text and therefore the lock hash, so it is out of scope here and recorded as
a finding.

## 2. Inventory: what each text-bearing file is needed for

| File | Tweet text fields | Golden reconstruction | Evaluation | Report | Tests | Remediation |
|---|---|---|---|---|---|---|
| `data/golden/candidates.jsonl` | `customer_message`, `context[].text` | **Yes**: the gold set | **Yes**: `load_gold` | counts only | 6 test modules (skip without it) | **IDs + hashes** (§3.1) |
| `reports/golden_eval/predictions.jsonl` | `message` (all rows); `reply` for baseline_b (128 verbatim historical brand replies) and agent_template (120 replies built from evidence) | — | artifact verification needs **no text**; judge re-scoring needs replies | failure examples | — | Drop `message`; hash the dataset-derived replies; keep agent_llm replies (§3.2) |
| `src/hiver_support/taxonomy.py` | 29 codebook example texts | — | **The frozen taxonomy hash covers them**, and every gold annotation binds that hash | — | `test_taxonomy.py` | **Edit script + hashes** (§3.3) |
| `reports/taxonomy_{discovery,probes,adjudication,candidate}.json` | cluster and probe exemplars | — | — | provenance of the taxonomy decision | docstring references only | Replace exemplar text with pair ids (§3.4) |
| `reports/classifier_dev_errors.json` | dev error examples | — | — | — | — | Replace with pair ids |
| `reports/llm_smoke_*.json` (3), `reports/agent_demo_run.json` | smoke-run messages and replies | — | — | Milestone 6–7 history | — | Replace with pair ids; keep counts and rates |
| `docs/ANNOTATION_GUIDE.md`, `INTENT_TAXONOMY.md`, `TAXONOMY_ADJUDICATION.md`, `AGENT.md`, `CLASSIFIER.md`, `reports/golden_eval/failure_analysis.md` | short quoted excerpts (10 / 4 / 3 / 1 / 1 / 1+ matched) | — | — | **Yes**: failure examples | — | Replace quotes with pair id + neutral paraphrase |
| `reports/llm_calls.jsonl` | none at HEAD (redacted in `409622d`) | — | cost accounting | — | — | History only (§6) |
| `reports/golden_eval/judge.jsonl` | generated rationales; 0 shingle matches | — | judge metrics | — | — | Keep. Short-quote blind spot: UNKNOWN |
| `tests/*` | generic brand phrases ("DM us") in 5 files; 0 customer text | — | — | — | yes | Keep: generic phrasing, not identifiable content |

**Scan blind spot.** 8-word shingles cannot see messages under 8 words, such as the
2-token version-number message of pair `143558__143556` quoted in `failure_analysis.md`. Remediation must therefore be
**structural**: every field or quote that holds tweet text is replaced, whatever the scan says.
A test enforces it (§5).

## 3. Architecture: ID-based local reconstruction

```
Kaggle download (data/raw/twcs, gitignored, scripts/fetch_data.py)
   ↓  existing deterministic reconstruction (load_brand_pairs → temporal_split)
local pair index  (data/interim/*.pkl, gitignored)
   ↓  scripts/materialize_text.py   [new]
     for every committed id record: rebuild text with the SAME code that wrote it,
     then check sha256(text) == committed hash, or raise
data/local/  (gitignored)
   candidates.jsonl            byte-identical to today's file
   predictions.jsonl           byte-identical to today's file
   taxonomy_examples.json      the 29 codebook texts
   ↓
existing loaders, GOLDEN_LOCK verification and taxonomy hash, unchanged in meaning
```

### 3.1 Golden candidates — ROUND TRIP MEASURED (200/200)

**Committed form:** `data/golden/candidates.ids.jsonl`. Each record holds:
- the identifiers: `pair_id`, `conversation_id`, `customer_tweet_id`, `created_at`;
- `context: [{tweet_id, author_role, created_at}]`;
- `customer_message_sha256`, `context_text_sha256[]`, `label_status`.

**Rebuild:** `golden.sampling._to_candidate(pair)` for each `pair_id` in the test pool.
**Measured: 200 of 200 records rebuild byte-identical to today's committed records.**

**Lock:** `GOLDEN_LOCK.json` `content_sha256` covers `customer_message`, so `verify_lock` on the
materialised file proves exact reconstruction. The lock itself needs no change.

**Line-ending defect, to fix in the same change.** The manifest's `candidates_sha256`
(`68e5db99…`) is the hash of the Windows CRLF working copy; the git blob, and a rebuild, hash to
`d033cd90…`. On Linux or macOS the manifest check would fail today. Fix: add a `.gitattributes`
rule (`*.jsonl text eol=lf`) and hash LF-normalised bytes.

### 3.2 Evaluation predictions

- **Committed form:** `message` removed. `reply` is kept for agent_llm, because it is model
  output and failure analysis reads it. For agent_template and baseline_b, the reply is replaced
  by `reply_sha256` plus the evidence case ids it was built from.
- **Rebuild:** `message` comes from the materialised candidates. baseline_b's reply is the
  nearest case's historical reply, looked up by case id. agent_template's reply is re-derived by
  the existing template generator from its evidence ids. Each is checked against `reply_sha256`.
- **Immutability of the original evaluation.** Before any redaction, record the sha256 of every
  original evaluation artifact (`reports/golden_eval/ORIGINAL_ARTIFACT_HASHES.json`).
  Materialisation must reproduce the original `predictions.jsonl` byte-for-byte and match that
  hash. The headline metrics are untouched: `metrics.json`, `summary.md` and `risk_coverage.*`
  contain no tweet text.
- **Artifact verification (target A) stays text-free:** `artifact_integrity.py`,
  `risk_coverage.py` and `failure_analysis.py` read no customer or brand text (verified by
  source). `failure_analysis.py` needs only whether context exists.
- **Feasibility of the reply rebuild: NOT YET MEASURED.** It must be shown 248/248
  byte-identical before redaction.

### 3.3 Taxonomy codebook examples — EDIT SHAPE MEASURED, ROUND TRIP NOT YET RUN, hash-critical

`Taxonomy.content_hash()` serialises every intent including `examples[].text`. The v0.3.0 hash
`613f5dfe…` is bound by all 200 annotations and the lock. Deleting the text naively changes the
hash and invalidates the gold set.

**Measured** against the source tweets:
- 3 of 29 examples equal the whitespace-collapsed tweet;
- 21 differ only by deletions (handles, emoji, trailing fragments);
- 5 include replacement edits (sizes not yet reviewed; one inspected example replaces 1 character).

**Committed form:** `{pair_id, note, edits, text_sha256}`, where `edits` is a list of
`(offset, deleted_length, inserted_chars)` applied to the collapsed source text. Inserted
characters must be reviewed so they are not identifying.

**Hash preservation.**
- `TAXONOMY.frozen_hash` becomes a recorded constant (`613f5dfe…`).
- With the dataset present, `content_hash()` is computed from the materialised texts, and a test
  asserts it equals the constant.
- Without the dataset, the recorded constant is used and the equality test **skips, never
  passes**. That follows the existing real-data convention.

**Must be proven before merge:** the materialised texts reproduce `613f5dfe…` exactly.

### 3.4 Exploratory reports and docs

- Exemplar texts are replaced by pair ids. Aggregate numbers (cluster sizes, probe rates,
  error counts) stay.
- Quoted examples in docs become `pair_id` plus a neutral description.
- These files are not inputs to any metric, test assertion or hash: verified by the dependency
  map above.

## 4. What must not change

- **Headline metrics:** `metrics.json`, `summary.md`, risk-coverage and failure-analysis counts.
- **Gold labels and hashes:** `annotations.jsonl`, `GOLDEN_LOCK.json` `content_sha256`,
  taxonomy hash `613f5dfe…`.
- **The 800 predictions** as materialised (byte-identical to the original file).

## 5. Verification (each check has a negative control)

| Check | Pass condition | Negative control |
|---|---|---|
| Materialisation | every record's text sha256 matches; the whole materialised file matches the original artifact hash | one flipped character in a source tweet → raises |
| Gold lock | `verify_lock` passes on materialised candidates | one edited message → raises |
| Taxonomy hash | computed == `613f5dfe…` | one wrong edit op → mismatch |
| Metrics | `artifact_integrity.py` 9/9 without the dataset | existing corruption controls |
| Boundary | `evaluation_boundary.py` 0 differences against materialised originals | existing flip control |
| No tweet text at HEAD | `privacy_scan.py`: 0 customer and 0 brand-reply matches; plus a **structural test** that no tracked JSON/JSONL holds a `customer_message`, `message`, `text` or dataset-reply field with a value | a planted field → test fails |
| Suite | full suite passes; real-data tests skip without the corpus | — |

**Cost to target A: none.** It stays text-free.

**Cost to golden-set access:** annotation, evaluation and the codebook examples now require the
Kaggle download (558 s measured) plus materialisation.

## 6. Git history remediation

**Exposure** (measured; only `main` exists: no tags, stashes, other branches, pull requests or
forks):

| Path | Commits that stored tweet text |
|---|---|
| `src/hiver_support/taxonomy.py`, `docs/ANNOTATION_GUIDE.md`, `docs/INTENT_TAXONOMY.md` | `b337f0b`, `cd75638` |
| `reports/taxonomy_{discovery,probes,candidate}.json` | `b337f0b` |
| `reports/taxonomy_adjudication.json`, `docs/TAXONOMY_ADJUDICATION.md` | `1645ba5` |
| `reports/classifier_dev_errors.json`, `docs/CLASSIFIER.md` | `4be4e59` |
| `reports/agent_demo_run.json`, `docs/AGENT.md` | `d256ef0`, `6a1fcfc`, `ae713c6`, `07f7ca8` |
| `data/golden/candidates.jsonl`, `reports/llm_smoke_test.json` | `6a1fcfc` |
| `reports/llm_smoke_gpt-oss-{120b,20b}.json` | `18855f9` |
| `reports/llm_calls.jsonl` (debug prompt row) | `18855f9` … `6772707` (redacted at `409622d`) |
| `reports/golden_eval/predictions.jsonl` | `6772707`, `b1429da` |
| `reports/golden_eval/failure_analysis.md` | `8586a29` |

Everything from `b337f0b` (Milestone 3) onward is reachable from `main`. Remediating HEAD alone
is insufficient: every clone carries the blobs.

**Options.**
- **H1 (recommended): rewrite blobs in place with `git filter-repo`.**
  - Keeps all 35 commits, messages, authorship and order, and produces a `commit-map`.
  - Each affected historical blob is replaced by its redacted equivalent, produced by the same
    redaction code used at HEAD. Blobs of unaffected paths are untouched.
  - **Consequence:** every commit hash changes. Docs cite hashes (`6772707`, `fbbe88f`,
    `81bafe4`, …), so a follow-up commit rewrites them through the commit-map, and the release
    audit records both old and new ids.
- **H2: remove affected paths from all of history** (`--invert-paths`). Simpler, but it deletes
  the history of `taxonomy.py`, docs and the evaluation artifacts. **Rejected:** it destroys the
  audit trail.
- **H3: single-commit orphan history.** Loses all provenance. **Rejected.**

**H1 procedure (after approval).**
1. Remediate HEAD (§3) in one commit; all §5 checks pass.
2. Create an offline mirror backup (`git clone --mirror`), kept outside the repository and never
   pushed. The owner decides when it is deleted, since it still contains the text.
3. In a fresh mirror, run `git filter-repo --blob-callback` using a prepared map
   `{old_blob_sha → redacted bytes}` for all affected blob versions.
4. **Verify the rewritten repository before pushing:**
   - `privacy_scan.py` over all reachable blobs: 0 matches;
   - the structural test on every commit's tree for the affected paths;
   - the full suite at the new HEAD;
   - `GOLDEN_LOCK` and taxonomy hash after materialisation;
   - the commit count is unchanged (35).
5. Update hash references in docs via the commit-map, then commit.
6. `git push --force-with-lease origin main` (**requires explicit owner approval at that moment**).
7. **Local cleanup:**
   - `git reflog expire --expire=now --all && git gc --prune=now --aggressive` in the working
     repository;
   - delete throwaway clones that hold old history and raw data (`fresh_clone`, `lite_clone`,
     `py312` in the session scratchpad);
   - remove the offline mirror once the owner is satisfied.
8. **GitHub:** the repository is private with no forks or pull requests. Old objects stay
   reachable by sha until GitHub garbage-collects them. If that matters, ask GitHub Support to
   purge cached views and unreferenced objects. Any other existing clone keeps the old history
   and must be deleted and re-cloned.

## 7. Estimated blast radius

- **Code:** loaders for candidates, predictions and taxonomy examples, a new materialiser,
  `.gitattributes`, the structural test.
- **Scripts:** annotation (reads candidates), evaluation, smoke and demo scripts.
- **Docs:** quotes and commit hashes.
- **Evaluation behaviour: none**, if every §5 check passes.

## 8. Open questions for the reviewer

1. Are agent_llm replies (model output that may paraphrase brand guidance, with 0 verbatim
   matches required) acceptable to keep in `predictions.jsonl`?
2. Are the 29 edit scripts acceptable (≤ 1 inserted character per replacement), or should the
   codebook examples move to a documented "v0.3.0 frozen constant, dataset-verified" scheme with
   no edit data at all?
3. H1 (rewrite blobs, keep history) versus accepting HEAD-only remediation for a private
   repository.
4. Retention period for the offline pre-rewrite mirror.
