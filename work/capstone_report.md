# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** musab855 (FlyRank ML Internship, Machine Learning track)
- **Lane:** Refresh / Content Opportunity Scoring (locked end of Week 4)
- **Repo:** https://github.com/musab855/flyrank-ml-internship
- **Date:** October 2026

---

## 1. Problem framing

**Decision.** A content/SEO team cannot review every page. The decision this work improves is
**which pages a reviewer opens first this week, in what order, and why** — the unit of analysis
is one content page (pseudonymous `content_id`) in a scored month.

**Output.** A ranked review queue: score → tier → action → reason codes → a per-row `wrong_if`
string. The output is a *page to review*, never a *change to ship*.

**Who acts.** An SEO specialist or content strategist owning a weekly review queue. They take
the top of the list, run the human-review checks (real traffic? decline or look-alike?
borderline? one client's problem?), then decide the content action themselves.

**Cost of a wrong call.** A false positive costs ~30 minutes of reviewer time (an assumption
written down, not a measurement). A false negative costs continued visibility loss on a page
that was already at stake. That asymmetry makes the system aggressive about *volume* and
conservative about *acting*.

**Why data/ML at all.** The transparent rule built in Week 4 is genuinely good at the top of
one list (P@50 0.880) but is silent below its thresholds by construction — 1,074 flagged pages
of 140,599. A fixed if-statement cannot weigh several moderate signals together across the rest
of the inventory. ML earns its place only there: ranking the pages the rule cannot see.

**Claim level.** Observed / measured / directional / decision-support. Nothing here is an
experiment, so no causal claim and no statement about how any search engine ranks pages.

## 2. Data safety

**Used.**

| Source | Role | Scale |
|---|---|---|
| `FlyRank/internship-warehouse` build v20260703 (Hugging Face, gated) — `fact_content_daily_performance` | the modeling frame | 78,835,655 rows total; March 2026 slice (`2026-03-01`..`2026-03-31`) aggregated to one row per page |
| same release — `dim_content` | grain/count contract checked in ML-04 before any feature was built | 519,606 content items |
| `data/raw/content_refresh_anonymized.csv` (committed starter slice) | age/freshness context in §6 only — different dataset, different label, never a queue input | 30,000 pages, 32 clients |

**Frame and exclusions (all printed by `work/notebooks/capstone.ipynb`).** 176,738 rows in the
March slice → drop 24,757 (position NaN: no GSC data in the feature window), 1,306
(`position == 0`, the "no data" placeholder), 10,076 (undefined label: one half of the month
unmeasurable) → **140,599 pages, 43 clients, base rate 0.4649**. Undefined labels are dropped,
not guessed. Every aggregation is filtered on `gsc_data_available = TRUE`, so zero-filled days
never read as "no demand".

**Deliberately excluded.** `fact_content_query_90d` (query-level rows — raw queries are never
public, so they never enter this work); `trend_pct` / `trend_direction` / `is_declining_label`
(derived from the outcome — leakage); `content_id` / `client_hash_id` as features (grouping and
folds only); every product decision flag or score (not present in the release, and never
rebuilt); the label's own window column `gsc_impressions_16to31` (kept in the cache only so
ML-09 could demonstrate that the harness detects it, dropped before modeling).

**Leakage risks considered.** Label-derived columns (asserted banned in code), future/overlapping
windows (features days 1–15, prediction moment 2026-03-16, label days 16–31 — the timeline is
printed, not assumed), decision-derived inputs (the rule's thresholds produce baseline *rows*
only, never model *inputs*), and grouped validation (no client on both sides of a fold).
ML-09's deliberate confession is part of the evidence: adding the label's window moved the
forest from ROC-AUC 0.654 → 0.997 (P@50 → 1.000), after which the column was removed.

**Public safety.** No client names, domains, URLs, titles, or queries appear anywhere in
`work/`. Outputs are aggregates, pseudonymous IDs, and charts. CI blocks any committed dataset
other than the anonymized starter slice, and the queue CSV is gitignored by design.

## 3. Baseline

**`HIGH_VOLUME_POOR_POSITION`** — flag pages with >= 500 impressions/day **and** average
position >= 11 (both measured on days 1–15), rank flagged pages by impressions, everyone else
scores 0. It is the right comparison because it is transparent (two thresholds a reviewer can
reproduce by hand), it was fixed in Week 4 before any model existed, and it is what a team
would actually run without ML.

**Two extra references keep the comparison fair:** `rule_full` (the same flag given the whole
frame to rank — the rule as a ranker) and `impressions_only` (visibility with no rule at all).

**Numbers, same frame, same run** (client-grouped OOF, n = 140,599, base rate 0.4649):

| method | P@10 | P@20 | P@50 | ROC-AUC | avg precision |
|---|---:|---:|---:|---:|---:|
| `rule` (baseline) | 1.000 | 0.950 | **0.880** | 0.505 | 0.469 |
| `rule_full` | 1.000 | 0.950 | 0.880 | 0.609 | 0.537 |
| `impressions_only` | 0.700 | 0.700 | 0.620 | 0.609 | 0.534 |

The rule's P@50 of 0.880 is **1.89x** the base rate. Its unaided whole-list ROC-AUC (0.505)
sits on the 0.500 no-skill line: a filter, not a ranker.

**Signal-audit verdicts inherited from Week 4** (`w04_baseline_score.ipynb`): impressions
cutoff **CONFIRMED** (decline rate rises across every impressions bucket); position cutoff
**MIXED** (47.2% → 48.0% → 45.9% → 42.5%, non-monotonic). Position >= 11 is therefore kept as
*policy*, not evidence, and this capstone inherits it unchanged.

## 4. Model / analysis

**Method.** A scoring/ranking task, not a classifier decision: every page gets a score; a human
reads a ranked list.

- **Features (3, all days 1–15):** `gsc_impressions_avg`, `gsc_clicks_avg`, `position_avg`.
  Deliberately small — one month, 43 clients; a wider list would fit noise. Derived inputs:
  `ctr` (used only for reason codes, never a feature — it is same-window kin of the label).
- **Label (observed, later window):** `is_declining_label = 1` when average daily impressions
  on days 16–31 < days 1–15. An outcome that happens after the features, not a rule I wrote.
  Base rate **0.4649**.
- **Models:** logistic regression (log1p on the two count features, standardized,
  `class_weight=balanced`) as the interpretable baseline, and a random forest
  (200 trees, `max_depth=10`, `min_samples_leaf=25`,
  `class_weight=balanced_subsample`) as the primary model. Both `random_state=42`.
- **Validation design:** client-grouped 5-fold out-of-fold scoring — every client in exactly
  one test fold, every page scored once by a model that never saw its client, metrics pooled
  over the single resulting queue. Base rate printed under every table.
- **Baseline comparison:** `rule`, `rule_full`, `impressions_only` enter the same table with
  no training step, so their numbers must be identical under any split (a harness check).

**Leakage checks (asserted in code, not asserted in prose):** banned-column set ∩ feature list
= ∅; the feature/label timeline is displayed; rule-derived columns never enter `BASE_FEATURES`;
the audit column is deleted before modeling; and ML-09's confession demonstrated that the
harness *would* scream if a leak arrived.

**Reproduction receipts.** This capstone re-derives all **25 published numbers** from ML-08/ML-09
within 0.005 (worst gap this run: 0.0000) and additionally matches every receipt in
`work/outputs/playbook_metrics.json` (frame facts hard-asserted: 140,599 pages / 1,074 flagged /
base rate 0.4649). If any of those asserts fail, the run stops and the paper must not ship.

## 5. Evaluation

**Same split, same pages, same run** (pooled client-grouped OOF, n = 140,599, base rate 0.4649):

| method | P@10 | P@20 | P@50 | ROC-AUC | avg precision |
|---|---:|---:|---:|---:|---:|
| `rule` (baseline) | 1.000 | 0.950 | **0.880** | 0.505 | 0.469 |
| `rule_full` | 1.000 | 0.950 | 0.880 | 0.609 | 0.537 |
| `impressions_only` | 0.700 | 0.700 | 0.620 | 0.609 | 0.534 |
| logistic regression | 0.700 | 0.700 | 0.780 | 0.630 | 0.571 |
| **random forest** | 0.500 | 0.550 | 0.660 | **0.654** | **0.583** |
| *base rate* | *0.465* | *0.465* | *0.465* | *0.500* | *0.465* |

**Did the model beat the baseline? Scoped, two answers:**

- **Top of the queue: no — the rule leads.** P@50 0.880 (rule) vs 0.660 (forest). A reviewer
  who reads only the first 50 rows is better served by the rule.
- **Whole list: yes, modestly — against the fair reference.** Forest ROC-AUC 0.654 vs
  `rule_full` 0.609 and average precision 0.583 vs 0.537: a measured **+0.045**, not the
  flattering +0.149 a tie-flawed rule AUC would suggest. The forest keeps ranking the 139,525
  pages the rule is silent on — that is exactly where Tier B comes from.

Neither result is a general "model beats rule" claim; both are scoped to this frame.

**Validation contrast (ML-09, same everything except the split):** row-random → client-grouped
moves the forest P@50 0.800 → 0.660, P@10 0.700 → 0.500, ROC-AUC 0.672 → 0.654; the share of
test rows whose client was also in training goes 100% → 0; the top-50 queue itself changes
(5 of 50 pages shared between splits). The grouped number is the one above, and the one that
answers the deployment question.

**Error analysis (short version).** The forest's pooled top-50 contains **17 pages that did not
decline** (base rate 0.465 → a random 50 would carry ~23). The false positives share one shape:
positions 35.9–40.8, 860–2,065 impressions/day, but only 0.5–1.7 clicks/day (CTR 0.0003–0.0008).
They are hard because the label can only ask "did impressions fall?" — at position ~40 the
impression count itself swings day to day, and a page can hold steady and still be worthless.
Kin-of-target check: without `gsc_impressions_avg`, logistic falls to ROC-AUC 0.494 (at/below
no-skill) and the forest to 0.591 — impressions carries most of the signal, is measured strictly
before the label window, and is disclosed as close kin. Forest impurity importance (mean over
grouped folds): impressions 0.749, position 0.159, clicks 0.092 — nothing towers toward 1.0,
and the honest AUC is 0.654 rather than 0.99.

**Interpretation.** The rule owns the very top (real, measured); the forest owns the whole
list (real, modest); logistic regression sits in between and buys interpretability cheaply.
The signal audit's MIXED verdict on the position cutoff is a negative result reported as one.

## 6. Interpretation

- **The rule owns the very top of one queue (measured), the forest owns the whole list
  (measured, modest), and the honest gap between them is +0.045 ROC-AUC** — a reordering of a
  review queue, not a claim about any search engine.
- **What the forest leans on:** impressions (impurity importance 0.749, position 0.159, clicks
  0.092); drop impressions and logistic falls to ROC-AUC 0.494 (no-skill) and the forest to
  0.591. Legal (measured days 1–15), close kin of the label, disclosed — which is why the
  leakage audit and the confession demo exist.
- **The decay question, answered honestly:** on the starter CSV (different dataset + label,
  context only), decline rate *falls* with age — 62.7% (<180 days, n=12,272) → 51.5%
  (180–365, n=11,368) → 42.6% (365+, n=6,360) against a 0.542 base rate — and freshness does
  not read at all (non-monotonic; 3 of 5 buckets under n=500). So "old content decays" does not
  survive contact with this slice, and the queue therefore ranks on **observed visibility at
  stake**, with age/freshness entering only as a review prompt.
- **A negative result reported as one:** the research paper's length gap does not reproduce in
  the starter slice (median 2,909 vs 2,848 words ≈ 2%, not 37.6%) — different dataset, label,
  and sample, so not a refutation, but a finding that the gap is not universal.
- **Where the naive split would have misled:** ML-09's row-random run would have shipped a
  *different queue* (5/50 overlap) assembled with client overlap — measured, not assumed.
- **Surprises worth naming:** the forest's P@50 *rises* to 0.740 without impressions while
  every whole-list metric falls — a single-K queue statistic can move against the whole-list
  evidence, which is exactly why P@10/20/50 are always reported next to ROC-AUC and average
  precision rather than whichever looks best.

## 7. Recommendation

**The tiered queue — 1,168 actionable pages (Tier A 1,074 + Tier B 94), 78.0% labelled
declining vs the 0.465 frame base rate (1.68x):**

- **Tier A:** every flagged page, impressions order (the measured P@50 0.880 order).
  Actions: `verify_traffic_first` (572 — CTR < 0.001, verify the traffic before any edit) or
  `review_first_refresh_and_ctr` (502).
- **Tier B:** the forest's tail on visible pages the rule does not flag (OOF score ≥ 0.80,
  impressions ≥ 100) — 94 pages, all `mid_volume_opportunity`, action `secondary_review`.
  Confidence is never `high` for Tier B: a rank from one month is not a high-confidence claim.
- **Confidence mix:** low 687 / medium 461 / high 20. `low` = borderline or near-zero CTR — the
  two ways this data misleads a reviewer.

**Archetypes → standing actions** (rule-based tags, not clusters; every non-queued page still
gets a standing action):

| archetype | n | decline rate | standing action |
|---|---:|---:|---|
| `high_volume_deep_position` (impr ≥ 500, pos ≥ 11) | 1,074 | 0.780 | `review_first_refresh_and_ctr` (or `verify_traffic_first`) |
| `page_one_high_stakes` (impr ≥ 500, pos < 11) | 2,031 | **0.453** | `protect_and_monitor` — decay watch, not a refresh trigger |
| `mid_volume_opportunity` (impr ≥ 100) | 17,658 | 0.544 | `monitor` (or `secondary_review` if Tier B) |
| `low_volume_long_tail` (impr < 100) | 119,836 | 0.451 | `monitor_only` |

The decline rates do the arguing: page-one pages decline *below* the base rate — queuing them
for refresh would spend reviewer hours where this frame says visibility is not at stake.

**Reason codes on every row** (feature-window observables only): `HIGH_VOLUME_POOR_POSITION`
1,074 (all of Tier A), `NEAR_ZERO_CTR_IMPRESSIONS` 572, `BORDERLINE_THRESHOLD` 170,
`MODEL_HIGH_DECLINE_RANK` 621 (corroboration, not proof), plus `PAGE_ONE_HIGH_STAKES` and
`LOW_VOLUME_WATCH` covering the rest of the frame.

**Capacity (30 min/page assumption, stated as an assumption):** top-10 = 5 review-hours with
10/10 declining; top-50 = **25 hours** (one focused week) with 44/50 declining and 158,626
impressions/day at stake; the full 1,168-page queue = 584 hours ≈ clearing at 20 pages/week
takes ~14.6 months — a **standing backlog**, not a one-shot project. Tier A holds **12.2% of the
frame's visibility on 0.76% of its pages** (1,082,306 of 8,895,076 impressions/day).

**How a reviewer uses it tomorrow:** take the top of Tier A; run the four human checks first
(real traffic? decline or look-alike — consolidation/seasonality/SERP-CTR/noise? borderline?
one client's problem?); only then choose the content action; write down what would make the
call wrong (`wrong_if` is already on the row).

**Explicit no-go list:** never publish/edit/delete/merge from a score; never act on
`low_volume_long_tail`; `verify_traffic_first` pages gate editing until traffic is verified;
never retrain unattended; never use the queue as a team/client performance metric.

**Monitoring triggers (from `playbook_metrics.json`, re-read by this capstone):** 3 already
**FIRING by design** — near-zero-CTR share of the top 50 (66%, 33/50) keeps the verification
gate on; single-client share of the top 50 (82%) warns "read within-client"; frame freshness
(a fixed March 2026 audit window read in a later month) — plus 2 awaiting a second month
(queue overlap, feature drift) and the MIXED position-cutoff verdict on monitor.

**Confidence and limits.** Decision-support: *these pages look worth reviewing first.* The
full limit list (seven items) is section 5 of `work/notebooks/capstone.ipynb` and section 6 of
the deployed paper: one month/one portfolio; the label is a proxy (regression to the mean not
measured); position cutoff is policy; half of Tier A may not be human traffic; nothing is
causal or about any search engine's ranking systems; the queue is client-concentrated
(largest client 74.7% of flagged pages); scores are ranks, not probabilities.

## 8. Reproducibility

**Run everything from a fresh clone:**

```bash
git clone https://github.com/musab855/flyrank-ml-internship.git
cd flyrank-ml-internship
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace work/notebooks/capstone.ipynb
```

The capstone is cache-first: if `work/outputs/feature_frame_march_audit.csv` exists it is used
(gitignored — regenerate on demand); otherwise the notebook re-scans the warehouse via DuckDB
over `hf://datasets/FlyRank/internship-warehouse`, needing a Hugging Face READ token from
Colab Secrets (`HF_TOKEN2`/`HF_TOKEN`), the environment, or `.env` (never pasted in a cell,
never printed). To reproduce the full chain: `w01` → `w02` → `w03_data_contract` →
`w04_baseline_score` → `w05_model` → `w06_validation_audit` → `w07_action_playbook` →
`capstone`, each executed top-to-bottom.

- **Seeds:** `random_state = 42` for fold construction, logistic regression, and the forest.
  Canonical row order (`content_id` stable sort) makes folds, tied queues, and the forest's
  bootstrap reproducible across environments.
- **Environment (this run):** Python 3.10.11, numpy 1.26.4, pandas 2.1.1,
  scikit-learn 1.3.1, matplotlib 3.8.0, duckdb 1.5.6 — same behavior checked against the
  published ML-08/ML-09 numbers (25/25 within 0.005, worst gap 0.0000).
- **Machine-checkable outputs:** `work/outputs/capstone_metrics.json` (frame, receipts,
  reproduction receipt, queue mix, capacity, limitations, figures) and
  `work/outputs/playbook_metrics.json` (ML-10 receipts). The queue CSV is gitignored by
  design — CI fails any committed dataset — and is regenerated by `w07_action_playbook.ipynb`.
- **Figures** (committed, regenerated by the capstone): `work/figures/queue_yield_curve.svg`,
  `queue_action_mix.svg`, `capacity_vs_yield.svg`, `decay_refresh_insight.svg`.

---

> **Claims checklist before submitting:**
> - [x] Language: observed / measured / directional / decision-support everywhere; the
>       notebook's own banned-phrasing scan returns 0 hits on its markdown.
> - [x] Metrics vs. base rate: 0.4649 printed under every results table; P@50 0.880 read as
>       1.89x base; ROC-AUC read against the 0.500 no-skill line.
> - [x] No causal claims without a design; nothing claims to predict or explain any search
>       engine's ranking systems.
> - [x] No client-identifying details — pseudonymous IDs, aggregates, charts only.
> - [x] Numbers in this report match a fresh re-run of `work/notebooks/capstone.ipynb`
>       (executed top-to-bottom, 0 errors, 25/25 receipts MATCH).
