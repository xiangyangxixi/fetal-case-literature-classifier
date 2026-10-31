# casefilter

Screens a large literature set (10k+ papers) for UltraPhenoCards intake: keep a paper if it yields **at least one** fetal case that has a prenatal ultrasound phenotype abnormality and one of the 11 genetic tests (Karyotyping, CMA, FISH, CNV-seq, WES, Targeted Panel, MLPA, Sanger, WGS, PCR, mtDNA-seq).

## How it decides

Stage 1 (rules, offline, free, explainable) applies these gates to title + abstract (+ full text if provided):

| Gate | Meaning |
|---|---|
| hard exclude | review / meta-analysis / guideline / editorial publication type or title; animal or cell study with no human context; no PMID |
| FETAL | fetus / prenatal / pregnancy context |
| PHENO | ultrasound modality or a known sonographic finding |
| METHOD | one of the 11 tests (PCR and Sanger alone are weak evidence) |
| CASE | case report wording, individual fetus description, or a cohort with counts |
| LINK | one sentence mentions fetal context and a test together |

Labels: `INCLUDE` (all gates met), `REVIEW` (uncertain, e.g. abstract omits the test, or only PCR/Sanger), `EXCLUDE`. Recall is favoured: when unsure the paper goes to `REVIEW`, never to `EXCLUDE`.

Stage 2 (optional, `--llm`) sends only `REVIEW` papers to Claude with the intake criteria and caches answers in `llm.cache.jsonl` so runs resume. Low-confidence answers stay `REVIEW` for manual checking.

## Usage

```bash
pip install -e .            # add ".[llm]" for stage 2
casefilter run papers.csv --out out            # rules only
export ANTHROPIC_API_KEY=...                   # stage 2
casefilter run papers.csv --out out --llm
casefilter eval out/all.csv gold.csv           # recall / precision on a labelled subset
```

Input: `.csv/.tsv/.jsonl` with columns `pmid,title,abstract` (optional `pub_types`, `fulltext`), or a PubMed XML export. Output in `out/`: `all.csv`, `include.csv`, `review.csv`, `exclude.csv` with detected methods, reason codes and evidence sentence.

## Recommended workflow

1. Hand-label 200-300 papers (`pmid,label`), run `casefilter eval`, check false negatives first.
2. Tune patterns in `src/casefilter/lexicon.py` until recall is acceptable.
3. Run all papers with `--llm`; manually review what remains in `review.csv`.

Full text matters: abstracts often omit the tests used, which is why those papers land in `REVIEW`.
