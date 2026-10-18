# Knowing Which, Not How Much

Code and data for the paper **"Knowing Which, Not How Much: Language Models Order Party Positions They Cannot Place"** by Yousef Attia and Jarosław A. Chudziak (Warsaw University of Technology), accepted at **PACLIC 40** (Taipei, December 2026).

Language models are asked to place a party's emphasis on an issue on a four-level scale (absolute framing) and to name the highest- and lowest-emphasis party among four (comparative framing). Answers are scored against the coded salience values of the Manifesto Project (CMP/MARPOR) for eight parties in Japan and South Korea across ten policy categories.

## Repository layout

| Path | Contents |
|---|---|
| `data/benchmark.json` | The benchmark: parties, issues, coded values, gold buckets, tie-aware gold rankings and prompts |
| `outputs/` | Raw model responses (`*.jsonl`), one file per model, for the free-text run, the tagged run (`structured/`) and their grounded counterparts |
| `scored/` | Scored responses, one file per task and run (written by the `score_*.py` scripts) |
| `run_eval*.py` | Query the models |
| `score_*.py` | Score the responses |
| `paper_stats*.py`, `validate_reffree.py` | Statistics reported in the paper |

The paper's main results use the tagged run, in which each prompt asks for a final machine-readable line (`EMPHASIS:` or `MOST:`/`LEAST:`). The free-text run, which uses the same prompts without that line, is kept because the paper compares the two read-out procedures.

## Reproducing the results

Requires Python 3.10+ and `pip install -r requirements.txt`.

The scored files are included, so the statistics can be recomputed directly:

```bash
python paper_stats_structured.py   # tagged run (main results)
python paper_stats.py              # free-text run (read-out comparison)
```

To rerun the full pipeline, put API keys in a `.env` file (see `run_eval.py` for the variable names), then:

```bash
# query (one model key from run_eval.py, or --all for the tagged run)
python run_eval.py <model>
python run_eval_grounded.py <model>
python run_eval_structured.py --all
python run_eval_structured.py --all --grounded

# score
python score_singleparty.py
python score_multiparty.py
python score_grounded.py
python score_structured.py
python score_structured.py --grounded
python score_selfconsistency.py
python score_selfconsistency.py _grounded
```

Model outputs are not deterministic across API versions, so a rerun will not reproduce the released responses exactly.

## Data source

Gold labels are derived from the Manifesto Project Dataset (version 2025a). The raw dataset is not redistributed here; it is available from [manifesto-project.wzb.eu](https://manifesto-project.wzb.eu/). Please cite it as requested by the Manifesto Project.

## Citation

```bibtex
@inproceedings{attia2026knowing,
  title     = {Knowing Which, Not How Much: Language Models Order Party Positions They Cannot Place},
  author    = {Attia, Yousef and Chudziak, Jaros{\l}aw A.},
  booktitle = {Proceedings of the 40th Pacific Asia Conference on Language, Information and Computation (PACLIC 40)},
  year      = {2026}
}
```
