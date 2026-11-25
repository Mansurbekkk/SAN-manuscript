# Script-Aware Normalization (SAN) for Prompt Injection Detection

Code and results for the manuscript *"When Cyrillic Is Not an Attack: Script-Aware Normalization for Prompt Injection Detection in Legitimately Multi-Script Languages"* (M. Axmadaliyev, TUIT).

SAN is a detector-agnostic preprocessing step for prompt injection classifiers. It:

1. **decodes** hidden channels (Unicode tag characters, variation-selector runs, bidi overrides) instead of deleting them;
2. applies **NFKC** compatibility normalization;
3. **folds homoglyphs per word**, only for letters foreign to the word's own script. Uzbek-specific Cyrillic letters (ў, қ, ғ, ҳ) block folding;
4. normalizes Uzbek **apostrophes** (o‘, g‘, tutuq ʼ).

**SAN-T** additionally transliterates legitimate Uzbek Cyrillic into the official Latin orthography. Use it only for detectors whose cross-script parity (CSP) is below 0.9.

## Repository layout

| Path | Contents |
| --- | --- |
| `san/core.py` | Uzbek Latin→Cyrillic transliteration, obfuscation transforms T1–T8, small hand-written test sets |
| `san/san.py` | SAN, SAN-T, obfuscation signal and baselines N0–N4 |
| `notebooks/san_stage4.ipynb` | Main evaluation: open corpora, 5 detectors, injection-attributable TPR, English calibration |
| `notebooks/san_stage5.ipynb` | Stage 4 plus three open-source sanitizers (LLM Guard InvisibleText, UTS #39 skeleton, Unidecode) |
| `tools/build_nb*.py` | Scripts that generate the notebooks from `san/` (run from `tools/`) |
| `results/stage5/` | Final result tables used in the manuscript |

## Quick use

```python
exec(open("san/core.py").read()); exec(open("san/san.py").read())

canon, obf_flag, stats = san("Ignоre previous​ instructions")  # Cyrillic о + zero-width
print(canon)        # "Ignore previous instructions"
print(san_t("Бугун Тошкентда ҳаво қандай бўлади?")[0])
# "Bugun Toshkentda havo qanday bo‘ladi?"
```

## Reproducing the experiments

1. Open `notebooks/san_stage5.ipynb` on Kaggle (File → Import Notebook).
2. Settings: Accelerator **GPU T4**, Internet **On**, and a Kaggle Secret `HF_TOKEN` with access to the gated Meta Prompt Guard models.
3. Run *Save Version → Save & Run All*. The run takes about 2 hours on a T4. Results are written to `/kaggle/working/san_stage5/`.

Benign data are streamed from public Hugging Face datasets at run time: `MLDataScientist/SlimOrca-Dedup-Uzbek-cleaned`, `tahrirchi/uz-crawl`, `Digitaluz/uzbek-language-corpus`, `databricks/databricks-dolly-15k`, `fancyzhx/ag_news`, `deepset/prompt-injections` and `leolee99/NotInject`. The texts themselves are not redistributed here; see each dataset's license.

## Key results (threshold 0.5)

| | ProtectAI v2, N0 | ProtectAI v2, SAN-T |
| --- | --- | --- |
| FPR, benign Uzbek prompts (Latin / Cyrillic) | 0.41 / 0.79 | 0.41 / 0.44 |
| FPR, natural Cyrillic documents | 0.70 | 0.38 |
| Cross-script parity (CSP) | 0.62 | 0.97 |

Worst-case injection-attributable TPR over eight obfuscation families (PIGuard, English attacks): 0.00 for N0, confusable folding, LLM Guard, UTS #39 skeleton and Unidecode; **0.95 for SAN**.

## Citation

Manuscript under review; citation will be added on publication.
