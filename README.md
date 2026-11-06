# LLM as Reviewer — SSDataBench (GSS 2018)

Research code for the project "LLM reviewer design v2": testing whether asking a model to
**state distributions** or **judge candidates** recovers more realistic synthetic data than
asking it to **generate** answers, on SSDataBench's GSS 2018 track, with inputs, outputs,
sample size and official scoring unchanged and **no human data** in generation.

- Research design: `docs/Research Design.md`
- Implementation spec: `docs/SPEC.md`
- Pre-registered prompt wordings: `docs/PROMPT-WORDINGS.md`

## Layout

```
reviewer_sim/     # main package (fields, llm client, prompts, parsing, conditions, scoring)
configs/          # one YAML per condition / phase
data/             # NOT tracked (gitignored); row lists, groups, stated tables
runs/             # NOT tracked; per-run outputs in the official sim_profiles_*.csv format
reports/          # phase reports (tracked)
patches/          # ssdatabench seed patch
tests/            # unit tests
```

`data/raw/ssdatabench` is a verbatim copy of the benchmark repo (read-only reference);
`data/raw/real_data`, `data/processed` come from previous verified projects.

## Setup

```bash
uv sync                 # create .venv and install deps
cp .env.example .env    # add OPENROUTER_API_KEY (never committed)
uv run pytest           # run tests
```

## Phases

See `docs/SPEC.md`. Each phase ends at a checkpoint report in `reports/`.

| Phase | Study | What |
| --- | --- | --- |
| 0 | Study 1 | Infrastructure, dev split, generator choice, S0–S7 on 2 questions |
| 1 | — | References B1/B2, Study 3 models, preliminary B4 |
| 2 | Study 2 pilot + Study 3 | All methods on 500 dev rows |
| 3 | Study 2 full | Choose method on dev evaluation part |
| 4 | Study 4 | Final test, second model, wordings |
| 5 | — | Second dataset |
