# Governors 2026, by synthetic voters

A forecast of the 36 governor races on 3 November 2026, built from **synthetic survey responses**: large language models answering an election survey while playing real respondents to the Cooperative Election Study (CES). The synthetic answers are combined with past results and published polls, and the election is simulated 10,000 times.

- **Website:** https://rallan-vijval.github.io/sta2453-us-midterm-forecast/
- **Course:** STA2453H, University of Toronto, Fall 2026, Assignment II
- **Team:** Vijval Rallan (CES, personas, Method A), Shrey Sati (model queries, tests, Method B, GitHub Actions), Allwin Steve Paul Justin Anantharaj (race data, simulation, Method C, forecast, website)

## Data Sources and Licences

### Cooperative Election Study (CES)

This project uses data from the 2024 and 2025 Cooperative Election Study (CES) Common Content datasets, obtained from Harvard Dataverse.

- **2024 CES:** [Dataset DOI: 10.7910/DVN/X11EP6](https://doi.org/10.7910/DVN/X11EP6). The dataset is licensed under [Creative Commons CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
- **2025 CES:** [Dataset DOI: 10.7910/DVN/DFUGJR](https://doi.org/10.7910/DVN/DFUGJR). The dataset page identifies the licence as [Creative Commons CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).

Small samples of the CES data are included in this repository for reproducibility and development. Please consult the original datasets and their documentation for full details.

The samples are intended for development and testing. Results based on sampled data may not be representative of the full CES datasets or the electorate.

## Use of AI tools

### Models studied

The language models that answered the survey are the subject of the study, not tools used to build it. All three were called through OpenRouter between 25 October and 1 November 2026:

- `openai/gpt-6-luna`
- `deepseek/deepseek-v4.1-flash`
- `mistralai/mistral-small-24b-instruct-2501`

The exact identifiers and settings are in `config/models.yaml`, and every call's model, tokens and cost is logged in `data/02-analysis_data/run_log.csv`.

### Coding assistants

Each of us used AI assistants while building the project. We made the analysis decisions, ran every script ourselves, checked the outputs against the data, and reviewed every change before merging.

**Allwin Steve Paul Justin Anantharaj**
- Tools: Claude (Anthropic), model `claude-opus-5-5`, used through the Claude app's agentic coding workspace (built on Claude Code); GitHub Copilot, for one commit that replaced member labels with full names in script headers. [Add any other assistant you used, for example to reword drafts.]
- What the tools did:
  - Wrote and debugged code for the prompt builder (`03`): the persona f-strings and label mappings.
  - Wrote and debugged code for the forecast (`05`): the fundamentals regression, poll average, calibration line, inverse-variance blend, Monte Carlo simulation, plurality rule, and the Georgia, Alaska and Maine rules.
  - Wrote and debugged code for the charts (`07`): the state-outline map, label placement and dark-theme fixes.
  - Wrote and debugged the special-race tests.
  - Helped with the CSS layout changes, the citations and chart descriptions, and the git and GitHub steps.
  - Checked my website text against the forecast output for wrong numbers and typos.
  - Explained the statistical methods so I could present them.
- What I did myself:
  - Decided to drop the territories, to keep Method C out of the forecast, and to blend the three sources by precision.
  - Ran the pipeline and checked prompts against the persona data with print statements.
  - Reviewed every chart and code change before committing.
  - Wrote my website sections (Combining the sources, Uncertainty, the ratings and seat-count guides, AI surveys vs polls, Main findings) from my own notes on the outputs.

**Vijval Rallan**
- Tools: Claude (claude.ai chat), model Claude Opus 5.5 (`claude-opus-5-5`).
- What the tools did:
  - Listed missing items (AI statement, citations, discussion) and inconsistencies between charts (axis direction, colour meanings).
  - Wrote the sampling code in `scripts/01a-download_ces.py` that saves the 1,000-row samples `CES24_sample.csv` and `CES25_sample.dta` and adds the `--sample-only` option, plus the fallback lines in `02-clean_data.py` that use the samples when the full CES files are absent.
- What I did myself: downloaded and checked the CES data; wrote the cleaning and persona sampling; chose which suggestions to adopt; reviewed and tested the sampling code on the real CES files; edited the outputs; wrote my sections of the prose.

**Shrey Sati**
- Tools: Claude, in the claude.ai agent workspace (built on Claude Code), with Claude Opus 5.5 (`claude-opus-5-5`).
- What the tools did:
  - Pipeline scripts: assisted with the poll scraper in `01b-download_polls.py`; wrote the weighting-frame code in `02-clean_data.py` and `03-build_prompts.py`; wrote `06-validation_checks.py` and the chart code in `07-plot_forecast.py`.
  - Website code: the Quarto setup (`_quarto.yml`, `index.qmd`, the section files) and `website/page.py` (the tables, ratings strip, seat bar and the independent-candidate columns).
- What I did myself:
  - Ran the full model run and the API calls with my own OpenRouter key, and set up the repository secret and spending limit.
  - Ran every script and test locally and checked the outputs.
  - Made the design choices: the dar[…paste the rest of Shrey's text here]

### Writing

The written analysis on the website (Methods, Uncertainty and Discussion) was written by us. [If an AI tool reworded or edited any of it, say which tool and for which sections.]



## Repository layout

- `data/00-simulated_data/` simulated inputs used to test the pipeline
- `data/01-raw_data/` CES samples, polls, candidates, past results, state outlines
- `data/02-analysis_data/` personas, weighting frame, model answers (`responses/`)
- `data/03-forecast/` forecast, simulations, validation, stability and cost summaries
- `scripts/` the pipeline, run in numbered order
- `tests/` pytest checks on simulated and real data
- `other/charts/` charts drawn by `07`; `other/sketches/` our early chart sketches
- `website/` Quarto source; `docs/` the built site served by GitHub Pages

## Running it

1. Install [uv](https://github.com/astral-sh/uv), then run `uv sync`.
2. Copy `.env.example` to `.env` and add your own OpenRouter key. Never commit `.env`.
3. Run the scripts in order:
   ```
   uv run python scripts/01a-download_ces.py          # full CES files go to data/01-raw_data/ces/ (gitignored)
   uv run python scripts/01b-download_polls.py
   uv run python scripts/01c-download_governor_data.py
   uv run python scripts/02-clean_data.py
   uv run python scripts/03-build_prompts.py
   uv run python scripts/04-query_models.py --config config/models.yaml --method A   # calls the models; costs money
   uv run python scripts/04-query_models.py --config config/models.yaml --method B
   uv run python scripts/04-query_models.py --config config/models.yaml --method C
   uv run python scripts/04-query_models.py --config config/models.yaml --method V
   uv run python scripts/05a-method_a.py
   uv run python scripts/05b-method_b.py
   uv run python scripts/05-forecast.py
   uv run python scripts/06-validation_checks.py
   uv run python scripts/07-plot_forecast.py
   uv run quarto render website
   ```
   Add `--dry-run` to any `04` command to print the first prompt without calling a model. `config/models_template.yaml` shows the config format.
4. Run the tests with `uv run pytest`.
   `scripts/00-simulate_data.py` makes simulated inputs, and `--simulated` runs steps 05–07 on them without calling any model.

## Daily runs

`.github/workflows/daily-queries.yml` re-asked Methods B, C and the validation checks every day from 28 October to 1 November 2026, then reran the forecast, redrew the charts, rebuilt the site and committed the results. It needs the repository secret `OPENROUTER_API_KEY`, and it was disabled after 1 November.

## Cost

About 68,000 model calls cost about US$3.55 in total: US$2.39 for the full run on 25 October, then about US$0.23 per day.