# AutoNumerics

## Repo Layout

- `app/`: main runnable entrypoints
- `scripts/`: small helper scripts
- `src/`: core pipeline modules
- `outputs_new_paper/`: saved runs, generated code snapshots, and other experiment artifacts

## Quickstart

```bash
# create/sync the local environment with uv
uv sync --extra dev

# activate the local virtual environment
source .venv/bin/activate

# install the git hooks once for local development
pre-commit install

# set your Anthropic API key before running any LLM-driven step
export ANTHROPIC_API_KEY="your_api_key_here"

# run the main pipeline on one benchmark problem
python app/run_single_problem.py --problem-id wave_2d_dirichlet_sin_sin

# optional: change the random seed
python app/run_single_problem.py --problem-id wave_2d_dirichlet_sin_sin --seed 1

# optional: convert a plain-language PDE description into normalized JSON
python scripts/convert_text_to_problem.py --description "Solve the 1D heat equation on [0,1] with zero Dirichlet boundary conditions."

# optional: read the PDE description from a file instead
python scripts/convert_text_to_problem.py --description-file path/to/problem.txt
```

## Notes

- Benchmark problem IDs are currently defined in `src/problem_lists_0125.py`. (one can use only id and description)
- Main pipeline outputs are written to `outputs_new_paper/run_<problem_id>_<timestamp>/`.
- The pipeline depends on live LLM calls, so it will not run without valid API credentials and network access.
