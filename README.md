# AI Code Review and Bug Fixing Agent

Local Python code review agent. It finds common issues in `.py` files, runs Bandit, Black, and Pytest, can review code with a **local Ollama model** (no OpenAI billing), and writes a Markdown report.

GitHub integration is not implemented yet.

## Project layout

```
app/                 Core modules
  analyzer.py        Local AST-based Python analysis
  quality.py         Bandit, Black (check only), and Pytest
  ai_reviewer.py     Review comments (Ollama by default, OpenAI optional)
  ollama_reviewer.py Local AI reviewer using Ollama
  fixer.py           Safe unused-import fixes (only with --fix)
  ai_fixer.py        Ollama-generated fixes (only with --ai-fix)
  reporter.py        Markdown + JSON reports
  github_client.py   Placeholder (not implemented)
sample_code/         Example files with intentional bugs
reports/             Generated review reports
tests/               Unit tests
main.py              Command-line entry point
```

## Requirements

- Python 3.10+
- Project packages: `pip install -r requirements.txt`
- For local AI review: [Ollama](https://ollama.com) (free, runs on your computer)

OpenAI is optional. Local review does **not** need `OPENAI_API_KEY`.

## Local AI setup (Ollama)

1. Install Ollama from https://ollama.com and start the Ollama app.
2. Download a model (once):

```bash
ollama pull llama3.2
```

A coding model also works, for example `ollama pull qwen2.5-coder`.

3. Optional: set these in `.env` (no API key required):

```
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2
```

If Ollama is not installed, not running, or the model is missing, the project prints a clear error and still writes a Markdown report with static analysis plus Bandit, Black, and Pytest.

## Usage

```bash
python main.py
python main.py sample_code
python main.py path/to/file.py
python main.py sample_code --provider ollama --model llama3.2
python main.py sample_code --provider none
```

Reports are written to `reports/` as Markdown (and JSON). Original source files are **never** changed. `--fix` copies the analyzed project, applies unused-import removals on that copy, then runs syntax, Bandit, Black `--check`, and Pytest. If those checks fail or Bandit findings get worse, the fix is rejected. Accepted copies and diffs are saved under `reports/fixes/`:

```bash
python main.py sample_code --fix
```

`--fix` only removes unused import lines, and only after verifying the reported line is that import. Black is run in **check** mode and does not rewrite files.

`--ai-fix` asks local Ollama (default `llama3.2`) for minimal edits for mutable defaults, bare `except:`, `None` comparisons, `eval`/`exec`, and unused variables. Fixes are applied only on a temporary copy, validated, and saved under `reports/fixes/<timestamp>/ai_fixes/`. Original files stay unchanged. This option requires `--provider ollama` and does not use OpenAI.

```bash
python main.py sample_code --ai-fix --provider ollama
```

Optional paid cloud review (not required):

```bash
python main.py sample_code --provider openai
```

## Tests

```bash
python -m pytest tests -q
python -m unittest discover -s tests -v
```

## What it checks

Static analysis:

- Syntax errors
- Unused imports
- Unused variables
- Bare `except:`
- Mutable default arguments
- Comparisons with `None`
- `eval()` / `exec()` calls

Quality tools (report only, no automatic source edits):

- Bandit (security)
- Black (formatting check)
- Pytest (project tests)

Local AI (Ollama), when available:

- Bugs
- Security vulnerabilities
- Code quality issues
- Performance issues
- Maintainability issues
