# harness

A minimal terminal agent runtime. Any model [litellm](https://docs.litellm.ai/docs/providers) supports (Anthropic, OpenAI, Gemini, Mistral, Groq, OpenRouter, Ollama, …) can read, search, edit files and run commands in a working folder you choose. You approve each change.

```
╭─ ◆ harness v0.1.0 ───────────────────────────────╮
│   model  anthropic/claude-sonnet-5-5             │
│  folder  ~/projects/my-app                       │
│     key  ✓ ready                                 │
╰──────────────────────────────────────────────────╯
❯ add a --verbose flag to cli.py

● read_file  cli.py
  ⎿  1  import argparse
     …
● edit_file  cli.py
    @@ -12,3 +12,4 @@ …
  Allow edit_file?  ❯ Yes (y) · Yes, don't ask again (a) · No (n)
```

## Quick start

Requires Python 3.10+ and git.

**Windows (PowerShell)**

```powershell
git clone https://github.com/basel-bilumina/benchmark_harness.git
cd benchmark_harness
python -m venv .venv
.venv\Scripts\pip install -e .
.venv\Scripts\harness config key anthropic     # paste your key (hidden)
.venv\Scripts\harness -w C:\path\to\a\project
```

**macOS / Linux**

```bash
git clone https://github.com/basel-bilumina/benchmark_harness.git
cd benchmark_harness
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/harness config key anthropic
.venv/bin/harness -w ~/path/to/a/project
```

To use `harness` without the `.venv\Scripts\` prefix, activate the venv first (`.venv\Scripts\activate` or `source .venv/bin/activate`). Install with `-e ".[dev]"` if you want pytest too.

Your keys and settings are stored in `~/.harness/`, outside the repo, so they're never committed.

## Configure

Settings live in `~/.harness/config.json`. Set `HARNESS_HOME` to use a different folder.

```bash
harness config key anthropic              # prompts for the key without echoing it
harness config set model anthropic/claude-sonnet-5-5
harness config set workdir C:\path\to\project
harness config show                       # keys are always masked
```

- **Keys** are stored by env var name (`anthropic` → `ANTHROPIC_API_KEY`; any explicit `NAME_API_KEY` works too) and exported to litellm at startup.
- **Model** is any litellm model id: `openai/gpt-5`, `gemini/gemini-2.5-pro`, `openrouter/…`, `ollama/llama3`.
- **Working folder** is where the agent's file tools are confined. If you don't set one, it's the folder you launch from. To override it for a single session, use `harness -w <dir>`.
- **Other settings:** `api_base` (custom or local endpoint), `max_steps` (tool rounds per request, default 30), `temperature`, `system_prompt` (extra instructions).

### OpenAI-compatible endpoints (e.g. Entrim)

For providers litellm doesn't know by name but that speak the OpenAI API, store the key under `openai`, point `api_base` at the provider, and prefix the model id with `openai/`. Example for [Entrim](https://docs.entrim.ai/model-api/):

```bash
harness config key openai                                   # paste your Entrim key
harness config set api_base https://api.entrim.ai/v1
harness config set model openai/zai-org/GLM-5.3-Flash
```

`api_base` applies to every model, so clear it again (`harness config set api_base none`) when you switch back to a native provider.

If the working folder has an `AGENTS.md` (or `CLAUDE.md`), it is added to the system prompt.

## Use

```bash
harness                        # or: python -m harness
harness -w ../other -m openai/gpt-5
```

| command | |
|---|---|
| `/model [id]` | show or switch model (saved) |
| `/key <provider> [key]` | store a key; omit the key to type it hidden. `/key` lines are never written to history |
| `/cd <path>` | change working folder (saved) |
| `/config`, `/set <name> <value>` | view or change settings |
| `/approve <tool\|all\|off>` | auto-approve `write_file`, `edit_file`, `run_shell` |
| `/clear` | new conversation |
| `/usage` | tokens and cost |
| `/exit` | quit (Ctrl+D) |

Keys: **Enter** sends · **Alt+Enter** adds a newline · **Ctrl+C** interrupts the agent.

## Tools

| tool | approval |
|---|---|
| `list_dir`, `read_file`, `search` | no |
| `write_file`, `edit_file` (shows a diff first) | yes |
| `run_shell` (cmd.exe on Windows, sh elsewhere; cwd = working folder, timeout kills the process tree) | yes |

File tools reject any path outside the working folder. `run_shell` is **not** sandboxed: it starts in the working folder, but the command itself can reach anything your user account can. That's why it requires approval.

## Layout

```
harness/
  config.py   config file, keys, masking
  tools.py    tool implementations, schemas, workspace sandbox
  agent.py    streaming litellm loop, tool dispatch, approvals, usage
  ui.py       rich + prompt_toolkit terminal UI, slash commands
  cli.py      `harness` entry point and `harness config …`
tests/        pytest suite (no network; the model is scripted)
```

```bash
.venv\Scripts\python -m pytest
```

Notes: litellm takes several seconds to import, so it loads in the background while you type your first message. `mcp` is listed as a dependency because litellm imports it whenever tools are passed.
