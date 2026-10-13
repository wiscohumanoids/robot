"""LLM settings, read from the environment (no rclpy, unit-testable).

Put them in a `.env` file at the repo root; real environment
variables win over the file. Under Docker, `docker/run.sh` mounts `.env` read-only at
/home/ros2_ws/.env, where the node finds it (spaces around `=` are fine).

  OPENAI_API_KEY    (OPEN_AI_KEY also works) required for the LLM planner; without it the node falls back to
                    the canned planner
  OPENAI_MODEL      default gpt-5.6-terra
  OPENAI_BASE_URL   default https://api.openai.com/v1 (any OpenAI-compatible server)
  LLM_TIMEOUT_S     per-request limit, default 30
"""
import os
from dataclasses import dataclass

DEFAULT_MODEL = 'gpt-5.6-terra'
DEFAULT_BASE_URL = 'https://api.openai.com/v1'
DEFAULT_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = ''
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = DEFAULT_TIMEOUT_S

    @property
    def configured(self):
        return bool(self.api_key)


def _clean_value(value):
    """One pair of surrounding quotes is removed; a trailing ` # comment` is dropped
    (after the closing quote, or after whitespace in an unquoted value)."""
    value = value.strip()
    if value[:1] in ('"', "'"):
        end = value.find(value[0], 1)
        rest = value[end + 1:].strip()
        if end != -1 and (not rest or rest.startswith('#')):
            return value[1:end]
        return value
    if value.startswith('#'):
        return ''
    for i, char in enumerate(value):
        if char == '#' and value[i - 1].isspace():
            return value[:i].rstrip()
    return value


def parse_env_file(text):
    """KEY=VALUE lines -> dict. Blank lines and # comments (whole-line or after a
    value) are skipped, one pair of surrounding quotes is removed, an optional leading
    `export ` is allowed."""
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        if line.startswith('export '):
            line = line[len('export '):]
        key, _, value = line.partition('=')
        values[key.strip()] = _clean_value(value)
    return values


def find_env_file(start_dir):
    """The nearest `.env` in start_dir or any parent, else None."""
    directory = os.path.abspath(start_dir)
    while True:
        candidate = os.path.join(directory, '.env')
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


def load_env(env=None, start_dir=None):
    """Copy .env values into `env` (default os.environ) for names not already set.
    Returns the path used, or None."""
    env = os.environ if env is None else env
    path = find_env_file(start_dir or os.getcwd())
    if path is None:
        return None
    with open(path) as f:
        for key, value in parse_env_file(f.read()).items():
            env.setdefault(key, value)
    return path


def config_from_env(env=None):
    env = os.environ if env is None else env
    try:
        timeout_s = float(env.get('LLM_TIMEOUT_S') or DEFAULT_TIMEOUT_S)
    except ValueError:
        timeout_s = DEFAULT_TIMEOUT_S
    return LLMConfig(
        api_key=(env.get('OPENAI_API_KEY') or env.get('OPEN_AI_KEY') or '').strip(),
        model=(env.get('OPENAI_MODEL') or DEFAULT_MODEL).strip(),
        base_url=(env.get('OPENAI_BASE_URL') or DEFAULT_BASE_URL).strip().rstrip('/'),
        timeout_s=timeout_s)
