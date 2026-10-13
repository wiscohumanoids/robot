"""One OpenAI chat call that returns JSON (standard library only, no rclpy).

Uses the Chat Completions endpoint with a strict JSON schema, so the reply is
parseable. `post` is injectable so tests never touch the network.

A transient failure (HTTP 408/429/5xx, timeout, connection error) is tried again, up to
MAX_ATTEMPTS requests in total with RETRY_WAIT_S between them; anything else fails at once.
"""
import json
import time
import urllib.error
import urllib.request


MAX_ATTEMPTS = 3
RETRY_WAIT_S = (1.0, 2.0)       # before the 2nd and the 3rd request
TRANSIENT_HTTP = (408, 429, 500, 502, 503, 504)


class LLMError(Exception):
    """The call failed or the reply was unusable. The message is user-readable.
    `transient` marks a failure that another request may not hit."""

    def __init__(self, message, transient=False):
        super().__init__(message)
        self.transient = transient


def _http_post(url, headers, body, timeout_s):
    request = urllib.request.Request(url, data=body, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', 'replace')[:300]
        raise LLMError(f'LLM request failed: HTTP {exc.code} {detail}',
                       transient=exc.code in TRANSIENT_HTTP) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMError(f'LLM request failed: {exc}', transient=True) from exc


def _post_with_retry(post, url, headers, body, timeout_s, sleep):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return post(url, headers, body, timeout_s)
        except LLMError as exc:
            if not exc.transient or attempt == MAX_ATTEMPTS:
                raise
        sleep(RETRY_WAIT_S[min(attempt, len(RETRY_WAIT_S)) - 1])


def chat_json(config, messages, schema, schema_name='plan', post=_http_post, sleep=time.sleep):
    """Send `messages`, return the reply parsed as JSON (a dict matching `schema`)."""
    if not config.configured:
        raise LLMError('no OPENAI_API_KEY configured')
    body = json.dumps({
        'model': config.model,
        'messages': messages,
        'response_format': {
            'type': 'json_schema',
            'json_schema': {'name': schema_name, 'strict': True, 'schema': schema},
        },
    }).encode('utf-8')
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {config.api_key}'}
    raw = _post_with_retry(post, f'{config.base_url}/chat/completions', headers, body,
                           config.timeout_s, sleep)
    try:
        message = json.loads(raw)['choices'][0]['message']
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LLMError('LLM reply was not a chat completion') from exc
    if message.get('refusal'):
        raise LLMError(f"LLM refused: {message['refusal']}")
    try:
        return json.loads(message.get('content') or '')
    except ValueError as exc:
        raise LLMError('LLM reply was not valid JSON') from exc
