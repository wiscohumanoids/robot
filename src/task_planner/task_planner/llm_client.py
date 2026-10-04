"""One OpenAI chat call that returns JSON (standard library only, no rclpy).

Uses the Chat Completions endpoint with a strict JSON schema, so the reply is
parseable. `post` is injectable so tests never touch the network.
"""
import json
import urllib.error
import urllib.request


class LLMError(Exception):
    """The call failed or the reply was unusable. The message is user-readable."""


def _http_post(url, headers, body, timeout_s):
    request = urllib.request.Request(url, data=body, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', 'replace')[:300]
        raise LLMError(f'LLM request failed: HTTP {exc.code} {detail}') from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMError(f'LLM request failed: {exc}') from exc


def chat_json(config, messages, schema, schema_name='plan', post=_http_post):
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
    raw = post(f'{config.base_url}/chat/completions', headers, body, config.timeout_s)
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
