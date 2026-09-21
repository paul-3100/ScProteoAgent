
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
ENV_EXAMPLE_PATH = PROJECT_DIR / ".env.example"


def load_runtime_env() -> Path | None:
    # Imported here so that importing this module neither needs python-dotenv nor reads `.env`.
    from dotenv import load_dotenv

    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=True)
        return ENV_PATH
    if ENV_EXAMPLE_PATH.exists():
        load_dotenv(ENV_EXAMPLE_PATH, override=False)
        return ENV_EXAMPLE_PATH
    return None


def _env_value(name: str, required: bool = True, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if isinstance(value, str):
        value = value.strip()
    placeholder_values = {
        "",
        "your_api_key",
        "your_deepseek_api_key",
        "your_openai_compatible_api_key",
        "your_model_name",
    }
    if not required and (value is None or value in placeholder_values):
        return default if default not in placeholder_values else None
    if required and (value is None or value in placeholder_values):
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            f"Create {ENV_PATH} from .env.example and fill your DeepSeek/OpenAI-compatible settings."
        )
    return value


class MissingLLM:
    def __init__(self, reason: str):
        self.reason = reason

    def __call__(self, *_args, **_kwargs):
        raise RuntimeError(self.reason)

    def invoke(self, *_args, **_kwargs):
        raise RuntimeError(self.reason)

    def bind_tools(self, _tools):
        return self


# --- API token accounting (2026-09-12) -------------------------------------
# The pipeline previously recorded no token usage at all, so wall-clock time was the only
# observable cost signal. These hooks append one JSON line per model call to
# <run_folder>/llm_usage.jsonl; wall-clock is never converted into money.
_USAGE_LOG_PATH = None


def set_usage_log(path) -> None:
    global _USAGE_LOG_PATH
    _USAGE_LOG_PATH = str(path) if path else None


def _usage_record(response) -> dict:
    record = {"input_tokens": None, "output_tokens": None, "total_tokens": None, "model": None}
    try:
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if usage:
            record["input_tokens"] = usage.get("prompt_tokens", usage.get("input_tokens"))
            record["output_tokens"] = usage.get("completion_tokens", usage.get("output_tokens"))
            record["total_tokens"] = usage.get("total_tokens")
            record["model"] = llm_output.get("model_name")
    except Exception:
        pass
    if record["input_tokens"] is None:
        try:
            for batch in getattr(response, "generations", []) or []:
                for gen in batch:
                    meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
                    if meta:
                        record["input_tokens"] = meta.get("input_tokens")
                        record["output_tokens"] = meta.get("output_tokens")
                        record["total_tokens"] = meta.get("total_tokens")
                        break
        except Exception:
            pass
    return record


def _write_usage(record: dict) -> None:
    path = _USAGE_LOG_PATH
    if not path:
        return
    try:
        import json as _json
        import time as _time
        record = dict(record)
        record["ts"] = _time.strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(_json.dumps(record, ensure_ascii=False) + chr(10))
    except Exception:
        pass


try:
    from langchain_core.callbacks import BaseCallbackHandler as _BaseCB

    class _UsageLogger(_BaseCB):
        def on_llm_end(self, response, **kwargs):
            _write_usage(_usage_record(response))

        def on_llm_error(self, error, **kwargs):
            _write_usage({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                          "model": None, "error": str(error)[:160]})
except Exception:
    class _UsageLogger:                 # type: ignore
        def on_llm_end(self, response, **kwargs):
            _write_usage(_usage_record(response))


USAGE_LOGGER = _UsageLogger()

# --- Runtime configuration is loaded on first use, not at import time ------------------------
# Importing this module must not read `.env`, require a provider SDK or create a client: the
# command line help, the offline reproduction entry points and the strict import test all run on
# machines where the agent environment is not configured yet. Everything that used to execute at
# import time now executes in `initialize_runtime()` on the first use of one of the names below
# (through the module `__getattr__` at the end of this file) or on an explicit call.
_RUNTIME_INITIALIZED = False
_RUNTIME_NAMES = frozenset({
    "load_dotenv", "genai", "ChatOpenAI", "OpenAI",
    "LOADED_ENV_FILE", "OPENAI_MODEL", "SCORE_MODEL", "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "GOOGLE_BASE_URL", "CALL_TIMEOUT_SECONDS", "_BudgetedChatOpenAI", "LLM", "SCORELLM",
    "client", "googleclient",
})


# --- Run-level request envelope (introduced in R26, hardened in R27 2026-09-14) -----
# A production run is authorized as a bounded resource envelope: a maximum number of request
# attempts, a cumulative input-token budget, a cumulative output-token budget and one timeout
# per call. It is enforced inside the chat model because that is the single point every model
# call passes through (planner, executor, analyzer, critic, report). An attempt is booked
# before anything is sent, so a failed, timed-out or blocked call consumes the same envelope.
# Without the budget directory the guard stays disabled and the client behaves as before.
# R27 changes: (1) the R27_ env prefix is accepted with R26_ as fallback; (2) the provider
# response id, returned model name and finish_reason are read from llm_output/generation_info
# where they actually live; (3) tool-call-only responses are archived in full instead of being
# written as an empty file, and an unparsable shape is labelled instead of silently empty;
# (4) the envelope can be shared across the processes of one run through a lock-protected
# state file; (5) a blocked attempt leaves an addressable record; (6) the pre-send projection
# keeps an explicit safety margin instead of treating the observed ratio as a guarantee.

import hashlib as _hashlib
import json as _json
import threading as _threading
import time as _time


class RequestBudgetExceeded(RuntimeError):
    '''Raised before a request is sent once the authorized envelope is exhausted.'''


def _envelope_env(suffix, default=None):
    '''Read one envelope setting, accepting the R27_ prefix and falling back to R26_.'''
    for prefix in ('R27_', 'R26_'):
        value = os.getenv(prefix + suffix)
        if value is not None and str(value).strip() != '':
            return str(value).strip()
    return default


def _budget_extract(result):
    '''Return text, message, tool calls and response metadata actually carried by a ChatResult.'''
    info = {'text': '', 'message': None, 'tool_calls': None, 'parse_status': 'no_generations',
            'generations': 0, 'llm_output': {}, 'finish_reason': None}
    generations = list(getattr(result, 'generations', None) or [])
    info['generations'] = len(generations)
    llm_output = getattr(result, 'llm_output', None)
    info['llm_output'] = llm_output if isinstance(llm_output, dict) else {}
    if info['llm_output'].get('finish_reason'):
        info['finish_reason'] = info['llm_output'].get('finish_reason')
    for generation in generations:
        message = getattr(generation, 'message', None)
        if message is None:
            continue
        info['message'] = message
        generation_info = getattr(generation, 'generation_info', None) or {}
        if generation_info.get('finish_reason'):
            info['finish_reason'] = generation_info.get('finish_reason')
        content = getattr(message, 'content', None)
        text = None
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            collected = []
            for item in content:
                if isinstance(item, str):
                    collected.append(item)
                elif isinstance(item, dict):
                    collected.append(str(item.get('text') or item.get('content') or ''))
            text = ''.join(collected)
        tool_calls = getattr(message, 'tool_calls', None) or []
        if isinstance(text, str) and text.strip():
            info['text'] = text
            info['parse_status'] = 'content_text'
            if tool_calls:
                info['tool_calls'] = tool_calls
            break
        if tool_calls:
            info['tool_calls'] = tool_calls
            info['parse_status'] = 'tool_calls_only'
            if isinstance(text, str):
                info['text'] = text
            break
        if isinstance(text, str):
            info['text'] = text
            info['parse_status'] = 'empty_content'
        else:
            info['parse_status'] = 'unparsable_generation'
    return info


def _budget_usage(message, llm_output=None):
    usage = (getattr(message, 'usage_metadata', None) or {}) if message is not None else {}
    metadata = (getattr(message, 'response_metadata', None) or {}) if message is not None else {}
    token_usage = metadata.get('token_usage') or metadata.get('usage') or {}
    if not token_usage and isinstance(llm_output, dict):
        token_usage = llm_output.get('token_usage') or llm_output.get('usage') or {}

    def pick(*keys):
        for key in keys:
            value = usage.get(key)
            if value is None:
                value = token_usage.get(key)
            if value is not None:
                return value
        return None

    return {'input_tokens': pick('input_tokens', 'prompt_tokens'),
            'output_tokens': pick('output_tokens', 'completion_tokens'),
            'total_tokens': pick('total_tokens')}


class _RequestBudget:
    '''Counts attempts and tokens, writes request/response evidence, stops when exhausted.'''

    def __init__(self) -> None:
        self.enabled = False
        self.dir = None
        self.shared = False
        self.margin = 1.0
        self.max_attempts = 0
        self.max_input_tokens = 0
        self.max_output_tokens = 0
        self.per_call_cap = 0
        self.stop_reserve = 0
        self.attempts = 0
        self.input_estimated = 0
        self.input_reported = 0
        self.output_reported = 0
        # Consumed input is a single running number: provider-reported usage drives it and the
        # observed reported/estimate ratio calibrates the pre-send projection (with a margin).
        self.input_used = 0.0
        self.calibration = 1.0
        self.calibration_calls = 0
        self.failed_attempts = 0
        self.stopped = False
        self.stop_reasons = []
        self._lock = _threading.Lock()

    def configure_from_env(self) -> None:
        directory = _envelope_env('BUDGET_DIR', '')
        if not directory:
            return
        self.dir = Path(directory)
        for sub in ('requests', 'raw_responses', 'responses'):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.max_attempts = int(_envelope_env('MAX_ATTEMPTS', '26'))
        self.max_input_tokens = int(_envelope_env('MAX_INPUT_TOKENS', '240000'))
        self.max_output_tokens = int(_envelope_env('MAX_OUTPUT_TOKENS', '40000'))
        self.per_call_cap = int(_envelope_env('PER_CALL_OUTPUT_CAP', '8000'))
        self.stop_reserve = int(_envelope_env('STOP_RESERVE', '2000'))
        try:
            self.margin = max(1.0, float(_envelope_env('INPUT_MARGIN', '1.0')))
        except Exception:
            self.margin = 1.0
        self.shared = str(_envelope_env('SHARED_ENVELOPE', '0')).lower() not in ('0', '', 'false', 'no')
        self.enabled = True
        self._note({'kind': 'budget_configured',
                    'max_attempts': self.max_attempts,
                    'max_input_tokens': self.max_input_tokens,
                    'max_output_tokens': self.max_output_tokens,
                    'per_call_output_cap': self.per_call_cap,
                    'stop_reserve_tokens': self.stop_reserve,
                    'input_margin': self.margin,
                    'shared_envelope': self.shared,
                    'input_estimate': 'tiktoken o200k_base over messages plus tool schema',
                    'input_accounting': 'one running total: provider-reported input, projected with margin*calibration*estimate before sending',
                    'enforcement_points': 'request attempts, pre-send input projection, provider-reported input/output, per-call max_tokens'})

    # --- evidence log -----------------------------------------------------
    def _note(self, payload) -> None:
        if not self.dir:
            return
        try:
            record = dict(payload)
            record.setdefault('ts', _time.strftime('%Y-%m-%d %H:%M:%S'))
            record['pid'] = os.getpid()
            with open(self.dir / 'llm_calls.jsonl', 'a', encoding='utf-8') as handle:
                handle.write(_json.dumps(record, ensure_ascii=False) + chr(10))
        except Exception:
            pass

    # --- cross-process lock ----------------------------------------------
    def _lock_dir(self):
        if not self.shared or not self.dir:
            return None
        path = str(self.dir / 'envelope.lock')
        deadline = _time.time() + 5.0
        while _time.time() < deadline:
            try:
                return os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            except FileExistsError:
                _time.sleep(0.05)
            except Exception:
                self._note({'kind': 'envelope_lock_error'})
                return None
        self._note({'kind': 'envelope_lock_timeout'})
        return None

    def _unlock_dir(self, handle) -> None:
        if handle is None:
            return
        try:
            os.close(handle)
        except Exception:
            pass
        try:
            os.remove(str(self.dir / 'envelope.lock'))
        except Exception:
            pass

    # --- shared state ----------------------------------------------------
    def _shared_load(self):
        if not self.dir or not self.shared:
            return None
        path = self.dir / 'envelope_state.json'
        if not path.exists():
            return {'attempts': 0, 'input_used': 0.0, 'output_reported': 0, 'input_reported': 0,
                    'failed_attempts': 0, 'calibration': 1.0, 'calibration_calls': 0, 'stopped': False, 'pids': []}
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                data = _json.load(handle)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _shared_save(self, state) -> None:
        if not self.dir or not self.shared or not isinstance(state, dict):
            return
        try:
            state['updated_ts'] = _time.strftime('%Y-%m-%d %H:%M:%S')
            state.setdefault('limits', {'max_attempts': self.max_attempts,
                                        'max_input_tokens': self.max_input_tokens,
                                        'max_output_tokens': self.max_output_tokens})
            target = self.dir / 'envelope_state.json'
            tmp = self.dir / 'envelope_state.json.tmp'
            with open(tmp, 'w', encoding='utf-8') as handle:
                handle.write(_json.dumps(state, ensure_ascii=False, indent=2))
            os.replace(tmp, target)
        except Exception:
            pass

    # --- stop and blocked-attempt evidence -------------------------------
    def _stop(self, reasons, index, estimate) -> None:
        self.stopped = True
        self.stop_reasons = list(reasons)
        self._note({'kind': 'budget_exhausted', 'reasons': reasons, 'call_index': index,
                    'attempt_index': index, 'estimated_input_tokens': estimate,
                    'attempts_used': self.attempts, 'failed_attempts': self.failed_attempts,
                    'input_used': round(self.input_used, 1), 'calibration': round(self.calibration, 4),
                    'input_reported': self.input_reported, 'output_reported': self.output_reported,
                    'input_margin': self.margin})
        if self.dir:
            try:
                with open(self.dir / 'STOP_BUDGET_EXHAUSTED.json', 'w', encoding='utf-8') as handle:
                    handle.write(_json.dumps({'reasons': reasons, 'call_index': index, 'attempt_index': index,
                                             'attempts_used': self.attempts,
                                             'estimated_input_tokens': estimate,
                                             'ts': _time.strftime('%Y-%m-%d %H:%M:%S')},
                                            ensure_ascii=False, indent=2))
            except Exception:
                pass
            state = self._shared_load() if self.shared else None
            if state is not None:
                state['stopped'] = True
                state['stop_reasons'] = list(reasons)
                self._shared_save(state)

    def _blocked(self, index, reasons, estimate, projection, remaining_output) -> None:
        record = {'kind': 'request_blocked', 'call_index': index, 'attempt_index': index,
                  'ts': _time.strftime('%Y-%m-%d %H:%M:%S'), 'reasons': reasons,
                  'estimated_input_tokens': estimate, 'projected_input_tokens': round(projection, 1),
                  'input_used_at_block': round(self.input_used, 1), 'calibration': round(self.calibration, 4),
                  'input_margin': self.margin, 'remaining_output_tokens_at_block': remaining_output,
                  'attempts_used': self.attempts, 'limits': {'max_attempts': self.max_attempts,
                                                              'max_input_tokens': self.max_input_tokens,
                                                              'max_output_tokens': self.max_output_tokens}}
        self._note(record)
        if self.dir:
            try:
                with open(self.dir / 'requests' / ('call_%03d_blocked.json' % index), 'w', encoding='utf-8') as handle:
                    handle.write(_json.dumps(record, ensure_ascii=False, indent=2))
            except Exception:
                pass

    def estimate_input_tokens(self, messages, kwargs) -> int:
        texts = []
        try:
            for message in messages or []:
                content = getattr(message, 'content', None)
                texts.append(content if isinstance(content, str) else _json.dumps(content, ensure_ascii=False, default=str))
            if kwargs and kwargs.get('tools'):
                texts.append(_json.dumps(kwargs['tools'], ensure_ascii=False, default=str))
        except Exception:
            texts.append(str(messages)[:200000])
        blob = chr(10).join(texts)
        try:
            import tiktoken
            return len(tiktoken.get_encoding('o200k_base').encode(blob)) + 8 * len(texts)
        except Exception:
            return int(len(blob) / 3) + 8 * len(texts)

    # --- admission -------------------------------------------------------
    def begin(self, messages, model_name, kwargs):
        if not self.enabled:
            return {'enabled': False}
        estimate = self.estimate_input_tokens(messages, kwargs)
        lock_handle = self._lock_dir()
        try:
            with self._lock:
                state = self._shared_load() if self.shared else None
                if isinstance(state, dict):
                    self.attempts = int(state.get('attempts') or 0)
                    self.input_used = float(state.get('input_used') or 0.0)
                    self.input_reported = int(state.get('input_reported') or 0)
                    self.output_reported = int(state.get('output_reported') or 0)
                    self.failed_attempts = int(state.get('failed_attempts') or 0)
                    self.calibration = float(state.get('calibration') or 1.0)
                    self.calibration_calls = int(state.get('calibration_calls') or 0)
                    if state.get('stopped'):
                        self.stopped = True
                index = self.attempts + 1
                remaining_output = self.max_output_tokens - self.output_reported
                projection = self.margin * self.calibration * estimate
                reasons = []
                if self.stopped:
                    reasons.append('already_stopped')
                if index > self.max_attempts:
                    reasons.append('request_attempts')
                if self.input_used + projection > self.max_input_tokens:
                    reasons.append('input_tokens')
                if remaining_output < self.stop_reserve:
                    reasons.append('output_tokens')
                if reasons:
                    self._blocked(index, reasons, estimate, projection, remaining_output)
                    self._stop(reasons, index, estimate)
                    raise RequestBudgetExceeded(
                        'request envelope exhausted (%s): attempts %d/%d booked, input %.0f + %.0f projected'
                        ' (margin %.2f, calibration %.3f) of %d, output %d/%d'
                        % (','.join(reasons), self.attempts, self.max_attempts, self.input_used, projection,
                           self.margin, self.calibration, self.max_input_tokens,
                           self.output_reported, self.max_output_tokens))
                self.attempts = index
                self.input_estimated += estimate
                cap = int(min(self.per_call_cap, max(512, remaining_output - 512)))
                prompt_blob = chr(10).join(
                    (getattr(m, 'content', '') if isinstance(getattr(m, 'content', ''), str)
                     else _json.dumps(getattr(m, 'content', None), ensure_ascii=False, default=str))
                    for m in (messages or []))
                request = {'kind': 'request', 'call_index': index,
                           'ts': _time.strftime('%Y-%m-%d %H:%M:%S'),
                           'model_requested': model_name,
                           'prompt_chars': len(prompt_blob),
                           'prompt_sha256': _hashlib.sha256(prompt_blob.encode('utf-8', 'replace')).hexdigest(),
                           'estimated_input_tokens': estimate,
                           'projected_input_tokens': round(projection, 1),
                           'input_used_before_call': round(self.input_used, 1),
                           'input_margin': self.margin,
                           'max_completion_tokens_requested': cap,
                           'remaining_output_tokens_at_send': remaining_output,
                           'n_messages': len(messages or []),
                           'shared_envelope': bool(self.shared),
                           'tool_schema_chars': len(_json.dumps((kwargs or {}).get('tools') or [], ensure_ascii=False, default=str))}
                try:
                    with open(self.dir / 'requests' / ('call_%03d.json' % index), 'w', encoding='utf-8') as handle:
                        handle.write(_json.dumps(request, ensure_ascii=False, indent=2))
                    with open(self.dir / 'requests' / ('call_%03d_prompt.txt' % index), 'w', encoding='utf-8') as handle:
                        handle.write(prompt_blob)
                    tools_blob = (kwargs or {}).get('tools')
                    if tools_blob:
                        with open(self.dir / 'requests' / ('call_%03d_tools.json' % index), 'w', encoding='utf-8') as handle:
                            handle.write(_json.dumps(tools_blob, ensure_ascii=False, default=str))
                except Exception:
                    pass
                if isinstance(state, dict):
                    state['attempts'] = index
                    pids = state.get('pids')
                    if not isinstance(pids, list):
                        pids = []
                    if os.getpid() not in pids:
                        pids.append(os.getpid())
                    state['pids'] = pids
                    self._shared_save(state)
                self._note(request)
        finally:
            self._unlock_dir(lock_handle)
        return {'enabled': True, 'index': index, 'request_kwargs': {'max_tokens': cap},
                'estimate': estimate, 'cap': cap, 'started': _time.perf_counter(),
                'request': request}

    # --- outcome ---------------------------------------------------------
    def finish(self, reservation, result) -> None:
        if not reservation.get('enabled'):
            return
        index = reservation['index']
        info = _budget_extract(result)
        message = info.get('message')
        llm_output = info.get('llm_output') or {}
        usage = _budget_usage(message, llm_output)
        metadata = (getattr(message, 'response_metadata', None) or {}) if message is not None else {}
        raw_text = info.get('text') or ''
        content_hash = _hashlib.sha256(raw_text.encode('utf-8', 'replace')).hexdigest()
        tool_calls = info.get('tool_calls') or None
        tool_blob = _json.dumps(tool_calls, ensure_ascii=False, default=str) if tool_calls else ''
        tool_hash = _hashlib.sha256(tool_blob.encode('utf-8', 'replace')).hexdigest() if tool_blob else ''
        elapsed = round(_time.perf_counter() - reservation.get('started', _time.perf_counter()), 3)
        returned_model = (metadata.get('model_name') or llm_output.get('model_name')
                          or llm_output.get('model') or 'not_provided')
        response_id = metadata.get('id') or llm_output.get('id') or 'not_provided'
        finish_reason = info.get('finish_reason') or metadata.get('finish_reason') or 'not_provided'
        reported = int(usage['input_tokens'] or 0)
        output_tokens = int(usage['output_tokens'] or 0)
        record = {'kind': 'response', 'call_index': index,
                  'ts': _time.strftime('%Y-%m-%d %H:%M:%S'), 'elapsed_s': elapsed,
                  'model_requested': (reservation.get('request') or {}).get('model_requested'),
                  'returned_model_id': str(returned_model), 'response_id': str(response_id),
                  'finish_reason': str(finish_reason), 'parse_status': info.get('parse_status'),
                  'generations': info.get('generations'), 'content_chars': len(raw_text),
                  'content_sha256': content_hash, 'tool_calls_count': len(tool_calls or []),
                  'tool_calls_sha256': tool_hash, 'response_chars': len(raw_text),
                  'response_sha256': content_hash, 'usage_input_tokens': usage['input_tokens'],
                  'usage_output_tokens': usage['output_tokens'], 'usage_total_tokens': usage['total_tokens'],
                  'input_accounted_by': 'provider_reported' if reported > 0 else 'estimate_only',
                  'estimated_input_tokens': reservation.get('estimate'),
                  'shared_envelope': bool(self.shared)}
        # The raw response is persisted before anything downstream can read it. A tool-call-only
        # response is not an empty response: the payload is archived here in full.
        if self.dir:
            try:
                if raw_text:
                    body = raw_text
                elif tool_calls:
                    body = _json.dumps({'parse_status': info.get('parse_status'),
                                        'note': 'no textual content; the tool-call payload is archived here',
                                        'tool_calls': tool_calls}, ensure_ascii=False, indent=2)
                else:
                    body = _json.dumps({'parse_status': info.get('parse_status'),
                                        'note': 'response carried neither text nor tool calls',
                                        'generations': info.get('generations')}, ensure_ascii=False, indent=2)
                with open(self.dir / 'raw_responses' / ('call_%03d.md' % index), 'w', encoding='utf-8') as handle:
                    handle.write(body)
                with open(self.dir / 'raw_responses' / ('call_%03d.json' % index), 'w', encoding='utf-8') as handle:
                    handle.write(_json.dumps({'call_index': index, 'text': raw_text,
                                              'parse_status': info.get('parse_status'), 'tool_calls': tool_calls,
                                              'response_id': str(response_id), 'returned_model_id': str(returned_model),
                                              'finish_reason': str(finish_reason), 'usage': usage},
                                             ensure_ascii=False, indent=2))
            except Exception:
                pass
        lock_handle = self._lock_dir()
        try:
            with self._lock:
                estimated = int(reservation.get('estimate') or 0)
                self.input_reported += reported
                self.output_reported += output_tokens
                if self.shared:
                    state = self._shared_load() or {}
                    if reported > 0:
                        state['calibration_calls'] = int(state.get('calibration_calls') or 0) + 1
                        state['input_used'] = float(state.get('input_used') or 0.0) + reported
                        if estimated > 0:
                            ratio = reported / float(estimated)
                            weight = 1.0 / int(state['calibration_calls'])
                            current = float(state.get('calibration') or 1.0)
                            state['calibration'] = max(0.5, min(1.5, current * (1 - weight) + ratio * weight))
                    else:
                        state['input_used'] = float(state.get('input_used') or 0.0) + max(0.5, float(state.get('calibration') or 1.0)) * estimated
                    state['input_reported'] = int(state.get('input_reported') or 0) + reported
                    state['output_reported'] = int(state.get('output_reported') or 0) + output_tokens
                    self._shared_save(state)
                    self.input_used = float(state.get('input_used') or 0.0)
                    self.output_reported = int(state.get('output_reported') or 0)
                    self.calibration = float(state.get('calibration') or 1.0)
                    self.calibration_calls = int(state.get('calibration_calls') or 0)
                elif reported > 0:
                    self.calibration_calls += 1
                    self.input_used += reported
                    if estimated > 0:
                        ratio = reported / float(estimated)
                        weight = 1.0 / self.calibration_calls
                        self.calibration = max(0.5, min(1.5, self.calibration * (1 - weight) + ratio * weight))
                else:
                    self.input_used += self.calibration * estimated
                record['cumulative_input_used'] = round(self.input_used, 1)
                record['calibration'] = round(self.calibration, 4)
                record['cumulative_input_reported'] = self.input_reported
                record['cumulative_output_reported'] = self.output_reported
                record['attempts_used'] = self.attempts
                record['input_margin'] = self.margin
        finally:
            self._unlock_dir(lock_handle)
        if self.dir:
            try:
                with open(self.dir / 'responses' / ('call_%03d.json' % index), 'w', encoding='utf-8') as handle:
                    handle.write(_json.dumps(record, ensure_ascii=False, indent=2))
            except Exception:
                pass
        self._note(record)

    def fail(self, reservation, error) -> None:
        if not reservation.get('enabled'):
            return
        # The attempt was booked before sending, so a failure consumes it. The input side is
        # charged at the calibrated estimate because the provider usage is unknown after an error.
        estimated = int(reservation.get('estimate') or 0)
        charge = self.calibration * estimated
        lock_handle = self._lock_dir()
        try:
            with self._lock:
                self.failed_attempts += 1
                self.input_used += charge
                if self.shared:
                    state = self._shared_load() or {}
                    state['failed_attempts'] = int(state.get('failed_attempts') or 0) + 1
                    state['input_used'] = float(state.get('input_used') or 0.0) + charge
                    self._shared_save(state)
        finally:
            self._unlock_dir(lock_handle)
        self._note({'kind': 'response_failed', 'call_index': reservation.get('index'),
                    'ts': _time.strftime('%Y-%m-%d %H:%M:%S'),
                    'elapsed_s': round(_time.perf_counter() - reservation.get('started', _time.perf_counter()), 3),
                    'error_type': type(error).__name__, 'error': str(error)[:600],
                    'input_accounted_by': 'estimate_only_failed_attempt',
                    'estimated_input_tokens': estimated, 'charged_input_tokens': round(charge, 1),
                    'attempts_used': self.attempts})

    def snapshot(self) -> dict:
        return {'enabled': self.enabled, 'shared': self.shared, 'attempts': self.attempts,
                'failed_attempts': self.failed_attempts, 'input_estimated': self.input_estimated,
                'input_reported': self.input_reported, 'input_used': round(self.input_used, 1),
                'calibration': round(self.calibration, 4), 'input_margin': self.margin,
                'output_reported': self.output_reported, 'stopped': self.stopped,
                'stop_reasons': list(self.stop_reasons),
                'limits': {'max_attempts': self.max_attempts, 'max_input_tokens': self.max_input_tokens,
                           'max_output_tokens': self.max_output_tokens, 'per_call_output_cap': self.per_call_cap}}


REQUEST_BUDGET = _RequestBudget()


def initialize_runtime() -> None:
    """Load the runtime environment and build the provider clients, exactly once, on first use.

    Values keep their original module-level names, so every caller is unchanged."""
    global _RUNTIME_INITIALIZED
    global load_dotenv, genai, ChatOpenAI, OpenAI
    global LOADED_ENV_FILE, OPENAI_MODEL, SCORE_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL
    global GOOGLE_BASE_URL, CALL_TIMEOUT_SECONDS
    global _BudgetedChatOpenAI, LLM, SCORELLM, client, googleclient, _missing_reason

    if _RUNTIME_INITIALIZED:
        return
    _RUNTIME_INITIALIZED = True

    from dotenv import load_dotenv
    from google import genai
    from langchain_openai import ChatOpenAI
    from openai import OpenAI

    LOADED_ENV_FILE = load_runtime_env()

    OPENAI_MODEL = _env_value("OPENAI_MODEL", required=False, default="local-fallback")
    SCORE_MODEL = _env_value("SCORE_MODEL", required=False, default=OPENAI_MODEL)
    OPENAI_API_KEY = _env_value("OPENAI_API_KEY", required=False)
    OPENAI_BASE_URL = _env_value("OPENAI_BASE_URL", required=False)
    GOOGLE_BASE_URL = _env_value("GOOGLE_BASE_URL", required=False)

    REQUEST_BUDGET.configure_from_env()
    CALL_TIMEOUT_SECONDS = float(_envelope_env('CALL_TIMEOUT_SEC', '0') or 0) or None

    class _BudgetedChatOpenAI(ChatOpenAI):
        '''ChatOpenAI with the run-level envelope applied around every request.'''

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            reservation = REQUEST_BUDGET.begin(messages, self.model_name, kwargs)
            call_kwargs = dict(kwargs)
            call_kwargs.update(reservation.get('request_kwargs') or {})
            try:
                result = super()._generate(messages, stop=stop, run_manager=run_manager, **call_kwargs)
            except BaseException as error:
                REQUEST_BUDGET.fail(reservation, error)
                raise
            REQUEST_BUDGET.finish(reservation, result)
            return result

    if OPENAI_API_KEY and OPENAI_BASE_URL:
        LLM = _BudgetedChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            callbacks=[USAGE_LOGGER],
            max_retries=0,
            request_timeout=CALL_TIMEOUT_SECONDS,
        )

        SCORELLM = _BudgetedChatOpenAI(
            model=SCORE_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            callbacks=[USAGE_LOGGER],
            max_retries=0,
            request_timeout=CALL_TIMEOUT_SECONDS,
        )

        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

        googleclient = None
        if GOOGLE_BASE_URL:
            googleclient = genai.Client(
                api_key=OPENAI_API_KEY,
                http_options={
                    "base_url": GOOGLE_BASE_URL,
                },
            )
    else:
        _missing_reason = (
            f"OPENAI_API_KEY/OPENAI_BASE_URL is not configured. "
            f"Create {ENV_PATH} from .env.example for LLM mode; current run will use local deterministic fallbacks where available."
        )
        LLM = MissingLLM(_missing_reason)
        SCORELLM = MissingLLM(_missing_reason)
        client = None
        googleclient = None


def __getattr__(name):
    """Resolve the lazily initialised runtime names of this module (PEP 562)."""
    if name in _RUNTIME_NAMES:
        initialize_runtime()
        if name in globals():
            return globals()[name]
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
