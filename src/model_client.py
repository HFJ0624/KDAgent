"""
Unified LLM HTTP client module.

This module provides ``ModelClient``, an HTTP client compatible with an
OpenAI Chat Completions interface, along with utility functions for loading/
querying model configs. Core features:

* Targets any OpenAI-compatible endpoint (OpenAI, Azure, vLLM, Ollama, etc.);
* Robust retry mechanism: automatic retries for network errors and recoverable
  errors such as 5xx/429;
* Exponential backoff + random jitter: avoids a "thundering herd" effect when
  a large number of clients retry at the same moment;
* Unified return structure: whether it succeeds or fails, it returns a dict
  containing fields such as ``success``, ``content``, ``status_code``,
  ``error_type``, ``error_message``, making it convenient for the caller to
  do logging and flow control.
"""

import json
import hashlib
import logging
import os
import random
import socket
import ssl
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


_NETWORK_EXCEPTIONS = (
    ConnectionResetError,
    ConnectionRefusedError,
    TimeoutError,
    socket.timeout,
    socket.gaierror,
    urllib.error.URLError,
    urllib.error.HTTPError,
    ssl.SSLError,
)

try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except Exception:
    requests = None  # type: ignore
    _HAS_REQUESTS = False


from .utils import get_api_key, sanitize_filename


class ModelClient:
    """
    OpenAI-compatible Chat Completions API client with retry support.

    This class wraps the HTTP call to the ``/chat/completions`` endpoint and
    adds exponential backoff retry, unified error classification, and structured
    returns. The attributes are as follows:

    Attributes:
        config: The raw model config dict (one entry of the ``models`` list in YAML).
        name: The model display name (for logs and filenames), default ``"unknown"``.
        provider: The model provider identifier, default ``"openai_compatible"``.
        base_url: The base address of the API, e.g. ``https://api.openai.com/v1``;
            a trailing ``/`` is stripped during construction.
        model: The model name actually sent to the API.
        temperature: Sampling temperature, default ``0.2``, controlling output randomness.
        max_tokens: Maximum number of generated tokens per request, default ``2048``.
        api_key_env: The environment variable that holds the API Key
            (e.g. ``OPENAI_API_KEY``).
        api_key: The API Key resolved at runtime; ``None`` if the env var is not set.
        logger: The logger, defaulting to a logger named ``rca_experiment``.
        timeout: Timeout for a single HTTP request (seconds), default ``120``.
        default_max_retries: Default max retry count for the ``chat`` method, default ``5``.
        default_retry_base_sleep: Default backoff base seconds for the ``chat``
            method, default ``5``.
    """

    def __init__(self, model_config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        # Save the original config dict for debugging and upper-layer inspection
        self.config = model_config
        # Display name, used for logs and generated filenames
        self.name = model_config.get("name", "unknown")
        # Provider identifier (pure metadata; does not affect request logic)
        self.provider = model_config.get("provider", "openai_compatible")
        # Base URL, with the trailing slash removed for easier path joining
        self.base_url = (model_config.get("base_url") or "").rstrip("/")
        # The actual model name sent to the server
        self.model = model_config.get("model") or ""
        # Sampling temperature, controlling output randomness
        self.temperature = float(model_config.get("temperature", 0.2))
        # Max number of tokens generated per single request
        self.max_tokens = int(model_config.get("max_tokens", 2048))
        # Internal thinking budget for reasoning models. None/0 means the
        # parameter is not sent to the server, to avoid non-reasoning models
        # such as qwen rejecting the request due to an unsupported field.
        configured_thinking_budget = model_config.get("thinking_budget")
        self.thinking_budget = (
            int(configured_thinking_budget)
            if configured_thinking_budget is not None
            and int(configured_thinking_budget) > 0
            else None
        )
        # Name of the environment variable holding the API Key
        self.api_key_env = model_config.get("api_key_env", "")
        # Resolve the API Key from the environment variable (read by utils.get_api_key)
        self.api_key = get_api_key(self.api_key_env) if self.api_key_env else None
        # Logger: prefer the caller-provided one, otherwise use the default logger
        self.logger = logger or logging.getLogger("rca_experiment")
        # HTTP request timeout (seconds)
        self.timeout = float(model_config.get("timeout", 120))
        # Default max retry count (can be overridden when calling chat)
        self.default_max_retries = int(model_config.get("max_retries", 5))
        # Default backoff base seconds (can be overridden when calling chat)
        self.default_retry_base_sleep = float(model_config.get("retry_base_sleep", 5))

    def is_configured(self) -> bool:
        """Return whether the client has a usable configured API Key."""
        return bool(self.api_key)

    def identifier(self) -> str:
        return sanitize_filename(self.name)

    def chat(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
        retry_base_sleep: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Call the chat completion endpoint with a robust retry mechanism.

        Overall flow:

        1. Determine ``max_retries``, ``timeout``, and ``retry_base_sleep`` from
           the passed arguments or instance defaults;
        2. Assemble the request URL, headers, and payload;
        3. Enter the ``for attempt in range(1, max_retries + 1)`` retry loop:
           send the request via ``_http_post_json`` and handle it by response
           status code (detailed below);
        4. On success (HTTP 200 with a valid JSON structure), return immediately
           without further retries;
        5. On failure, wait using the exponential backoff formula
           ``sleep = retry_base_sleep * 2^(attempt-1) + jitter``
           before the next round;
        6. If all retries fail, return the last error information.

        Status-code classification logic:

        * **2xx (mainly 200)**: treated as success, parse
          ``choices[0].message.content`` as the model output; if the JSON
          structure is not as expected, record ``ResponseParseError`` and retry;
        * **5xx**: classified as ``ServerError``, treated as a retryable error;
        * **429**: classified as ``RateLimitError`` (rate limiting triggered),
          treated as a retryable error;
        * **others** (e.g., 4xx non-429 client errors): classified as
          ``HTTPError`` and continue the retry flow (the upper-layer business
          decides whether to finally accept them).

        Network exception classification (done in the ``_NETWORK_EXCEPTIONS``
        catch branch):

        * ``ConnectionResetError`` → ``RemoteDisconnected``;
        * ``socket.timeout`` / ``TimeoutError`` → ``TimeoutError``;
        * ``ssl.SSLError`` → ``SSLError``;
        * ``urllib.error.HTTPError`` → ``HTTPError`` (and try to record ``status_code``);
        * ``urllib.error.URLError`` → ``URLError``;
        * other unexpected exceptions → use ``type(e).__name__`` as ``error_type``.

        Return structure:
            {
                "success": bool,        # whether the call succeeded
                "content": str,         # the text content generated by the model
                "raw_response": Any,    # the raw text returned by the server
                "status_code": int|None,# HTTP status code of the latest response
                "error_type": str|None, # error-type classification (None on success)
                "error_message": str|None, # detailed error message (None on success)
                "elapsed": float        # total elapsed time (seconds)
            }
        """
        max_retries = max_retries or self.default_max_retries
        timeout = timeout or self.timeout
        retry_base_sleep = retry_base_sleep or self.default_retry_base_sleep

        result: Dict[str, Any] = {
            "success": False,
            "content": "",
            "reasoning_content": "",
            "finish_reason": "",
            "raw_response": "",
            "status_code": None,
            "error_type": None,
            "error_message": None,
            "elapsed": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            # Logical generation calls and underlying HTTP attempts must be
            # counted separately; this field records the number of network
            # requests actually issued during this chat call, for cost/stability audit.
            "network_attempts": 0,
            # Redacted final request snapshot: only the model-visible messages and
            # request params are kept, never the Authorization value. In production
            # it is persisted by the upper layer together with case/branch metadata.
            "request_snapshot": None,
            "request_control": None,
            "request_fingerprint": "",
        }

        if not self.is_configured():
            result["error_type"] = "ConfigurationError"
            result["error_message"] = (
                f"API key not configured. Set env var '{self.api_key_env}' for model '{self.name}'."
            )
            self.logger.error("[%s] %s", self.name, result["error_message"])
            return result

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.thinking_budget is not None:
            payload["thinking_budget"] = self.thinking_budget

        redacted_headers = {
            "Authorization": "<REDACTED>",
            "Content-Type": "application/json",
        }
        request_snapshot = {
            "url": url,
            "headers": redacted_headers,
            "payload": payload,
        }
        request_json = json.dumps(
            request_snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        result["request_snapshot"] = request_snapshot
        result["request_control"] = {
            "max_retries": max_retries,
            "timeout": timeout,
            "retry_base_sleep": retry_base_sleep,
        }
        result["request_fingerprint"] = hashlib.sha256(
            request_json.encode("utf-8")
        ).hexdigest()

        last_error_type: Optional[str] = None
        last_error_message: Optional[str] = None
        last_status_code: Optional[int] = None
        start = time.time()

        for attempt in range(1, max_retries + 1):
            result["network_attempts"] = attempt
            # ---------- Single attempt: send the HTTP request ----------
            try:
                data, status_code, text = self._http_post_json(url, headers, payload, timeout)
                result["status_code"] = status_code
                result["raw_response"] = text

                # ---------- Handle by status code ----------
                if status_code == 200:
                    # 200 success: parse choices[0].message.content as output
                    # Compatible with various API response structures, also
                    # extracts reasoning_content and finish_reason
                    try:
                        content = ""
                        reasoning_content = ""
                        finish_reason = ""
                        
                        if isinstance(data, dict):
                            choices = data.get("choices", [])
                            if choices and isinstance(choices, list):
                                first_choice = choices[0]
                                if isinstance(first_choice, dict):
                                    message = first_choice.get("message")
                                    if isinstance(message, dict):
                                        content = message.get("content", "")
                                        reasoning_content = message.get("reasoning_content", "")
                                    elif isinstance(message, str):
                                        content = message
                                
                                finish_reason = first_choice.get("finish_reason", "")
                            
                        # Compatible with cases where content is a list
                        # (returned by some multimodal/tool calls)
                        if isinstance(content, list):
                            content = "".join(
                                part.get("text", "") or ""
                                for part in content
                                if isinstance(part, dict)
                            )
                        if isinstance(reasoning_content, list):
                            reasoning_content = "".join(
                                part.get("text", "") or ""
                                for part in reasoning_content
                                if isinstance(part, dict)
                            )
                        
                        result["content"] = content or ""
                        result["reasoning_content"] = reasoning_content or ""
                        result["finish_reason"] = finish_reason or ""

                        # DashScope's OpenAI-compatible response reports the total token count in
                        # usage, and reasoning models additionally list the internal
                        # reasoning_tokens in completion_tokens_details. Missing
                        # fields default to 0 for compatibility with regular models.
                        usage = data.get("usage", {}) if isinstance(data, dict) else {}
                        usage = usage if isinstance(usage, dict) else {}
                        completion_details = usage.get("completion_tokens_details", {})
                        completion_details = (
                            completion_details
                            if isinstance(completion_details, dict)
                            else {}
                        )

                        def usage_int(value: Any) -> int:
                            try:
                                return int(value or 0)
                            except (TypeError, ValueError):
                                return 0

                        result["prompt_tokens"] = usage_int(usage.get("prompt_tokens"))
                        result["completion_tokens"] = usage_int(usage.get("completion_tokens"))
                        result["reasoning_tokens"] = usage_int(
                            completion_details.get("reasoning_tokens")
                        )
                        result["total_tokens"] = usage_int(usage.get("total_tokens"))
                        
                        # Check whether content is empty but reasoning_content has content
                        if not content and reasoning_content:
                            result["error_type"] = "empty_final_content_with_reasoning"
                            result["error_message"] = "Model produced reasoning_content but no final content."
                        
                        # Check whether the output was truncated due to length
                        if finish_reason == "length":
                            if not result["error_type"]:
                                result["error_type"] = "output_truncated_by_length"
                            else:
                                result["error_type"] += ";output_truncated_by_length"
                        
                        result["success"] = True
                        result["elapsed"] = time.time() - start
                        return result
                    except (KeyError, IndexError, TypeError) as e:
                        last_error_type = "ResponseParseError"
                        last_error_message = f"Response JSON structure error: {e}"
                        self.logger.warning(
                            "[%s] Response parse error on attempt %d/%d: %s",
                            self.name, attempt, max_retries, last_error_message
                        )
                elif status_code >= 500:
                    # 5xx server error: treat as a retryable server error
                    last_error_type = "ServerError"
                    last_error_message = f"HTTP {status_code}: {text[:200]}"
                    self.logger.warning(
                        "[%s] Server error on attempt %d/%d: HTTP %d",
                        self.name, attempt, max_retries, status_code
                    )
                elif status_code == 429:
                    # 429 rate limiting: treat as a retryable rate-limit error
                    last_error_type = "RateLimitError"
                    last_error_message = f"Rate limited: HTTP 429"
                    self.logger.warning(
                        "[%s] Rate limited on attempt %d/%d",
                        self.name, attempt, max_retries
                    )
                else:
                    # Other status codes (mostly 4xx client errors): classify as a generic HTTPError
                    last_error_type = "HTTPError"
                    last_error_message = f"HTTP {status_code}: {text[:200]}"
                    self.logger.warning(
                        "[%s] HTTP error on attempt %d/%d: HTTP %d",
                        self.name, attempt, max_retries, status_code
                    )
            except _NETWORK_EXCEPTIONS as e:
                # ---------- Network-layer exception classification ----------
                error_type = type(e).__name__
                if isinstance(e, ConnectionResetError):
                    error_type = "RemoteDisconnected"
                elif isinstance(e, (socket.timeout, TimeoutError)):
                    error_type = "TimeoutError"
                elif isinstance(e, ssl.SSLError):
                    error_type = "SSLError"
                elif isinstance(e, urllib.error.HTTPError):
                    error_type = "HTTPError"
                    # Try to extract the server status code from the HTTPError
                    result["status_code"] = getattr(e, "code", None)
                elif isinstance(e, urllib.error.URLError):
                    error_type = "URLError"

                last_error_type = error_type
                last_error_message = str(e)
                self.logger.warning(
                    "[%s] Retry %d/%d failed: %s, sleeping %.1fs",
                    self.name, attempt, max_retries, error_type,
                    retry_base_sleep * (2 ** (attempt - 1))
                )
            except Exception as e:
                # Fallback: unexpected other exceptions, use the exception class name as error_type
                last_error_type = type(e).__name__
                last_error_message = str(e)
                self.logger.warning(
                    "[%s] Retry %d/%d failed with unexpected error: %s",
                    self.name, attempt, max_retries, last_error_type
                )

            # ---------- Exponential backoff before the next round ----------
            if attempt < max_retries:
                sleep_s = self._compute_backoff(attempt, retry_base_sleep)
                time.sleep(sleep_s)

        result["error_type"] = last_error_type
        result["error_message"] = last_error_message or f"All {max_retries} retries failed"
        result["elapsed"] = time.time() - start
        self.logger.error(
            "[%s] Failed after %d retries: %s - %s",
            self.name, max_retries, last_error_type, last_error_message
        )
        return result

    def _compute_backoff(self, attempt: int, base_sleep: float) -> float:
        """
        Compute the backoff time (seconds) before the ``attempt``-th retry.

        Formula:

            sleep = base_sleep * 2^(attempt - 1) + jitter

        where ``jitter`` is a uniform random number in ``[0, exponential * 0.3]``.

        Purpose of jitter:
            Exponential backoff alone makes the wait time of the N-th retry fixed
            at ``base_sleep * 2^(N-1)``; if many clients (e.g., concurrent requests
            during a batch evaluation) fail at the same time, they would all retry
            at the same moment, forming a "thundering herd" effect and giving the
            server a second shock. By adding a random jitter on top of the backoff
            time (0 ~ 30% in this implementation), the otherwise synchronized
            retry times are scattered, significantly reducing the congestion
            probability.
        """
        # Basic exponential backoff: first retry is base_sleep, second is 2*base_sleep, and so on
        exponential = base_sleep * (2 ** (attempt - 1))
        # Random jitter: uniform value in the 0 ~ 30% range of the current exponential backoff
        jitter = random.uniform(0, exponential * 0.3)
        return exponential + jitter

    def _http_post_json(
        self, url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float
    ):
        """
        Send a single JSON POST request using ``urllib``.

        Request flow:

        1. Serialize ``payload`` into a UTF-8 byte stream as the request body;
        2. Build the POST request object via ``urllib.request.Request``;
        3. Inject headers such as ``Authorization`` and ``Content-Type`` one by
           one via ``req.add_header``;
        4. Send the request via ``urllib.request.urlopen``, read the response
           body, and return the parsed JSON, HTTP status code, and raw text.

        Exception-handling strategy:

        * ``urllib.error.HTTPError``: read the error response body (for log
          review); if ``e.code`` is 0, rewrite it to 599; then continue to raise
          upward so the upper-level ``chat`` method classifies it as ``HTTPError``;
        * ``urllib.error.URLError``: further break down its ``reason`` and
          re-raise more specific exceptions such as ``socket.timeout`` /
          ``ssl.SSLError`` / ``ConnectionResetError``;
        * ``ConnectionResetError`` / ``ConnectionRefusedError`` /
          ``TimeoutError`` / ``socket.timeout`` / ``ssl.SSLError``: re-raise
          as-is, and the ``chat`` method catches and classifies them uniformly;
        * If the response body fails to parse as JSON: return
          ``(None, status, text)`` so the upper layer can proceed based on
          ``status`` (e.g., recognizing an HTML error page).

        Returns:
            Tuple[Any, int, str]: the parsed JSON object (or None), the HTTP
            status code, and the raw response text.
        """
        # Serialize the payload into a UTF-8 byte stream
        body = json.dumps(payload).encode("utf-8")
        # Build the urllib Request object (method="POST")
        req = urllib.request.Request(url, data=body, method="POST")
        # Inject the request headers (Authorization, Content-Type, etc.)
        for k, v in headers.items():
            req.add_header(k, v)

        try:
            # Send the request and read the response
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8", errors="replace")
                status = resp.status
        except urllib.error.HTTPError as e:
            # The server returned a non-2xx status: try to read the error body for logging
            text = ""
            if hasattr(e, "read"):
                try:
                    text = e.read().decode("utf-8", errors="replace")
                except Exception:
                    text = str(e)
            # Fall back to 0 when the underlying code is unavailable, then rewrite it to 599
            status = e.code if hasattr(e, "code") and e.code is not None else 0
            if status == 0:
                status = 599
            # Re-raise so the chat method classifies it uniformly
            raise
        except urllib.error.URLError as e:
            # URLError's reason may be a more specific sub-exception; refine it
            reason = getattr(e, "reason", None)
            if isinstance(reason, (socket.timeout, TimeoutError)):
                raise socket.timeout(str(reason)) from e
            if isinstance(reason, ssl.SSLError):
                raise
            if isinstance(reason, ConnectionResetError):
                raise
            raise
        except (ConnectionResetError, ConnectionRefusedError, TimeoutError, socket.timeout, ssl.SSLError):
            # Connection reset/refused, timeout, and TLS/SSL errors are re-raised
            # as-is so the chat() method classifies them uniformly.
            raise

        # Try to parse the response body as JSON; return None on failure, keeping the raw text
        try:
            return json.loads(text), status, text
        except json.JSONDecodeError:
            return None, status, text


def load_model_configs(config_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load model configs from a YAML file, returning a dict keyed by the model ``name``.

    YAML structure requirements: the top level contains a ``models`` list, each
    element being one config entry that must include a ``name`` field (otherwise
    a ``ValueError`` is raised). Example::

        models:
          - name: my-gpt
            provider: openai_compatible
            base_url: https://api.openai.com/v1
            model: gpt-4o

    Raises:
        ValueError: raised when the ``models`` field is missing/not a list, or
            when an entry lacks ``name``.

    Returns:
        Dict[str, Dict[str, Any]]: the config mapping keyed by model name.
    """
    from .utils import read_yaml

    data = read_yaml(config_path)
    models = data.get("models", [])
    if not isinstance(models, list):
        raise ValueError(f"'models' in {config_path} must be a list.")

    configs: Dict[str, Dict[str, Any]] = {}
    for cfg in models:
        name = cfg.get("name")
        if not name:
            raise ValueError(f"Model entry missing 'name' field: {cfg}")
        configs[name] = cfg
    return configs


def get_model_config(config_path: str, model_name: str) -> Dict[str, Any]:
    """
    Get a config by model name, with case-insensitive fallback lookup.

    Lookup strategy:

    1. First use exact matching (case-sensitive);
    2. If not found, match against all keys lowercased, to accommodate callers
       with inconsistent casing (e.g., ``"GPT-4o"`` vs ``"gpt-4o"``);
    3. If still not found, raise ``KeyError`` and list all available model names
       in the error message.

    Raises:
        KeyError: raised when the specified ``model_name`` does not exist in the config.
    """
    configs = load_model_configs(config_path)
    if model_name in configs:
        return configs[model_name]
    # Case-insensitive fallback lookup
    lower_map = {k.lower(): v for k, v in configs.items()}
    if model_name.lower() in lower_map:
        return lower_map[model_name.lower()]
    raise KeyError(
        f"Model '{model_name}' not found in {config_path}. Available models: {list(configs.keys())}"
    )


def list_available_models(config_path: str) -> list:
    """
    Return the names of all available models in the config file (sorted alphabetically).

    Commonly used to list candidate models in CLI/log output so callers can
    quickly inspect the configuration.
    """
    configs = load_model_configs(config_path)
    return sorted(configs.keys())
