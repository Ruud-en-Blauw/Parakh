"""OpenRouter provider adapter with strict safety and zero-cost enforcement."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .base import (
    ProviderCapabilities,
    ProviderConfigurationError,
    ProviderRequest,
    ProviderResponse,
    ProviderRuntimeError,
)

DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_TIMEOUT_SECONDS = 120.0
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_RETRIES = 3
MAX_BACKOFF_SECONDS = 10.0

_OPENROUTER_PARAMETERS = frozenset({
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "max_completion_tokens",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "stop",
    "response_format",
    "seed",
})


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        raise ProviderRuntimeError("provider redirect rejected; endpoint must be final", retryable=False)


def load_openrouter_api_key(env_path: Path | str | None = None) -> str:
    """Read the OpenRouter API key strictly from an ignored local file (.env.openrouter).

    The key is NEVER logged, printed, or included in reports.
    """
    candidate_paths: list[Path] = []
    if env_path is not None:
        candidate_paths.append(Path(env_path))
    else:
        # Check current working directory, Parakh root, and parent directories
        cur = Path.cwd().resolve()
        for directory in [cur, *cur.parents]:
            candidate_paths.append(directory / ".env.openrouter")
        repo_root = Path(__file__).resolve().parents[2]
        candidate_paths.append(repo_root / ".env.openrouter")

    for p in candidate_paths:
        if p.is_file():
            try:
                content = p.read_text(encoding="utf-8")
                for line in content.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() in {"OPENROUTER_API_KEY", "OPEN_ROUTER_KEY"}:
                            cleaned = v.strip().strip("'\"")
                            if cleaned:
                                return cleaned
            except OSError as err:
                raise ProviderConfigurationError(f"failed to read openrouter key file: {err}") from err

    raise ProviderConfigurationError(
        "OpenRouter key not found. Key must be configured strictly in .env.openrouter (OPENROUTER_API_KEY=...)"
    )


def _messages(request: ProviderRequest) -> list[dict[str, str]]:
    return [dict(message) for message in request.candidate.messages]


def _validate_endpoint(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ProviderConfigurationError("endpoint must use http or https")
    if not parsed.netloc:
        raise ProviderConfigurationError("endpoint must specify host")
    return url.rstrip("/")


def _safe_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    protected = {"model", "messages", "stream", "prompt", "tools", "functions", "tool_choice", "plugins", "web_search_options"}
    forbidden = protected.intersection(parameters)
    if forbidden:
        raise ProviderConfigurationError(f"request parameters contain forbidden keys: {sorted(forbidden)}")
    checked: dict[str, Any] = {}
    for k, v in parameters.items():
        if k in _OPENROUTER_PARAMETERS:
            checked[k] = v
    return checked


class OpenRouterAdapter:
    """OpenAI-compatible OpenRouter provider adapter.

    Guarantees:
    1. Zero-price enforcement: aborts if catalog price or returned cost is non-zero.
    2. Hard request cap counter: enforced before dispatch so cap is never exceeded.
    3. No redirect follow: rejects any HTTP redirects.
    4. Connect + read timeouts on every call.
    5. Bounded backoff-and-resume on 429/5xx honoring Retry-After.
    6. API key never logged, printed, or emitted in metadata.
    """

    name = "openrouter"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_requests: int | None = None,
        catalog_models: list[dict[str, Any]] | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_backoff: float = MAX_BACKOFF_SECONDS,
        opener: Any | None = None,
    ) -> None:
        self.model = model
        self.base_url = _validate_endpoint(base_url)
        self.endpoint = f"{self.base_url}/chat/completions"
        self._api_key = api_key
        self.max_requests = max_requests
        self.request_count = 0
        self.max_retries = max_retries
        self.max_backoff = max_backoff
        self._opener = opener or build_opener(_RejectRedirects)
        self.capabilities = ProviderCapabilities(
            self.name,
            model,
            supports_streaming=False,
            supports_tools=False,
            context_window=None,
            pricing_known=True,
        )

        # Enforce catalog zero-price if catalog models are provided
        if catalog_models is not None:
            self._validate_catalog_price(catalog_models)

    def __repr__(self) -> str:
        return f"OpenRouterAdapter(model={self.model!r}, base_url={self.base_url!r}, max_requests={self.max_requests!r})"

    def __str__(self) -> str:
        return f"OpenRouterAdapter(model={self.model!r})"

    def _get_api_key(self) -> str:
        if self._api_key is None:
            self._api_key = load_openrouter_api_key()
        return self._api_key

    def _validate_catalog_price(self, catalog_models: list[dict[str, Any]]) -> None:
        normalized_model = self.model.removeprefix("openrouter/")
        found = False
        for m in catalog_models:
            m_id = m.get("id", "").removeprefix("openrouter/")
            if m_id == normalized_model:
                found = True
                # Accept both the raw /models shape and the snapshot written by snapshot_catalog.
                pricing = m.get("pricing") or {}
                prompt_price = float(pricing.get("prompt", m.get("prompt_price_usd", 0)))
                completion_price = float(pricing.get("completion", m.get("completion_price_usd", 0)))
                if prompt_price != 0.0 or completion_price != 0.0:
                    raise ProviderConfigurationError(
                        f"catalog price for {self.model} is non-zero (prompt: {prompt_price}, completion: {completion_price}); ABORTED"
                    )
                break
        if not found and not self.model.endswith(":free"):
            # If not explicitly marked free or in free catalog, abort to be safe
            raise ProviderConfigurationError(
                f"model {self.model} not verified as free in catalog; ABORTED"
            )

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        # 1. Hard request cap counter: enforced BEFORE dispatch
        if self.max_requests is not None and self.request_count >= self.max_requests:
            raise ProviderRuntimeError(
                f"hard request cap of {self.max_requests} exceeded before dispatch",
                retryable=False,
            )
        self.request_count += 1

        # 2. Key resolution without logging
        key = self._get_api_key()

        # 3. Request payload construction
        payload: dict[str, Any] = {
            "model": self.model.removeprefix("openrouter/"),
            "messages": _messages(request),
        }
        params = _safe_parameters(request.parameters)
        payload.update(params)
        if request.seed is not None:
            payload.setdefault("seed", request.seed)

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "HTTP-Referer": "https://github.com/karmi-parakh",
            "X-Title": "Parakh ModelLab",
        }

        timeout = min(
            request.timeout_seconds if request.timeout_seconds is not None else DEFAULT_TIMEOUT_SECONDS,
            MAX_TIMEOUT_SECONDS,
        )

        # 4. Dispatch with bounded backoff-and-resume on 429/5xx honoring Retry-After
        response_data, latency_ms = self._dispatch_with_retry(payload, headers, timeout)

        # 5. Check response structure
        choices = response_data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderRuntimeError("openrouter response missing choices[0]", retryable=False)
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ProviderRuntimeError("openrouter response missing message content", retryable=False)
        finish_reason = choice.get("finish_reason", "stop")

        # 6. Usage and Returned Price validation: ABORT if non-zero cost
        usage = response_data.get("usage") or {}
        returned_cost = usage.get("cost")
        if returned_cost is not None:
            try:
                cost_float = float(returned_cost)
                if cost_float > 0.0:
                    raise ProviderRuntimeError(
                        f"openrouter returned non-zero price/cost ({cost_float}) for {self.model}; ABORTED",
                        retryable=False,
                    )
            except (ValueError, TypeError) as err:
                raise ProviderRuntimeError(f"malformed cost field in response: {err}", retryable=False) from err

        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")

        # 7. Safe metadata (never contains key or bearer tokens)
        metadata = {
            "provider": self.name,
            "model": self.model,
            "request_model": request.model,
            "provider_model": response_data.get("model"),
            "provider_response_id": response_data.get("id"),
            "latency_ms": latency_ms,
            "returned_price_usd": 0.0,
            "request_parameters": params,
        }

        return ProviderResponse(
            text=message["content"],
            finish_reason=str(finish_reason),
            input_tokens=int(input_tokens) if input_tokens is not None else None,
            output_tokens=int(output_tokens) if output_tokens is not None else None,
            cost_minor=0,
            currency="USD",
            completion_latency_ms=latency_ms,
            metadata=metadata,
        )

    def _dispatch_with_retry(
        self, payload: dict[str, Any], headers: dict[str, str], timeout: float
    ) -> tuple[dict[str, Any], float]:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = Request(self.endpoint, data=data_bytes, headers=headers, method="POST")

        attempts = 0
        while True:
            attempts += 1
            t0 = time.monotonic()
            try:
                with self._opener.open(req, timeout=timeout) as resp:
                    latency_ms = (time.monotonic() - t0) * 1000.0
                    body = resp.read(MAX_RESPONSE_BYTES + 1)
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise ProviderRuntimeError("response exceeds maximum allowed bytes", retryable=False)
                    return json.loads(body.decode("utf-8")), latency_ms
            except HTTPError as err:
                status = err.code
                if (status == 429 or 500 <= status < 600) and attempts <= self.max_retries:
                    retry_after_raw = err.headers.get("Retry-After")
                    delay = 1.0 * (2 ** (attempts - 1))
                    if retry_after_raw:
                        try:
                            delay = max(delay, float(retry_after_raw))
                        except (ValueError, TypeError):
                            pass
                    delay = min(delay, self.max_backoff)
                    time.sleep(delay)
                    continue

                err_msg = f"HTTP {status}"
                try:
                    err_body = err.read().decode("utf-8", errors="replace")
                    parsed_err = json.loads(err_body)
                    if "error" in parsed_err and "message" in parsed_err["error"]:
                        err_msg += f": {parsed_err['error']['message']}"
                except (OSError, json.JSONDecodeError):
                    pass
                raise ProviderRuntimeError(
                    f"openrouter request failed: {err_msg}",
                    retryable=(status == 429 or 500 <= status < 600),
                ) from err
            except URLError as err:
                if attempts <= self.max_retries:
                    time.sleep(min(1.0 * (2 ** (attempts - 1)), self.max_backoff))
                    continue
                raise ProviderRuntimeError(f"openrouter transport failure: {err.reason}", retryable=True) from err


def fetch_free_catalog(
    base_url: str = "https://openrouter.ai/api/v1",
    api_key: str | None = None,
    snapshot_date: str | None = None,
    opener: Any | None = None,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Fetch all models from GET /models where prompt and completion prices are strictly zero."""
    base_url = _validate_endpoint(base_url)
    url = f"{base_url}/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = Request(url, headers=headers, method="GET")
    _opener = opener or build_opener(_RejectRedirects)

    try:
        with _opener.open(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as err:
        raise ProviderRuntimeError(f"failed to fetch models catalog: {err}", retryable=True) from err

    date_str = snapshot_date or time.strftime("%Y-%m-%d")
    raw_models = data.get("data", [])
    free_models: list[dict[str, Any]] = []

    for m in raw_models:
        pricing = m.get("pricing", {})
        prompt_str = str(pricing.get("prompt", "1"))
        comp_str = str(pricing.get("completion", "1"))
        try:
            p_val = float(prompt_str)
            c_val = float(comp_str)
        except (ValueError, TypeError):
            continue

        if p_val == 0.0 and c_val == 0.0:
            free_models.append({
                "id": m.get("id"),
                "name": m.get("name", m.get("id")),
                "context_length": int(m.get("context_length") or 0),
                "supported_parameters": list(m.get("supported_parameters") or []),
                "data_terms": "unknown",
                "private_data_allowed": False,
                "snapshot_date": date_str,
                "prompt_price_usd": "0",
                "completion_price_usd": "0",
            })

    return free_models


def snapshot_catalog(
    out_dir: Path | str = "docs/live_pilots/catalog",
    base_url: str = "https://openrouter.ai/api/v1",
    api_key: str | None = None,
    snapshot_date: str | None = None,
    opener: Any | None = None,
) -> Path:
    """Fetch free models and write dated catalog JSON."""
    date_str = snapshot_date or time.strftime("%Y-%m-%d")
    free_models = fetch_free_catalog(
        base_url=base_url,
        api_key=api_key,
        snapshot_date=date_str,
        opener=opener,
    )
    target = Path(out_dir)
    if target.suffix == ".json":
        out_file = target
        out_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        target.mkdir(parents=True, exist_ok=True)
        out_file = target / f"openrouter-free-{date_str}.json"
    out_file.write_text(json.dumps(free_models, indent=2, sort_keys=True), encoding="utf-8")
    return out_file


def get_key_limits(
    base_url: str = "https://openrouter.ai/api/v1",
    api_key: str | None = None,
    opener: Any | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Query GET /key and return sanitized limits (never returning API key secret)."""
    base_url = _validate_endpoint(base_url)
    url = f"{base_url}/key"
    if not api_key:
        api_key = load_openrouter_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    req = Request(url, headers=headers, method="GET")
    _opener = opener or build_opener(_RejectRedirects)

    try:
        with _opener.open(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as err:
        raise ProviderRuntimeError(f"failed to fetch key limits: {err}", retryable=True) from err

    key_data = data.get("data", {})
    return {
        "label": key_data.get("label"),
        "limit": key_data.get("limit"),
        "limit_remaining": key_data.get("limit_remaining"),
        "is_free_tier": key_data.get("is_free_tier", True),
        "free_model_daily_requests": key_data.get("free_model_daily_requests", {}),
    }
