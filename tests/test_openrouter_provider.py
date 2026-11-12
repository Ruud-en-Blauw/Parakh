"""Tests for OpenRouter provider adapter, zero-price enforcement, request cap, and registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Self
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from contracts.schema_check import errors as schema_errors
from model_lab.benchmark import load_suite
from model_lab.cli.commands import main as cli_main
from model_lab.domain.isolation import candidate_input
from model_lab.providers.base import (
    ProviderConfigurationError,
    ProviderRequest,
    ProviderRuntimeError,
)
from model_lab.providers.openrouter import (
    OpenRouterAdapter,
    _RejectRedirects,
    fetch_free_catalog,
    load_openrouter_api_key,
    snapshot_catalog,
)
from model_lab.providers.registry import (
    build_model_registry,
    build_shortlist_per_class,
    determine_model_class,
)


class MockResponse:
    def __init__(self, data: dict[str, Any], headers: dict[str, str] | None = None, status: int = 200) -> None:
        self._body = json.dumps(data).encode("utf-8")
        self.headers = headers or {}
        self.code = status

    def read(self, amt: int | None = None) -> bytes:
        return self._body

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass


class MockOpener:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.dispatched: list[Request] = []

    def open(self, req: Request, timeout: float = 60.0) -> Any:
        self.dispatched.append(req)
        if not self.responses:
            raise RuntimeError("no more mocked responses")
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp
ROOT = Path(__file__).parents[1]


def _make_candidate_request(model: str = "test/free-model:free") -> ProviderRequest:
    case = load_suite(ROOT / "benchmarks/seed_cases.jsonl").cases[0]
    return ProviderRequest(
        candidate=candidate_input(case),
        model=model,
    )


def test_zero_price_filter() -> None:
    catalog_data = {
        "data": [
            {
                "id": "vendor/free-model:free",
                "name": "Vendor Free",
                "context_length": 65536,
                "supported_parameters": ["temperature", "top_p"],
                "pricing": {"prompt": "0", "completion": "0"},
            },
            {
                "id": "vendor/paid-model",
                "name": "Vendor Paid",
                "context_length": 131072,
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            },
            {
                "id": "vendor/prompt-free-only",
                "name": "Prompt Free Only",
                "context_length": 32768,
                "pricing": {"prompt": "0", "completion": "0.000001"},
            },
        ]
    }
    mock_opener = MockOpener([MockResponse(catalog_data)])
    models = fetch_free_catalog(
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        snapshot_date="2026-10-18",
        opener=mock_opener,
    )
    assert len(models) == 1
    assert models[0]["id"] == "vendor/free-model:free"
    assert models[0]["prompt_price_usd"] == "0"
    assert models[0]["completion_price_usd"] == "0"
    assert models[0]["private_data_allowed"] is False
    assert models[0]["data_terms"] == "unknown"


def test_nonzero_catalog_price_aborts() -> None:
    catalog = [
        {"id": "vendor/model-a", "pricing": {"prompt": "0.001", "completion": "0"}},
        {"id": "vendor/model-b", "pricing": {"prompt": "0", "completion": "0.002"}},
    ]
    with pytest.raises(ProviderConfigurationError, match="non-zero"):
        OpenRouterAdapter(
            "vendor/model-a",
            api_key="test-key",
            catalog_models=catalog,
        )

    with pytest.raises(ProviderConfigurationError, match="non-zero"):
        OpenRouterAdapter(
            "vendor/model-b",
            api_key="test-key",
            catalog_models=catalog,
        )


def test_nonzero_returned_price_aborts() -> None:
    response_with_cost = {
        "id": "resp-123",
        "model": "vendor/free-model:free",
        "choices": [{"message": {"content": "Hello student!"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cost": 0.00015,  # Non-zero returned price
        },
    }
    opener = MockOpener([MockResponse(response_with_cost)])
    adapter = OpenRouterAdapter("vendor/free-model:free", api_key="test-key", opener=opener)
    with pytest.raises(ProviderRuntimeError, match="non-zero price/cost"):
        adapter.generate(_make_candidate_request("vendor/free-model:free"))


def test_request_cap_never_exceeded() -> None:
    valid_response = {
        "id": "resp-123",
        "model": "vendor/free-model:free",
        "choices": [{"message": {"content": "Answer"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0},
    }
    opener = MockOpener([MockResponse(valid_response), MockResponse(valid_response)])
    adapter = OpenRouterAdapter(
        "vendor/free-model:free",
        api_key="test-key",
        max_requests=2,
        opener=opener,
    )

    req = _make_candidate_request("vendor/free-model:free")
    resp1 = adapter.generate(req)
    assert resp1.text == "Answer"
    resp2 = adapter.generate(req)
    assert resp2.text == "Answer"
    assert len(opener.dispatched) == 2

    # Third call must be rejected BEFORE dispatch
    with pytest.raises(ProviderRuntimeError, match="hard request cap of 2 exceeded before dispatch"):
        adapter.generate(req)

    # Dispatch count remains 2 (cap strictly enforced before dispatch)
    assert len(opener.dispatched) == 2


def test_bounded_429_backoff_resume_honoring_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    err_429_a = HTTPError(
        url="https://openrouter.ai/api/v1/chat/completions",
        code=429,
        msg="Too Many Requests",
        hdrs={"Retry-After": "2.5"},  # type: ignore[arg-type]
        fp=None,  # type: ignore[arg-type]
    )
    err_429_b = HTTPError(
        url="https://openrouter.ai/api/v1/chat/completions",
        code=429,
        msg="Too Many Requests",
        hdrs={"Retry-After": "1.0"},  # type: ignore[arg-type]
        fp=None,  # type: ignore[arg-type]
    )
    success = MockResponse({
        "id": "resp-456",
        "model": "vendor/free-model:free",
        "choices": [{"message": {"content": "Recovered!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "cost": 0.0},
    })

    opener = MockOpener([err_429_a, err_429_b, success])
    adapter = OpenRouterAdapter(
        "vendor/free-model:free",
        api_key="test-key",
        max_retries=3,
        opener=opener,
    )

    resp = adapter.generate(_make_candidate_request("vendor/free-model:free"))
    assert resp.text == "Recovered!"
    assert len(opener.dispatched) == 3
    # Two sleep calls, honoring retry-after
    assert len(sleep_calls) == 2
    assert sleep_calls[0] == 2.5
    assert sleep_calls[1] == 2.0  # max(2.0 exponential, 1.0 retry-after)


def test_bounded_retries_exhausted_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    err_429 = HTTPError(
        url="https://openrouter.ai/api/v1/chat/completions",
        code=429,
        msg="Rate limit",
        hdrs={},  # type: ignore[arg-type]
        fp=None,  # type: ignore[arg-type]
    )
    opener = MockOpener([err_429, err_429, err_429, err_429])
    adapter = OpenRouterAdapter(
        "vendor/free-model:free",
        api_key="test-key",
        max_retries=2,
        opener=opener,
    )

    with pytest.raises(ProviderRuntimeError, match="HTTP 429"):
        adapter.generate(_make_candidate_request("vendor/free-model:free"))
    assert len(opener.dispatched) == 3  # initial + 2 retries
    assert len(sleep_calls) == 2


def test_api_key_never_appears_in_logs_reports(tmp_path: Path) -> None:
    secret_key = "sk-or-v1-secret-test-key-never-leak-998877"
    env_file = tmp_path / ".env.openrouter"
    env_file.write_text(f"OPENROUTER_API_KEY={secret_key}\n", encoding="utf-8")

    loaded = load_openrouter_api_key(env_file)
    assert loaded == secret_key

    valid_response = {
        "id": "resp-789",
        "model": "vendor/free-model:free",
        "choices": [{"message": {"content": "Safe content"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 6, "cost": 0.0},
    }
    opener = MockOpener([MockResponse(valid_response)])
    adapter = OpenRouterAdapter(
        "vendor/free-model:free",
        api_key=secret_key,
        opener=opener,
    )
    resp = adapter.generate(_make_candidate_request("vendor/free-model:free"))

    # Assert secret key is NOT in metadata or string representation
    serialized_metadata = json.dumps(resp.metadata)
    assert secret_key not in serialized_metadata
    assert secret_key not in repr(resp)
    assert secret_key not in str(resp)


def test_redirects_not_followed() -> None:
    handler = _RejectRedirects()
    req = Request("https://openrouter.ai/api/v1/chat/completions")
    with pytest.raises(ProviderRuntimeError, match="redirect rejected"):
        handler.redirect_request(req, None, 301, "Moved Permanently", {}, "https://evil.com")


def test_registry_builder_validates_against_schema() -> None:
    schema_path = Path(__file__).resolve().parents[1] / "benchmarks_student" / "model_registry.v1.schema.json"
    assert schema_path.exists(), "Schema file must exist"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    model_evals = [
        {
            "model_id": "meta-llama/llama-3.3-70b-instruct:free",
            "provider": "openrouter",
            "context_length": 131072,
            "scores": {
                "code": 0.82,
                "maths": 0.75,
                "study": 0.88,
                "hindi": 0.70,
                "mixed": 0.74,
                "summary": 0.80,
            },
            "supports_structured_output": True,
        },
        {
            "model_id": "google/gemini-2.0-flash-lite-preview:free",
            "provider": "openrouter",
            "context_length": 32768,
            "scores": {
                "code": 0.65,
                "maths": 0.60,
                "study": 0.70,
                "hindi": 0.55,
                "mixed": 0.58,
                "summary": 0.62,
            },
            "supports_structured_output": False,
        },
        {
            "model_id": "mistralai/mistral-7b-instruct:free",
            "provider": "openrouter",
            "context_length": 8192,
            "scores": {
                "code": 0.40,
                "maths": 0.35,
                "study": 0.45,
                "hindi": 0.30,
                "mixed": 0.32,
                "summary": 0.38,
            },
            "supports_structured_output": False,
        },
    ]

    registry = build_model_registry(model_evals, snapshot_date="2026-10-18")
    validation_errors = schema_errors(registry, schema)
    assert not validation_errors, f"Validation errors: {validation_errors}"

    # Verify capability classes
    models = registry["models"]
    assert models[0]["class"] == "advanced"
    assert models[1]["class"] == "balanced"
    assert models[2]["class"] == "economy"

    # Test shortlister
    shortlist = build_shortlist_per_class(registry, top_n=1)
    assert shortlist["advanced"] == ["meta-llama/llama-3.3-70b-instruct:free"]
    assert shortlist["balanced"] == ["google/gemini-2.0-flash-lite-preview:free"]
    assert shortlist["economy"] == ["mistralai/mistral-7b-instruct:free"]


def test_determine_model_class() -> None:
    assert determine_model_class(131072, {"code": 0.8, "maths": 0.8, "study": 0.8, "hindi": 0.8, "mixed": 0.8, "summary": 0.8}) == "advanced"
    assert determine_model_class(32768, {"code": 0.6, "maths": 0.6, "study": 0.6, "hindi": 0.6, "mixed": 0.6, "summary": 0.6}) == "balanced"
    assert determine_model_class(8192, {"code": 0.4, "maths": 0.4, "study": 0.4, "hindi": 0.4, "mixed": 0.4, "summary": 0.4}) == "economy"


def test_catalog_snapshot_with_fake_transport(tmp_path: Path) -> None:
    catalog_data = {
        "data": [
            {
                "id": "qwen/qwen-2.5-72b-instruct:free",
                "name": "Qwen 2.5 72B",
                "context_length": 65536,
                "supported_parameters": ["temperature", "top_p"],
                "pricing": {"prompt": "0", "completion": "0"},
            },
            {
                "id": "paid/vendor-model",
                "name": "Paid Vendor",
                "context_length": 32768,
                "pricing": {"prompt": "0.0001", "completion": "0.0002"},
            },
        ]
    }
    mock_opener = MockOpener([MockResponse(catalog_data)])
    snapshot_file = snapshot_catalog(
        out_dir=tmp_path / "catalog",
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        snapshot_date="2026-10-18",
        opener=mock_opener,
    )

    assert snapshot_file.exists()
    content = json.loads(snapshot_file.read_text(encoding="utf-8"))
    assert len(content) == 1
    assert content[0]["id"] == "qwen/qwen-2.5-72b-instruct:free"
    assert content[0]["prompt_price_usd"] == "0"
    assert content[0]["completion_price_usd"] == "0"
    assert content[0]["private_data_allowed"] is False
    assert content[0]["data_terms"] == "unknown"


def test_catalog_snapshot_cli_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    catalog_data = {
        "data": [
            {
                "id": "deepseek/deepseek-r1:free",
                "name": "DeepSeek R1 Free",
                "context_length": 65536,
                "supported_parameters": ["temperature"],
                "pricing": {"prompt": "0", "completion": "0"},
            }
        ]
    }
    mock_opener = MockOpener([MockResponse(catalog_data)])
    monkeypatch.setattr("model_lab.providers.openrouter.build_opener", lambda *args: mock_opener)

    out_dir = tmp_path / "cli_catalog"
    code = cli_main([
        "catalog",
        "snapshot",
        "--out",
        str(out_dir),
        "--base-url",
        "https://openrouter.ai/api/v1",
        "--date",
        "2026-10-18",
    ])
    assert code == 0
    captured = capsys.readouterr()
    assert '"status": "ok"' in captured.out

    expected_file = out_dir / "openrouter-free-2026-10-18.json"
    assert expected_file.exists()
    data = json.loads(expected_file.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["id"] == "deepseek/deepseek-r1:free"
