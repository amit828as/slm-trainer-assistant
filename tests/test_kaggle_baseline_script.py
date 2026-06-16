import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _load_script_module() -> types.ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_kaggle_baseline.py"
    spec = importlib.util.spec_from_file_location("run_kaggle_baseline_under_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_hf_model_uses_processor_for_gemma4(monkeypatch) -> None:
    script = _load_script_module()
    calls: dict[str, list[tuple[str, dict]]] = {
        "processor": [],
        "tokenizer": [],
        "model": [],
    }

    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            return SimpleNamespace(
                model_type="gemma4",
                architectures=["Gemma4ForConditionalGeneration"],
            )

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            calls["processor"].append((model_name, kwargs))
            return SimpleNamespace(kind="processor")

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            calls["tokenizer"].append((model_name, kwargs))
            return SimpleNamespace(kind="tokenizer")

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            calls["model"].append((model_name, kwargs))
            return SimpleNamespace(kind="model")

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = FakeAutoConfig
    fake_transformers.AutoProcessor = FakeAutoProcessor
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    text_processor, model = script.load_hf_model("google/gemma-4-E4B-it")

    assert text_processor.kind == "processor"
    assert model.kind == "model"
    assert calls["processor"] == [
        (
            "google/gemma-4-E4B-it",
            {"padding_side": "left", "trust_remote_code": True},
        )
    ]
    assert calls["tokenizer"] == []
    assert calls["model"] == [
        (
            "google/gemma-4-E4B-it",
            {"device_map": "auto", "dtype": "auto", "trust_remote_code": True},
        )
    ]


def test_load_hf_model_uses_tokenizer_for_non_gemma4(monkeypatch) -> None:
    script = _load_script_module()
    calls: dict[str, list[str]] = {
        "processor": [],
        "tokenizer": [],
    }

    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            return SimpleNamespace(model_type="llama", architectures=["LlamaForCausalLM"])

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            calls["processor"].append(model_name)
            return SimpleNamespace(kind="processor")

    class FakeAutoTokenizer:
        eos_token_id = 2

        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            calls["tokenizer"].append(model_name)
            return SimpleNamespace(kind="tokenizer")

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            return SimpleNamespace(kind="model")

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = FakeAutoConfig
    fake_transformers.AutoProcessor = FakeAutoProcessor
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    text_processor, _model = script.load_hf_model("example/text-only-model")

    assert text_processor.kind == "tokenizer"
    assert calls["processor"] == []
    assert calls["tokenizer"] == ["example/text-only-model"]


def test_generate_response_uses_processor_template_and_parser() -> None:
    script = _load_script_module()

    class FakeInputIds:
        shape = (1, 3)

    class FakeInputs(dict):
        def __init__(self) -> None:
            super().__init__({"input_ids": FakeInputIds()})
            self.moved_to = None

        def to(self, device: str):
            self.moved_to = device
            return self

    class FakeGeneratedRow:
        def __getitem__(self, item):
            return ["generated-token-ids", item.start]

    class FakeOutput:
        def __getitem__(self, index: int):
            assert index == 0
            return FakeGeneratedRow()

    class FakeProcessor:
        eos_token_id = 99

        def __init__(self) -> None:
            self.template_kwargs = None
            self.decoded = None

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            enable_thinking: bool,
        ) -> str:
            self.template_kwargs = {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
            return "rendered prompt"

        def __call__(self, *, text: str, return_tensors: str):
            assert text == "rendered prompt"
            assert return_tensors == "pt"
            return FakeInputs()

        def decode(self, generated_ids, *, skip_special_tokens: bool) -> str:
            self.decoded = {
                "generated_ids": generated_ids,
                "skip_special_tokens": skip_special_tokens,
            }
            return "<raw response>"

        def parse_response(self, response: str) -> str:
            assert response == "<raw response>"
            return "parsed response"

    class FakeModel:
        device = "cuda:0"

        def __init__(self) -> None:
            self.generate_kwargs = None

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            return FakeOutput()

    processor = FakeProcessor()
    model = FakeModel()

    response = script.generate_response(
        processor,
        model,
        "How should I run a baseline?",
        max_new_tokens=24,
    )

    assert response == "parsed response"
    assert processor.template_kwargs["enable_thinking"] is False
    assert processor.decoded == {
        "generated_ids": ["generated-token-ids", 3],
        "skip_special_tokens": False,
    }
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["max_new_tokens"] == 24
    assert model.generate_kwargs["pad_token_id"] == 99
    assert model.generate_kwargs["input_ids"].shape == (1, 3)


def test_login_from_kaggle_secret_is_noop_outside_kaggle(monkeypatch) -> None:
    script = _load_script_module()
    monkeypatch.delitem(sys.modules, "kaggle_secrets", raising=False)

    assert script.login_from_kaggle_secret("HF_TOKEN") is False
    assert script.login_from_kaggle_secret("") is False
