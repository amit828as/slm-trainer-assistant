import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_script_module() -> types.ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_kaggle_baseline.py"
    spec = importlib.util.spec_from_file_location("run_kaggle_baseline_under_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_eval_file(path: Path, *, eval_id: str, media: bool = False) -> None:
    payload = {
        "id": eval_id,
        "category": "debugging",
        "difficulty": "beginner",
        "question": f"What should I do for {eval_id}?",
        "expected_traits": ["answers carefully"],
        "anti_traits": [],
    }
    if media:
        payload["media"] = [
            {
                "type": "image",
                "path": "evals/media/debugging_loss_quality.png",
                "description": "Synthetic chart.",
            }
        ]
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _install_fake_cuda(monkeypatch) -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda index: "Fake GPU",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_load_hf_model_uses_processor_for_gemma4(monkeypatch) -> None:
    script = _load_script_module()
    _install_fake_cuda(monkeypatch)
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

    class FakeAutoModelForMultimodalLM:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            raise AssertionError("multimodal loader should not be used for text evals")

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = FakeAutoConfig
    fake_transformers.AutoProcessor = FakeAutoProcessor
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    fake_transformers.AutoModelForMultimodalLM = FakeAutoModelForMultimodalLM
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
    _install_fake_cuda(monkeypatch)
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

    class FakeAutoModelForMultimodalLM:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            raise AssertionError("multimodal loader should not be used for text evals")

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = FakeAutoConfig
    fake_transformers.AutoProcessor = FakeAutoProcessor
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    fake_transformers.AutoModelForMultimodalLM = FakeAutoModelForMultimodalLM
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    text_processor, _model = script.load_hf_model("example/text-only-model")

    assert text_processor.kind == "tokenizer"
    assert calls["processor"] == []
    assert calls["tokenizer"] == ["example/text-only-model"]


def test_load_hf_model_uses_multimodal_model_when_media_is_present(monkeypatch) -> None:
    script = _load_script_module()
    _install_fake_cuda(monkeypatch)
    calls: dict[str, list[tuple[str, dict]]] = {
        "processor": [],
        "causal": [],
        "multimodal": [],
    }

    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            return SimpleNamespace(model_type="gemma4", architectures=[])

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            calls["processor"].append((model_name, kwargs))
            return SimpleNamespace(kind="processor")

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            raise AssertionError("tokenizer should not be used for multimodal evals")

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            calls["causal"].append((model_name, kwargs))
            return SimpleNamespace(kind="causal")

    class FakeAutoModelForMultimodalLM:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            calls["multimodal"].append((model_name, kwargs))
            return SimpleNamespace(kind="multimodal")

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = FakeAutoConfig
    fake_transformers.AutoProcessor = FakeAutoProcessor
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    fake_transformers.AutoModelForMultimodalLM = FakeAutoModelForMultimodalLM
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    text_processor, model = script.load_hf_model("google/gemma-4-E4B-it", use_multimodal=True)

    assert text_processor.kind == "processor"
    assert model.kind == "multimodal"
    assert calls["processor"] == [
        (
            "google/gemma-4-E4B-it",
            {"padding_side": "left", "trust_remote_code": True},
        )
    ]
    assert calls["causal"] == []
    assert calls["multimodal"] == [
        (
            "google/gemma-4-E4B-it",
            {"device_map": "auto", "dtype": "auto", "trust_remote_code": True},
        )
    ]


def test_load_hf_model_fails_when_cuda_is_unavailable(monkeypatch) -> None:
    script = _load_script_module()

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(SystemExit, match="CUDA is not available"):
        script._assert_cuda_available()


def test_load_hf_model_fails_when_model_offloads_to_cpu_or_disk() -> None:
    script = _load_script_module()
    model = SimpleNamespace(
        hf_device_map={
            "model.embed_tokens": 0,
            "model.layers.0": "cuda:0",
            "model.layers.1": "cpu",
            "model.layers.2": "disk",
        }
    )

    with pytest.raises(RuntimeError, match="partially offloaded to CPU/disk"):
        script.assert_model_not_cpu_offloaded(model)


def test_load_hf_model_fails_when_model_device_is_cpu() -> None:
    script = _load_script_module()
    model = SimpleNamespace(device="cpu")

    with pytest.raises(RuntimeError, match="loaded on CPU"):
        script.assert_model_not_cpu_offloaded(model)


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


def test_generate_response_uses_multimodal_template_for_image_media(monkeypatch) -> None:
    script = _load_script_module()
    monkeypatch.setattr(script, "_load_image", lambda media: f"image:{media.path}")

    class FakeInputIds:
        shape = (1, 4)

    class FakeInputs(dict):
        def __init__(self) -> None:
            super().__init__({"input_ids": FakeInputIds()})
            self.to_args = None
            self.to_kwargs = None

        def to(self, *args, **kwargs):
            self.to_args = args
            self.to_kwargs = kwargs
            return self

    class FakeGeneratedRow:
        def __getitem__(self, item):
            return ["generated-token-ids", item.start]

    class FakeOutput:
        def __getitem__(self, index: int):
            assert index == 0
            return FakeGeneratedRow()

    class FakeProcessor:
        def __init__(self) -> None:
            self.messages = None
            self.template_kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.template_kwargs = kwargs
            return FakeInputs()

        def decode(self, generated_ids, *, skip_special_tokens: bool) -> str:
            assert generated_ids == ["generated-token-ids", 4]
            assert skip_special_tokens is True
            return "image response"

    class FakeModel:
        device = "cuda:0"
        dtype = "bfloat16"

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
        "What does this chart show?",
        media=[
            SimpleNamespace(
                type="image",
                path="evals/media/debugging_loss_curve.png",
            )
        ],
        max_new_tokens=32,
    )

    assert response == "image response"
    assert processor.template_kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "enable_thinking": False,
    }
    assert processor.messages[1]["content"][0] == {
        "type": "image",
        "image": "image:evals/media/debugging_loss_curve.png",
    }
    assert processor.messages[1]["content"][1] == {
        "type": "text",
        "text": "What does this chart show?",
    }
    assert model.generate_kwargs["max_new_tokens"] == 32
    assert model.generate_kwargs["input_ids"].shape == (1, 4)


def test_login_from_kaggle_secret_is_noop_outside_kaggle(monkeypatch) -> None:
    script = _load_script_module()
    monkeypatch.delitem(sys.modules, "kaggle_secrets", raising=False)

    assert script.login_from_kaggle_secret("HF_TOKEN") is False
    assert script.login_from_kaggle_secret("") is False


def test_run_kaggle_baselines_skips_existing_and_reuses_model_groups(
    monkeypatch,
    tmp_path: Path,
) -> None:
    script = _load_script_module()
    eval_dir = tmp_path / "evals" / "golden"
    output_dir = tmp_path / "evals" / "reports"
    eval_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    _write_eval_file(eval_dir / "beginner_questions.jsonl", eval_id="beginner-001")
    _write_eval_file(eval_dir / "debugging_questions.jsonl", eval_id="debugging-001")
    _write_eval_file(
        eval_dir / "proactive_risk_detection_questions.jsonl",
        eval_id="proactive-001",
    )
    _write_eval_file(
        eval_dir / "multimodal_image_questions.jsonl",
        eval_id="image-001",
        media=True,
    )
    skipped_report = output_dir / "debugging_gemma4_e4b_it.json"
    skipped_report.write_text('{"already": "there"}\n', encoding="utf-8")
    load_calls: list[bool] = []

    def fake_login(secret_name: str | None) -> bool:
        assert secret_name == "HF_TOKEN"
        return True

    def fake_load_hf_model(model_name: str, *, use_multimodal: bool = False):
        assert model_name == "google/gemma-4-E4B-it"
        load_calls.append(use_multimodal)
        return (
            SimpleNamespace(kind=f"processor:{use_multimodal}"),
            SimpleNamespace(kind=f"model:{use_multimodal}"),
        )

    def fake_generate_response(processor, model, question: str, *, media=None, max_new_tokens: int):
        assert max_new_tokens == 123
        return f"{processor.kind}|{model.kind}|media={bool(media)}|{question}"

    monkeypatch.setattr(script, "login_from_kaggle_secret", fake_login)
    monkeypatch.setattr(script, "load_hf_model", fake_load_hf_model)
    monkeypatch.setattr(script, "generate_response", fake_generate_response)

    report_paths = script.run_kaggle_baselines(
        eval_dir,
        output_dir,
        model_name="google/gemma-4-E4B-it",
        max_new_tokens=123,
        hf_token_secret="HF_TOKEN",
        report_suffix="gemma4_e4b_it",
        skip_existing=True,
    )

    assert load_calls == [False, True]
    assert report_paths == [
        output_dir / "beginner_gemma4_e4b_it.json",
        output_dir / "proactive_gemma4_e4b_it.json",
        output_dir / "multimodal_image_gemma4_e4b_it.json",
    ]
    assert skipped_report.read_text(encoding="utf-8") == '{"already": "there"}\n'
    beginner_payload = json.loads(report_paths[0].read_text(encoding="utf-8"))
    proactive_payload = json.loads(report_paths[1].read_text(encoding="utf-8"))
    image_payload = json.loads(report_paths[2].read_text(encoding="utf-8"))
    assert beginner_payload["metadata"]["hf_login"] == "kaggle_secret"
    assert beginner_payload["metadata"]["model"] == "google/gemma-4-E4B-it"
    assert beginner_payload["metadata"]["max_new_tokens"] == 123
    assert beginner_payload["metadata"]["multimodal"] is False
    assert proactive_payload["results"][0]["eval_id"] == "proactive-001"
    assert image_payload["metadata"]["multimodal"] is True
    assert beginner_payload["results"][0]["response"].startswith(
        "processor:False|model:False|media=False"
    )
    assert image_payload["results"][0]["response"].startswith(
        "processor:True|model:True|media=True"
    )


def test_run_kaggle_baselines_can_skip_multimodal_files(monkeypatch, tmp_path: Path) -> None:
    script = _load_script_module()
    eval_dir = tmp_path / "evals" / "golden"
    output_dir = tmp_path / "evals" / "reports"
    eval_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    _write_eval_file(eval_dir / "beginner_questions.jsonl", eval_id="beginner-001")
    _write_eval_file(
        eval_dir / "multimodal_image_questions.jsonl",
        eval_id="image-001",
        media=True,
    )
    load_calls: list[bool] = []

    monkeypatch.setattr(script, "login_from_kaggle_secret", lambda secret_name: False)
    monkeypatch.setattr(
        script,
        "load_hf_model",
        lambda model_name, *, use_multimodal=False: (
            load_calls.append(use_multimodal) or SimpleNamespace(kind="processor"),
            SimpleNamespace(kind="model"),
        ),
    )
    monkeypatch.setattr(
        script,
        "generate_response",
        lambda processor, model, question, *, media=None, max_new_tokens: "response",
    )

    report_paths = script.run_kaggle_baselines(
        eval_dir,
        output_dir,
        model_name="google/gemma-4-E4B-it",
        max_new_tokens=123,
        text_only=True,
    )

    assert load_calls == [False]
    assert report_paths == [output_dir / "beginner_gemma_4_e4b_it.json"]


def test_run_kaggle_baselines_can_auto_version_output_dir(monkeypatch, tmp_path: Path) -> None:
    script = _load_script_module()
    eval_dir = tmp_path / "evals" / "golden"
    reports_dir = tmp_path / "evals" / "reports"
    eval_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)
    _write_eval_file(eval_dir / "beginner_questions.jsonl", eval_id="beginner-001")
    (reports_dir / "beginner_gemma4_e4b_it.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(script, "login_from_kaggle_secret", lambda secret_name: False)
    monkeypatch.setattr(
        script,
        "load_hf_model",
        lambda model_name, *, use_multimodal=False: (
            SimpleNamespace(kind="processor"),
            SimpleNamespace(kind="model"),
        ),
    )
    monkeypatch.setattr(
        script,
        "generate_response",
        lambda processor, model, question, *, media=None, max_new_tokens: "response",
    )

    report_paths = script.run_kaggle_baselines(
        eval_dir,
        reports_dir / "baseline",
        model_name="google/gemma-4-E4B-it",
        max_new_tokens=123,
        auto_version_output_dir=True,
    )

    assert report_paths == [reports_dir / "baseline_v2" / "beginner_gemma_4_e4b_it.json"]


def test_system_prompt_keeps_words_separated_and_encourages_adaptive_depth() -> None:
    script = _load_script_module()

    assert "most likely risk" in script.SYSTEM_PROMPT
    assert "do not be shallow" in script.SYSTEM_PROMPT
    assert "Use adaptive depth:" in script.SYSTEM_PROMPT
    assert "Do not treat training loss" in script.SYSTEM_PROMPT
    assert "mostlikely" not in script.SYSTEM_PROMPT


def test_parse_args_accepts_batch_mode(monkeypatch, tmp_path: Path) -> None:
    script = _load_script_module()
    eval_dir = tmp_path / "evals" / "golden"
    output_dir = tmp_path / "evals" / "reports"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_kaggle_baseline.py",
            "--eval-dir",
            str(eval_dir),
            "--output-dir",
            str(output_dir),
            "--model",
            "google/gemma-4-E4B-it",
            "--max-new-tokens",
            "256",
            "--report-suffix",
            "gemma4_e4b_it",
            "--skip-existing",
        ],
    )

    args = script.parse_args()

    assert args.eval_dir == eval_dir
    assert args.output_dir == output_dir
    assert args.eval_file is None
    assert args.output is None
    assert args.model == "google/gemma-4-E4B-it"
    assert args.max_new_tokens == 256
    assert args.report_suffix == "gemma4_e4b_it"
    assert args.skip_existing is True
