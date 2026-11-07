"""Policy models, systems, checkpoints and generation.

The experiment runs on two model families so that findings are not one model's
quirk (see MODELS): Qwen3-1.7B is the main model, Gemma 3 1B the replication.
Each model gets its own folders: ckpt/<model>/, runs/p2/<model>/,
results/p2/<model>/outputs/ and <model>/ inside the Hub adapter repo.

A *system* is a name such as "base", "sft", "dpo" or "grpo-step625" (a GRPO
checkpoint). Every trained system is a LoRA adapter stored under <ckpt>/<name>/ together with
mtrlhf_meta.json {"base_model", "parent"}. Loading a system loads the base model
and merges the adapter chain (e.g. base -> sft -> dpo), so all methods share
exactly the same starting point.

Adapters are optionally mirrored to a private Hugging Face Hub repository, so a
stopped Kaggle/Colab session loses nothing: missing adapters are pulled back.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

DEFAULT_BASE = "Qwen/Qwen3-1.7B"

# Policy models. "systems" lists what each one trains (the replication model runs the
# systems the main claims rest on); "overrides" adapt batch sizes per training method
# (Gemma 3 has a 262k vocabulary and runs in fp32 on a T4, so its logits are large).
MODELS = {
    "qwen3-1.7b": {"model_id": "Qwen/Qwen3-1.7B", "role": "main",
                   "systems": ["base", "opus-mt", "sft", "sft+bo8", "rsft", "dpo", "grpo", "grpo_guarded"]},
    "gemma3-1b": {"model_id": "google/gemma-3-1b-it", "role": "replication",
                  "systems": ["base", "sft", "dpo", "grpo", "grpo_guarded"],
                  "overrides": {"sft": {"batch_size": 8, "grad_accum": 4},
                                "dpo": {"batch_size": 4, "grad_accum": 8},
                                "grpo": {"batch_size": 4, "grad_accum": 16, "epochs": 1}}},
    "qwen2.5-1.5b": {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "role": "earlier default",
                     "systems": ["base", "sft", "dpo", "grpo", "grpo_guarded"]},
}
# Models trained in bf16 whose activations overflow in fp16 (T4 GPUs have no bf16):
# they run in fp32 on such GPUs.
FP16_UNSAFE = ("gemma",)


def model_paths(tag: str | None) -> dict:
    """Model id and per-model folders. tag=None keeps the single-model layout."""
    if not tag:
        return {"tag": None, "model_id": DEFAULT_BASE, "ckpt": "ckpt", "runs": "runs/p2",
                "outputs": "results/p2/outputs", "model_data": "data/p2", "hub_prefix": "", "overrides": {},
                "systems": MODELS["qwen3-1.7b"]["systems"]}
    if tag not in MODELS:
        raise KeyError(f"Unknown model {tag!r}. Known: {', '.join(MODELS)}")
    m = MODELS[tag]
    return {"tag": tag, "model_id": m["model_id"], "ckpt": f"ckpt/{tag}", "runs": f"runs/p2/{tag}",
            "outputs": f"results/p2/{tag}/outputs", "model_data": f"data/p2/{tag}", "hub_prefix": tag,
            "overrides": m.get("overrides", {}), "systems": m["systems"]}
META = "mtrlhf_meta.json"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", META)

# Systems that are not LoRA adapters on the policy base model.
EXTERNAL = {
    "opus-mt": {"kind": "seq2seq", "model_id": "Helsinki-NLP/opus-mt-en-de", "pairs": {("en", "de")}},
    "qwen7b": {"kind": "causal", "model_id": "Qwen/Qwen2.5-7B-Instruct"},
}


# --- checkpoint store ----------------------------------------------------------------

class CheckpointStore:
    """Local adapter folders, optionally mirrored to a private Hugging Face Hub repo."""

    def __init__(self, root: str = "ckpt", hub_repo: str | None = None, hub_prefix: str = ""):
        self.root = root
        self.hub_repo = hub_repo
        self.hub_prefix = hub_prefix.strip("/")
        os.makedirs(root, exist_ok=True)

    def _remote(self, name: str) -> str:
        return f"{self.hub_prefix}/{name}" if self.hub_prefix else name

    @classmethod
    def from_env(cls, root: str = "ckpt", hub_repo: str | None = None, hub_prefix: str = "") -> "CheckpointStore":
        """Use `hub_repo` (or $MTRLHF_HUB_REPO). "auto" means <your HF user>/mtrlhf-en-de-adapters,
        created as a private repo if the token can write; otherwise local only."""
        hub_repo = hub_repo or os.environ.get("MTRLHF_HUB_REPO") or None
        if hub_repo == "none" or root.rstrip("/").endswith("_debug"):
            hub_repo = None  # debug stores never touch the shared adapter repo
        if hub_repo == "auto":
            try:
                from huggingface_hub import HfApi

                api = HfApi()
                hub_repo = f"{api.whoami()['name']}/mtrlhf-en-de-adapters"
                api.create_repo(hub_repo, private=True, exist_ok=True)
            except Exception as e:
                print(f"(Hub mirror disabled: {e.__class__.__name__}: {e}. "
                      "Use a Hugging Face token with write access to keep adapters across sessions.)")
                hub_repo = None
        return cls(root, hub_repo, hub_prefix)

    def path(self, name: str) -> str:
        return os.path.join(self.root, name)

    def exists_locally(self, name: str) -> bool:
        return os.path.exists(os.path.join(self.path(name), META))

    def ensure(self, name: str) -> str:
        """Local path of an adapter, pulling it from the Hub if needed."""
        if not self.exists_locally(name) and self.hub_repo:
            from huggingface_hub import snapshot_download

            remote = self._remote(name)
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    snapshot_download(self.hub_repo, allow_patterns=[f"{remote}/*"], local_dir=tmp)
                    if os.path.isdir(os.path.join(tmp, remote)):
                        shutil.copytree(os.path.join(tmp, remote), self.path(name), dirs_exist_ok=True)
            except Exception as e:
                print(f"(could not pull {remote} from {self.hub_repo}: {e})")
        if not self.exists_locally(name):
            raise FileNotFoundError(f"No adapter '{name}' in {self.root}" +
                                    (f" or {self.hub_repo}" if self.hub_repo else ""))
        return self.path(name)

    def has(self, name: str) -> bool:
        try:
            self.ensure(name)
            return True
        except FileNotFoundError:
            return False

    def save(self, name: str, adapter_dir: str, base_model: str, parent: str | None, extra: dict | None = None):
        """Copy adapter files from `adapter_dir` into the store, add metadata, push to the Hub."""
        dst = self.path(name)
        os.makedirs(dst, exist_ok=True)
        for f in ADAPTER_FILES:
            src = os.path.join(adapter_dir, f)
            if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(os.path.join(dst, f)):
                shutil.copy(src, dst)
        with open(os.path.join(dst, META), "w") as fh:
            json.dump({"base_model": base_model, "parent": parent, **(extra or {})}, fh, indent=2)
        self.push(name)
        return dst

    def push(self, name: str) -> None:
        if not self.hub_repo:
            return
        from huggingface_hub import upload_folder

        try:
            upload_folder(repo_id=self.hub_repo, folder_path=self.path(name), path_in_repo=self._remote(name),
                          commit_message=f"adapter {self._remote(name)}")
            print(f"pushed adapter {self._remote(name)} to {self.hub_repo}")
        except Exception as e:
            print(f"(push of {name} failed: {e})")

    def meta(self, name: str) -> dict:
        with open(os.path.join(self.ensure(name), META)) as fh:
            return json.load(fh)


# --- devices and loading -------------------------------------------------------------

def bf16_supported() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def fp16_safe(model_id: str) -> bool:
    return not any(k in model_id.lower() for k in FP16_UNSAFE)


def pick_dtype(device: str | None = None, model_id: str = DEFAULT_BASE):
    """bf16 on GPUs that have it; otherwise fp16, or fp32 for models that overflow in fp16."""
    import torch

    if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
        if bf16_supported():
            return torch.bfloat16
        return torch.float16 if fp16_safe(model_id) else torch.float32
    return torch.float32


def default_device(index: int = 0) -> str:
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{min(index, torch.cuda.device_count() - 1)}"


def neutral_generation_config(model, tokenizer) -> None:
    """Drop the model's sampling defaults (Qwen ships temperature 0.7, top_k 20,
    repetition_penalty 1.05) so that every generation setting is explicit."""
    from transformers import GenerationConfig

    eos = [tokenizer.eos_token_id]
    old = getattr(model, "generation_config", None)
    if old is not None and old.eos_token_id is not None:
        eos = old.eos_token_id if isinstance(old.eos_token_id, list) else [old.eos_token_id]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos[0]
    model.generation_config = GenerationConfig(eos_token_id=eos, pad_token_id=pad,
                                               bos_token_id=getattr(old, "bos_token_id", None))


def adapter_chain(name: str, store: CheckpointStore) -> tuple[str | None, list[str]]:
    """(base_model, [adapters from root to leaf]) for a system name."""
    chain, base, n = [], None, name
    while n and n != "base":
        meta = store.meta(n)
        chain.insert(0, n)
        base = meta["base_model"]
        n = meta.get("parent")
    return base, chain


NO_THINKING = "{%- set enable_thinking = false %}\n"


def disable_thinking(tokenizer) -> None:
    """Qwen3 'thinks' (<think>...</think>) unless enable_thinking=False is passed to the
    chat template. TRL's SFT and DPO trainers cannot pass it, so the template itself is
    changed to default to non-thinking mode. Training and inference then render the
    same prompt, ending in an empty <think></think> block, and the model answers directly."""
    tpl = getattr(tokenizer, "chat_template", None)
    if isinstance(tpl, str) and "enable_thinking" in tpl and not tpl.startswith(NO_THINKING):
        tokenizer.chat_template = NO_THINKING + tpl


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    disable_thinking(tok)
    return tok


def load_causal(name: str, store: CheckpointStore | None = None, base_model: str = DEFAULT_BASE,
                device: str | None = None, dtype=None):
    """Load a system as a plain (merged) causal LM. Returns (model, tokenizer, base_model)."""
    from transformers import AutoModelForCausalLM

    chain = []
    if name in EXTERNAL:
        base_model = EXTERNAL[name]["model_id"]
    elif name != "base":
        b, chain = adapter_chain(name, store)
        base_model = b or base_model
    device = device or default_device()
    dtype = dtype or pick_dtype(device, base_model)
    tok = load_tokenizer(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    for adapter in chain:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, store.ensure(adapter)).merge_and_unload()
    neutral_generation_config(model, tok)
    return model.to(device).eval(), tok, base_model


# --- generation ----------------------------------------------------------------------

class Translator:
    """Batched HF generation for a causal chat model or a seq2seq MT model."""

    def __init__(self, model, tokenizer, kind: str = "causal", batch_size: int = 32,
                 max_new_tokens: int = 256):
        self.model, self.tok, self.kind = model, tokenizer, kind
        self.batch_size, self.max_new_tokens = batch_size, max_new_tokens
        self.device = next(model.parameters()).device

    @classmethod
    def for_system(cls, name: str, store: CheckpointStore | None, base_model: str = DEFAULT_BASE,
                   device: str | None = None, **kw) -> "Translator":
        if name in EXTERNAL and EXTERNAL[name]["kind"] == "seq2seq":
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            mid = EXTERNAL[name]["model_id"]
            tok = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForSeq2SeqLM.from_pretrained(mid).to(device or default_device()).eval()
            return cls(model, tok, kind="seq2seq", **kw)
        model, tok, _ = load_causal(name, store, base_model, device)
        return cls(model, tok, kind="causal", **kw)

    def _inputs(self, sources, src_lang, tgt_lang):
        from mtrlhf.data.prompts import build_messages

        if self.kind == "seq2seq":
            return list(sources)
        return [self.tok.apply_chat_template(build_messages(s, src_lang, tgt_lang), tokenize=False,
                                             add_generation_prompt=True) for s in sources]

    def generate(self, sources: list[str], src_lang: str = "en", tgt_lang: str = "de", n: int = 1,
                 do_sample: bool = False, temperature: float = 1.0, top_p: float = 1.0,
                 seed: int = 0, max_new_tokens: int | None = None) -> list[list[str]]:
        """Return n raw replies per source (greedy when do_sample is False)."""
        import torch

        if n > 1 and not do_sample:
            raise ValueError("n > 1 needs do_sample=True (greedy decoding gives one output)")
        texts = self._inputs(sources, src_lang, tgt_lang)
        self.tok.padding_side = "left" if self.kind == "causal" else "right"
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        per_batch = max(1, self.batch_size // n)
        out: list[list[str]] = [[] for _ in texts]
        gen = dict(max_new_tokens=max_new_tokens or self.max_new_tokens, num_return_sequences=n,
                   do_sample=do_sample, pad_token_id=self.tok.pad_token_id)
        if do_sample:
            gen.update(temperature=temperature, top_p=top_p, top_k=0)
        torch.manual_seed(seed)
        with torch.inference_mode():
            for start in range(0, len(order), per_batch):
                idx = order[start:start + per_batch]
                enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                               add_special_tokens=self.kind == "seq2seq").to(self.device)
                enc.pop("token_type_ids", None)  # some tokenizers add it; decoder-only models reject it
                ids = self.model.generate(**enc, **gen)
                if self.kind == "causal":
                    ids = ids[:, enc["input_ids"].shape[1]:]
                dec = self.tok.batch_decode(ids, skip_special_tokens=True)
                for j, i in enumerate(idx):
                    out[i] = dec[j * n:(j + 1) * n]
        return out

    def close(self) -> None:
        import gc

        import torch

        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
