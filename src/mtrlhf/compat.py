"""Workarounds for version clashes with packages preinstalled on Kaggle / Colab."""

from __future__ import annotations

import importlib

_PEFT_MODULES_WITH_TORCHAO_CHECK = ("peft.import_utils", "peft.tuners.lora.torchao", "peft.utils.quantization_utils")


def ignore_incompatible_torchao() -> bool:
    """Let PEFT treat an incompatible torchao as absent.

    PEFT 0.21 raises ImportError whenever it creates LoRA layers and finds a torchao
    older than 0.16, even when torchao is never used. Kaggle ships torchao 0.10, so
    training and adapter loading failed there. This project does not use torchao, so
    in that case PEFT is told torchao is unavailable. Returns True if it patched PEFT.
    """
    try:
        from peft import import_utils
    except ImportError:
        return False
    try:
        import_utils.is_torchao_available()
        return False
    except ImportError as e:
        if "torchao" not in str(e):
            raise

    def torchao_unavailable() -> bool:
        return False

    for name in _PEFT_MODULES_WITH_TORCHAO_CHECK:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(module, "is_torchao_available"):
            module.is_torchao_available = torchao_unavailable
    print("(compat: PEFT rejects the installed torchao version; torchao is not used here, so it is ignored)")
    return True
