"""``AiDiagMixin`` 单测：默认诊断 Provider 选择的回退路径（纯函数级 mock）。"""

from types import SimpleNamespace

from astrbot.core.provider.entities import ProviderType

from cost_control.ai_diag import AiDiagMixin


def _make_mixin(provider_insts: list, using=None) -> AiDiagMixin:
    """构造仅含 ``context.provider_manager`` 的 AiDiagMixin 实例。"""
    m = AiDiagMixin()
    m.context = SimpleNamespace(
        provider_manager=SimpleNamespace(
            provider_insts=provider_insts,
            get_using_provider=lambda **kwargs: using,
        )
    )
    return m


def _provider(pid: str, adapter: str, ptype: ProviderType) -> SimpleNamespace:
    return SimpleNamespace(meta=lambda: SimpleNamespace(id=pid, type=adapter, provider_type=ptype))


def test_default_provider_id_prefers_using_provider():
    m = _make_mixin(
        [_provider("p1", "openai_chat_completion", ProviderType.CHAT_COMPLETION)],
        using=_provider("using", "zhipu_chat_completion", ProviderType.CHAT_COMPLETION),
    )
    assert m._get_default_provider_id() == "using"


def test_default_provider_id_falls_back_to_first_chat_provider():
    # meta.type 是 adapter 名（openai_responses 等），永远不等于 "chat_completion"；
    # 回退须按 provider_type 枚举命中 chat provider 并跳过 TTS/STT 等。
    m = _make_mixin(
        [
            _provider("tts", "azure_tts", ProviderType.TEXT_TO_SPEECH),
            _provider("stt", "openai_whisper_api", ProviderType.SPEECH_TO_TEXT),
            _provider("resp", "openai_responses", ProviderType.CHAT_COMPLETION),
            _provider("chat", "openai_chat_completion", ProviderType.CHAT_COMPLETION),
        ],
        using=None,
    )
    assert m._get_default_provider_id() == "resp"


def test_default_provider_id_none_without_chat_provider():
    m = _make_mixin(
        [_provider("tts", "azure_tts", ProviderType.TEXT_TO_SPEECH)],
        using=None,
    )
    assert m._get_default_provider_id() is None
