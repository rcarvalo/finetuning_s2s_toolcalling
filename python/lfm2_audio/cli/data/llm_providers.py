"""Le fournisseur LLM d'un générateur, choisi par un drapeau.

Chaque fournisseur est importé à la demande : lancer Anthropic ne doit pas
exiger ``google-genai`` installé, ni l'inverse.
"""

from __future__ import annotations

from collections.abc import Iterator

from lfm2_audio.scorer.text.anthropic_judge import Effort
from lfm2_audio.scorer.text.judge import Judge

PROVIDERS = ("gemini", "anthropic")
DEFAULT_MODELS = {"gemini": "gemini-3.6-flash", "anthropic": "claude-opus-5"}
KEY_ENV = {"gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def make_judge(
    provider: str,
    model: str | None = None,
    *,
    max_usd: float | None = None,
    batch: bool = False,
    effort: Effort = "low",
) -> Judge:
    """Un juge prêt à l'emploi ; lève ``ValueError`` si les identifiants manquent."""
    if provider not in PROVIDERS:
        raise ValueError(f"fournisseur inconnu : {provider!r} (attendu : {', '.join(PROVIDERS)})")
    model_id = model or DEFAULT_MODELS[provider]
    judge: Judge
    if provider == "gemini":
        from lfm2_audio.scorer.text.gemini_judge import GeminiJudge

        judge = GeminiJudge(model_id)
        ready = judge.has_credentials
    elif provider == "anthropic":
        if batch:
            from lfm2_audio.scorer.text.anthropic_batch_judge import AnthropicBatchJudge

            judge = AnthropicBatchJudge(model_id, effort=effort, max_usd=max_usd)
            ready = judge.has_credentials
        else:
            from lfm2_audio.scorer.text.anthropic_judge import AnthropicJudge
            from lfm2_audio.scorer.text.llm_spend import SpendMeter

            judge = AnthropicJudge(model_id, effort=effort, meter=SpendMeter(model_id, max_usd=max_usd))
            ready = judge.has_credentials
    if not ready:
        raise ValueError(f"{KEY_ENV[provider]} absent")
    return judge


def judge_stream(judge: Judge, prompts: list[str]) -> Iterator[str]:
    """Les réponses dans l'ordre des prompts, par lot si le juge sait le faire."""
    judge_many = getattr(judge, "judge_many", None)
    if judge_many is not None:
        yield from judge_many(prompts)
        return
    for prompt in prompts:
        yield judge.judge(prompt)


def spend_line(judge: Judge) -> str | None:
    """La ligne ``===SPEND===`` d'un juge qui compte, sinon rien."""
    meter = getattr(judge, "meter", None)
    return None if meter is None else str(meter.summary())
