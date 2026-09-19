from types import SimpleNamespace

from openjev.prompter import PromptFormat, render_prompt
from openjev.schemas import QuestionNoul


def test_instruct_skips_chat_template() -> None:
    """Явный instruct не должен вызывать apply_chat_template (System One)."""
    tokenizer = SimpleNamespace(
        chat_template="{{ messages }}",
        apply_chat_template=lambda *args, **kwargs: "SHOULD_NOT_USE_CHAT_TEMPLATE",
    )
    question = QuestionNoul(
        type="noul",
        instructions="Is it urgent?",
    )
    prompt = render_prompt(
        "Hello",
        question,
        tokenizer=tokenizer,
        prompt_format=PromptFormat.INSTRUCT,
    )
    assert "SHOULD_NOT_USE_CHAT_TEMPLATE" not in prompt
    assert "### Instruction:" in prompt
    assert "Select single option letter:" in prompt
