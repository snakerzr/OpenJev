import pytest

from openjev.decoding import (
    LOGPROB_FLOOR,
    max_logprob_for_letter,
    pack_answer,
    softmax_from_logprobs,
)
from openjev.errors import EngineError
from openjev.schemas import QuestionNoul, QuestionScore


def test_max_logprob_alias_matching() -> None:
    top_logprobs = {"A": -2.5, " A": -0.8, "B": -1.2}
    assert max_logprob_for_letter(top_logprobs, "A") == -0.8
    assert max_logprob_for_letter(top_logprobs, "B") == -1.2
    assert max_logprob_for_letter(top_logprobs, "C") == LOGPROB_FLOOR


def test_softmax_all_floor_raises_error() -> None:
    all_floors = [LOGPROB_FLOOR, LOGPROB_FLOOR, LOGPROB_FLOOR]
    with pytest.raises(EngineError, match="Prediction degraded"):
        softmax_from_logprobs(all_floors, temperature=1.0, strict_floor_check=True)


def test_score_expected_value_math() -> None:
    question = QuestionScore(
        type="score",
        instructions="Rate",
        criteria=["Low", "Med", "High"],
    )
    labels = ("0", "1", "2")
    probs = [0.1, 0.2, 0.7]
    ans = pack_answer(question, labels, probs)

    assert ans.type == "score"
    assert ans.score == 2
    assert ans.expected == 1.6


def test_noul_true_mapping() -> None:
    question = QuestionNoul(
        type="noul",
        instructions="Is urgent?",
        criteria={"true": "Yes", "false": "No"},
    )
    labels = ("true", "false")
    probs = [0.82, 0.18]
    ans = pack_answer(question, labels, probs)

    assert ans.type == "noul"
    assert ans.noul == 0.82
    assert ans.confidence == 0.82
