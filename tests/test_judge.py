import pytest

from experience_learning.judge import OpenAICompatibleSemanticJudge
from experience_learning.types import Verdict


def test_judge_parser_accepts_constrained_json() -> None:
    result = OpenAICompatibleSemanticJudge._parse_response(
        '{"verdict":"EQUIVALENT","confidence":0.9,"rationale":"same transition"}'
    )
    assert result.verdict is Verdict.EQUIVALENT
    assert result.confidence == 0.9


@pytest.mark.parametrize(
    "response",
    [
        "EQUIVALENT",
        "```json\n{\"verdict\":\"EQUIVALENT\"}\n```",
        '{"verdict":"CORRECT","confidence":1.0,"rationale":"alias"}',
        '{"verdict":"EQUIVALENT"}',
        '{"verdict":"EQUIVALENT","confidence":2.0,"rationale":"bad"}',
    ],
)
def test_judge_parser_rejects_malformed_or_ambiguous_output(response: str) -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleSemanticJudge._parse_response(response)
