# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests: leading content newlines after the reasoning end tag.

The Qwen3 chat template renders assistant history as ``<think>{reasoning}

{content}``.
The model emits the ``

`` separator itself right after ``</think>``; with
``strip_leading_content_newlines=True`` (the qwen3 default) those newlines
must not surface as the first content delta, and a degenerate
whitespace-only reply must yield content is None.
"""

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import simulate_reasoning_streaming
from vllm.parser.qwen3 import Qwen3Parser, qwen3_config

_VOCAB = {
    "<think>": 50,
    "</think>": 51,
    "<tool_call>": 60,
    "</tool_call>": 61,
}


def _parser(strip: bool = True) -> Qwen3Parser:
    cfg = qwen3_config(strip_leading_content_newlines=strip)
    return Qwen3Parser(make_mock_tokenizer(_VOCAB), parser_engine_config=cfg)


class TestNonStreaming:
    def test_separator_newlines_stripped(self):
        parser = _parser()
        reasoning, content = parser.extract_reasoning(
            "<think>Reasoning.</think>\n\nAnswer.",
            None,
        )
        assert reasoning == "Reasoning."
        assert content == "Answer."

    def test_internal_content_newlines_preserved(self):
        parser = _parser()
        reasoning, content = parser.extract_reasoning(
            "<think>Step 1.</think>\n\nPara 1.\n\nPara 2.", None
        )
        assert reasoning == "Step 1."
        assert content == "Para 1.\n\nPara 2."

    def test_whitespace_only_content_becomes_none(self):
        parser = _parser()
        reasoning, content = parser.extract_reasoning(
            "<think>Still thinking...</think>\n\n", None
        )
        assert reasoning == "Still thinking..."
        assert content is None


class TestStreaming:
    def test_separator_stripped_from_first_content_delta(self):
        reasoning, content = simulate_reasoning_streaming(
            _parser(),
            ["Let me analyze.", "</think>", "\n\n", "The answer"],
        )
        assert reasoning == "Let me analyze."
        assert content == "The answer"

    def test_separator_arrives_in_reasoning_delta(self):
        reasoning, content = simulate_reasoning_streaming(
            _parser(),
            ["thinking.\n", "</think>\n\n", "done"],
        )
        # qwen3 config keeps trailing reasoning whitespace.
        assert reasoning == "thinking.\n"
        assert content == "done"

    def test_internal_newlines_preserved(self):
        reasoning, content = simulate_reasoning_streaming(
            _parser(),
            ["I think.", "</think>", "\n\n", "Para 1.\n\n", "Para 2."],
        )
        assert reasoning == "I think."
        assert content == "Para 1.\n\nPara 2."

    def test_whitespace_only_reply_yields_no_content(self):
        """Degenerate think-end + separator + EOS reply: no phantom content.

        The ``\\n\\n`` after ``</think>`` is reasoning-tail whitespace in this
        config (trailing-whitespace stripping is off for qwen3), so there are
        no content deltas at all — never a phantom ``"\\n\\n"`` content chunk.
        """
        reasoning, content = simulate_reasoning_streaming(
            _parser(),
            ["thinking.\n", "</think>", "\n\n"],
        )
        assert reasoning == "thinking.\n"
        assert content == ""

    def test_tool_call_after_think_end_unaffected(self):
        """Whitespace-only content before a tool call stays dropped."""
        reasoning, content = simulate_reasoning_streaming(
            _parser(),
            ["Let me check.", "</think>", "\n\n", "<tool_call>", "<function=test>"],
        )
        assert reasoning == "Let me check."
        assert content == ""  # helper joins zero content deltas; nothing leaked


class TestStripDisabled:
    def test_disabled_keeps_separator(self):
        parser = _parser(strip=False)
        reasoning, content = parser.extract_reasoning(
            "<think>Reasoning.</think>\n\nAnswer.", None
        )
        assert reasoning == "Reasoning."
        assert content == "\n\nAnswer."
