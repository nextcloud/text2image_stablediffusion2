"""Tests for SDXL long-prompt encoding."""

import unittest

import torch

from ex_app.lib.prompt_encoding import encode_prompt


class _FakeTokenizer:
    def __init__(self, model_max_length=6, extra_token=None):
        self.model_max_length = model_max_length
        self.bos_token_id = 101
        self.eos_token_id = 102
        self.pad_token_id = 0
        self.extra_token = extra_token

    def __call__(self, prompt, truncation=False, **kwargs):
        input_ids = [ord(character) for character in prompt]
        if self.extra_token is not None:
            input_ids.append(self.extra_token)
        if truncation:
            input_ids = input_ids[: self.model_max_length]
        return {"input_ids": input_ids}

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def prepare_for_model(self, token_ids, max_length, **kwargs):
        prepared = [self.bos_token_id, *token_ids, self.eos_token_id]
        prepared.extend([self.pad_token_id] * (max_length - len(prepared)))
        return {"input_ids": prepared}


class _EncoderOutput:
    def __init__(self, hidden_states, pooled_output):
        self.hidden_states = hidden_states
        self.pooled_output = pooled_output

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.pooled_output


class _FakeTextEncoder:
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size
        self.dtype = torch.float32
        self.input_ids = []
        self.call_count = 0

    def __call__(self, input_ids, output_hidden_states):
        self.call_count += 1
        self.input_ids.extend(input_ids.detach().cpu().tolist())
        hidden_state = input_ids.to(torch.float32).unsqueeze(-1)
        hidden_state = hidden_state.repeat(1, 1, self.embedding_size)
        return _EncoderOutput(
            hidden_states=(hidden_state - 1, hidden_state, hidden_state + 1),
            pooled_output=hidden_state.sum(dim=1),
        )


class _FakePipeline:
    def __init__(self, tokenizer_2=None):
        self._execution_device = torch.device("cpu")
        self.tokenizer = _FakeTokenizer()
        self.tokenizer_2 = tokenizer_2 if tokenizer_2 is not None else _FakeTokenizer()
        self.text_encoder = _FakeTextEncoder(embedding_size=2)
        self.text_encoder_2 = _FakeTextEncoder(embedding_size=3)


class EncodeSdxlPromptTests(unittest.TestCase):
    def test_short_prompt_uses_one_chunk_for_both_encoders(self):
        pipe = _FakePipeline()

        prompt_embeds, pooled_prompt_embeds = encode_prompt(pipe, "ab")

        expected_tokens = [101, ord("a"), ord("b"), 102, 0, 0]
        self.assertEqual(pipe.text_encoder.input_ids, [expected_tokens])
        self.assertEqual(pipe.text_encoder_2.input_ids, [expected_tokens])
        self.assertEqual(prompt_embeds.shape, (1, 6, 5))
        self.assertEqual(pooled_prompt_embeds.shape, (1, 3))

    def test_long_prompt_encodes_every_token_across_multiple_chunks(self):
        pipe = _FakePipeline()

        prompt_embeds, _ = encode_prompt(pipe, "abcdef")

        self.assertEqual(
            pipe.text_encoder.input_ids,
            [
                [101, ord("a"), ord("b"), ord("c"), ord("d"), 102],
                [101, ord("e"), ord("f"), 102, 0, 0],
            ],
        )
        self.assertEqual(pipe.text_encoder.call_count, 2)
        self.assertEqual(pipe.text_encoder_2.call_count, 2)
        self.assertEqual(prompt_embeds.shape, (1, 12, 5))

    def test_shorter_tokenization_is_padded_with_an_empty_chunk(self):
        pipe = _FakePipeline(tokenizer_2=_FakeTokenizer(extra_token=999))

        prompt_embeds, _ = encode_prompt(pipe, "abcd")

        self.assertEqual(
            pipe.text_encoder.input_ids,
            [
                [101, ord("a"), ord("b"), ord("c"), ord("d"), 102],
                [101, 102, 0, 0, 0, 0],
            ],
        )
        self.assertEqual(len(pipe.text_encoder_2.input_ids), 2)
        self.assertEqual(prompt_embeds.shape, (1, 12, 5))

    def test_pooled_embeddings_come_from_the_first_chunk(self):
        pipe = _FakePipeline()

        _, pooled_prompt_embeds = encode_prompt(pipe, "abcdef")

        first_chunk = torch.tensor(pipe.text_encoder_2.input_ids[0])
        expected = first_chunk.sum().repeat(3).reshape(1, 3).to(torch.float32)
        torch.testing.assert_close(pooled_prompt_embeds, expected)
