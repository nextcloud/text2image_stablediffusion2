from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


LIB_DIR = Path(__file__).resolve().parents[1] / "ex_app" / "lib"
sys.path.insert(0, str(LIB_DIR))

import compel_support


class _FakeCompelForSDXL:
    def __init__(self, pipe, device):
        self.pipe = pipe
        self.device = device


class _FakeCompel:
    def __init__(self, conditioning):
        self.conditioning = conditioning
        self.prompt = None

    def __call__(self, prompt):
        self.prompt = prompt
        return self.conditioning


class InitSdxlCompelTests(unittest.TestCase):
    def test_init_sdxl_compel_returns_none_when_compel_is_unavailable(self):
        with patch.object(compel_support, "CompelForSDXL", None):
            compel = compel_support.init_sdxl_compel(pipe=object(), device="cpu")

        self.assertIsNone(compel)

    def test_init_sdxl_compel_builds_compel_with_pipe_and_device(self):
        pipe = object()
        with patch.object(compel_support, "CompelForSDXL", _FakeCompelForSDXL):
            compel = compel_support.init_sdxl_compel(pipe=pipe, device="cuda")

        self.assertIsInstance(compel, _FakeCompelForSDXL)
        self.assertIs(compel.pipe, pipe)
        self.assertEqual(compel.device, "cuda")


class BuildPromptConditioningTests(unittest.TestCase):
    def test_build_prompt_conditioning_returns_none_without_compel(self):
        conditioning = compel_support.build_prompt_conditioning("hello world", None)

        self.assertIsNone(conditioning)

    def test_build_prompt_conditioning_returns_conditioning_with_compel(self):
        expected = types.SimpleNamespace(embeds="embeds", pooled_embeds="pooled")
        compel = _FakeCompel(expected)

        conditioning = compel_support.build_prompt_conditioning("hello world", compel)

        self.assertEqual(compel.prompt, "hello world")
        self.assertIs(conditioning, expected)
