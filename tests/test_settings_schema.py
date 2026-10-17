"""settings_schema: every Config field has an entry, and the validators accept good and refuse hostile values.

Run: uv run python -m unittest discover -s tests -v
Scratch files are temp dirs named parley_test_schema_*.
"""
import dataclasses
import tempfile
import unittest
from pathlib import Path

from parley import config, settings_schema as ss


def config_paths() -> set[str]:
    out = set()
    for table in dataclasses.fields(config.Config):
        for f in dataclasses.fields(getattr(config.Config(), table.name)):
            out.add(f"{table.name}.{f.name}")
    return out


class CoverageTests(unittest.TestCase):
    def test_every_config_field_has_an_entry_and_vice_versa(self):
        self.assertEqual(config_paths() - set(ss.BY_PATH), set(), "Config field without a schema entry")
        self.assertEqual(set(ss.BY_PATH) - config_paths(), set(), "schema entry without a Config field")
        self.assertEqual(len(ss.SETTINGS), len(ss.BY_PATH), "duplicate path")

    def test_apply_modes_valid_and_restart_list_matches_plan(self):
        for s in ss.SETTINGS:
            self.assertIn(s.apply, ss.APPLY_MODES, s.path)
            self.assertTrue(s.label and s.help, s.path)
        self.assertEqual(ss.RESTART_PATHS, {"stt.model", "stt.device", "stt.compute_type", "stt.cpu_threads",
                                            "tts.device", "gpu.download_cuda"})
        self.assertEqual(ss.BY_PATH["updates.check"].apply, ss.NEXT_LAUNCH)

    def test_defaults_validate(self):
        cfg = config.Config()
        for path, s in ss.BY_PATH.items():
            table, key = path.split(".")
            value = getattr(getattr(cfg, table), key)
            if path == "tts.lang":
                value = "en-us"
            self.assertEqual(ss.validate(path, value), value, path)

    def test_numeric_and_choice_entries_have_limits(self):
        for s in ss.SETTINGS:
            if s.type in ("int", "float", "text"):
                self.assertIsNotNone(s.range, s.path)
            if s.type == "choice":
                self.assertTrue(s.choices, s.path)


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="parley_test_schema_")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def ok(self, path, value, expect=None):
        self.assertEqual(ss.validate(path, value), value if expect is None else expect, (path, value))

    def bad(self, path, *values):
        for v in values:
            with self.subTest(path=path, value=v):
                with self.assertRaises(ValueError):
                    ss.validate(path, v)

    def test_unknown_setting(self):
        with self.assertRaises(ValueError):
            ss.validate("llm.nope", 1)

    def test_enums(self):
        for b in ss.BACKENDS:
            self.ok("llm.backend", b)
        self.bad("llm.backend", "gpt", "", None, 1, "Builtin")
        self.ok("llm.preset", "tiny")
        self.bad("llm.preset", "huge")
        self.ok("llm.runtime", "vulkan")
        self.bad("llm.runtime", "metal", "cuda ")
        self.ok("stt.compute_type", "int8_float16")
        self.bad("stt.compute_type", "int4")
        self.bad("stt.device", "gpu", "dml")
        self.bad("tts.device", "gpu")
        self.bad("llm.effort", "max")

    def test_ranges(self):
        for path, lo, hi in [("audio.preroll_ms", 0, 2000), ("stt.beam_size", 1, 10), ("llm.history_turns", 0, 100), ("llm.gpu_layers", -1, 999), ("stt.cpu_threads", 0, 64)]:
            self.ok(path, lo)
            self.ok(path, hi)
            self.bad(path, lo - 1, hi + 1, 1.5, "3", True, None)
        for path, lo, hi in [("tts.speed", 0.5, 2.0), ("llm.temperature", 0.0, 2.0)]:
            self.ok(path, lo)
            self.ok(path, hi)
            self.ok(path, 1, 1.0)
            self.bad(path, lo - 0.01, hi + 0.01, float("nan"), float("inf"), "1", True)

    def test_num_ctx_is_zero_or_512_to_131072(self):
        for good in (0, 512, 8192, 131072):
            self.ok("llm.num_ctx", good)
        self.bad("llm.num_ctx", 1, 511, 131073, -1, 1.5, "0", True, None)

    def test_bools_are_strict(self):
        self.ok("audio.sounds", False)
        self.bad("audio.sounds", 1, 0, "true", None)

    def test_hotkey(self):
        self.ok("hotkey.key", "pause")
        self.ok("hotkey.key", " F13 ", "f13")
        self.ok("hotkey.key", "a")
        self.bad("hotkey.key", "", "nosuchkey", "mro", "__class__", "ctrl+x", "a b", "x" * 40, 5, None)

    def test_model_names(self):
        self.ok("llm.model_repo", "")
        self.ok("llm.model_repo", "unsloth/Qwen3.5-9B-GGUF")
        self.bad("llm.model_repo", "a", "../x", "a/../b", "a/b/c", "https://x/y", "a b/c", None)
        self.ok("llm.model_file", "")
        self.ok("llm.model_file", "Qwen3.5-9B-Q4_K_M.gguf")
        self.bad("llm.model_file", "..\\x.gguf", "../x.gguf", "C:\\x.gguf", "/x.gguf", "x.bin", "a/b.gguf", "NUL.gguf", ".gguf")

    def test_model_path(self):
        good = self.dir / "m.gguf"
        good.write_bytes(b"GGUF")
        txt = self.dir / "m.txt"
        txt.write_bytes(b"x")
        self.ok("llm.model_path", "")
        self.ok("llm.model_path", str(good))
        self.bad("llm.model_path", str(txt), str(self.dir / "missing.gguf"), str(self.dir), "x\0.gguf", 5)

    def test_base_url(self):
        self.ok("llm.base_url", "")
        self.ok("llm.base_url", "http://127.0.0.1:11434")
        self.ok("llm.base_url", "https://api.example.com/v1")
        self.bad("llm.base_url", "ftp://x/y", "file:///c:/x", "javascript:alert(1)", "//host/x", "http://", "localhost:8080",
                 "http://u:p@host/", "http://host:99999/", "http://ho st/", "x" * 400, 5)

    def test_system_prompt_length(self):
        self.ok("llm.system_prompt", "a" * 4000)
        self.bad("llm.system_prompt", "a" * 4001, None, 5)

    def test_voice_and_lang(self):
        self.ok("tts.voice", "af_heart")
        self.ok("tts.voice", "bm_george")
        self.bad("tts.voice", "", "../x", "af heart", "AF_HEART", "heart", "a" * 60)
        self.assertEqual(ss.voice_lang("af_heart"), "en-us")
        self.assertEqual(ss.voice_lang("bf_emma"), "en-gb")
        self.assertEqual(ss.voice_lang("jf_alpha"), "ja")
        self.assertIsNone(ss.voice_lang("qq_x"))
        self.ok("tts.lang", "en-gb")
        self.bad("tts.lang", "", "en_gb", "english", "en-gb-x")

    def test_free_text_fields(self):
        self.ok("stt.model", "auto")
        self.ok("stt.model", "distil-large-v3")
        self.bad("stt.model", "..\\x", "a b", "")
        self.ok("stt.language", "")
        self.ok("stt.language", "de")
        self.bad("stt.language", "english", "E1")
        self.ok("llm.keep_alive", "30m")
        self.ok("llm.keep_alive", "-1")
        self.bad("llm.keep_alive", "soon", "30 m")

    def test_audio_device(self):
        self.ok("audio.input_device", None)
        self.assertIsNone(ss.validate("audio.input_device", ""))
        self.assertIsNone(ss.validate("audio.output_device", ""))
        self.ok("audio.input_device", 2)
        self.ok("audio.input_device", "USB Microphone")
        self.bad("audio.input_device", " ", -1, True, 1.5, "x\n", "x" * 300)

    def test_api_key_error_does_not_echo_the_key(self):
        secret = "sk-ant-secret value"
        with self.assertRaises(ValueError) as cm:
            ss.validate("llm.api_key", secret)
        self.assertNotIn("secret", str(cm.exception))
        self.ok("llm.api_key", "sk-ant-abc123")
        self.ok("llm.api_key", "")
        self.ok("llm.api_key", "k" * 512)

    def test_api_key_shape(self):
        self.bad("llm.api_key", "k" * 513, "nul\0key", "bell\x07key", "caf\u00e9", "tab\tkey", "new\nline", None, 5)


if __name__ == "__main__":
    unittest.main()
