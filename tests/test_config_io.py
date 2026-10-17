"""config_io (comment-preserving read-modify-write), config.load errors, api_key repr, model_file confinement.

Run: uv run python -m unittest discover -s tests -v
Scratch dirs are temp dirs named parley_test_configio_*; nothing touches the real Parley home.
"""
import dataclasses
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from parley import config, config_io, llama_server

SAMPLE = """# my notes
[hotkey]
key = "pause"  # hold to talk

[llm]
backend = "builtin"
system_prompt = \"\"\"line one
line two\"\"\"
"""


class TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="parley_test_configio_")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "config.toml"


class SaveChangesTests(TmpCase):
    def test_round_trip_keeps_comments_and_untouched_keys(self):
        self.path.write_text(SAMPLE, encoding="utf-8")
        config_io.save_changes(self.path, {"hotkey.key": "f13", "tts.speed": 1.25})
        text = self.path.read_text(encoding="utf-8")
        self.assertIn("# my notes", text)
        self.assertIn('key = "f13"  # hold to talk', text)
        self.assertIn("line one\nline two", text)
        cfg = config.load(self.path)
        self.assertEqual((cfg.hotkey.key, cfg.tts.speed, cfg.llm.backend), ("f13", 1.25, "builtin"))
        self.assertEqual(cfg.llm.system_prompt, "line one\nline two")

    def test_creates_missing_file_and_no_bak(self):
        sub = self.dir / "new" / "config.toml"
        config_io.save_changes(sub, {"stt.beam_size": 3})
        self.assertEqual(config.load(sub).stt.beam_size, 3)
        self.assertFalse(sub.with_name("config.toml.bak").exists())

    def test_bak_holds_previous_version(self):
        self.path.write_text(SAMPLE, encoding="utf-8")
        config_io.save_changes(self.path, {"hotkey.key": "f13"})
        self.assertEqual(self.path.with_name("config.toml.bak").read_text(encoding="utf-8"), SAMPLE)

    def test_none_removes_key(self):
        self.path.write_text('[audio]\ninput_device = "USB"\n', encoding="utf-8")
        config_io.save_changes(self.path, {"audio.input_device": None})
        self.assertIsNone(config.load(self.path).audio.input_device)

    def test_hostile_strings_stay_values(self):
        evil = 'x"\n[tools]\nenabled = false\n# \\'
        config_io.save_changes(self.path, {"llm.system_prompt": evil})
        cfg = config.load(self.path)
        self.assertEqual(cfg.llm.system_prompt, evil)
        self.assertTrue(cfg.tools.enabled)

    def test_untyped_value_and_bad_name_rejected(self):
        with self.assertRaises(TypeError):
            config_io.save_changes(self.path, {"llm.model": ["a"]})
        with self.assertRaises(ValueError):
            config_io.save_changes(self.path, {"llm.model\n[x]": "a"})
        self.assertFalse(self.path.exists())

    def test_outside_edit_between_load_and_save_is_preserved(self):
        self.path.write_text(SAMPLE, encoding="utf-8")
        config.load(self.path)
        with open(self.path, "a", encoding="utf-8") as f:  # an editor changes the file after our load
            f.write("\n[tts]\nvoice = \"bf_emma\"  # edited outside\n")
        config_io.save_changes(self.path, {"hotkey.key": "f13"})
        text = self.path.read_text(encoding="utf-8")
        self.assertIn('voice = "bf_emma"  # edited outside', text)
        self.assertEqual(config.load(self.path).hotkey.key, "f13")

    def test_corrupt_file_refused_and_untouched(self):
        bad = 'hotkey = [\nkey = "pause\n'
        self.path.write_text(bad, encoding="utf-8")
        with self.assertRaises(config_io.ConfigFileError) as cm:
            config_io.save_changes(self.path, {"hotkey.key": "f13"})
        self.assertIn("won't overwrite", str(cm.exception))
        self.assertEqual(self.path.read_text(encoding="utf-8"), bad)
        self.assertFalse(self.path.with_name("config.toml.bak").exists())

    def test_atomic_write_failure_keeps_original(self):
        self.path.write_text(SAMPLE, encoding="utf-8")
        with mock.patch("parley.config_io.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                config_io.save_changes(self.path, {"hotkey.key": "f13"})
        self.assertEqual(self.path.read_text(encoding="utf-8"), SAMPLE)
        self.assertEqual([p.name for p in self.dir.iterdir() if p.name.endswith(".tmp")], [])

    def test_no_bak_while_plaintext_key_present(self):
        secret = 'sk-test-not-a-real-key'
        self.path.write_text(f'[llm]\napi_key = "{secret}"\n', encoding="utf-8")
        config_io.save_changes(self.path, {"hotkey.key": "f13"})
        self.assertFalse(self.path.with_name("config.toml.bak").exists())
        # a key being added by this save counts too
        other = self.dir / "other.toml"
        other.write_text("[llm]\nbackend = \"openai\"\n", encoding="utf-8")
        config_io.save_changes(other, {"llm.api_key": secret})
        self.assertFalse(other.with_name("other.toml.bak").exists())
        # once the key is gone from the file, the next save may keep a backup
        config_io.save_changes(self.path, {"llm.api_key": ""})
        config_io.save_changes(self.path, {"hotkey.key": "f14"})
        self.assertTrue(self.path.with_name("config.toml.bak").exists())
        self.assertNotIn(secret, self.path.with_name("config.toml.bak").read_text(encoding="utf-8"))

    def test_crlf_file_survives(self):
        self.path.write_bytes(b'[hotkey]\r\nkey = "pause"\r\n')
        config_io.save_changes(self.path, {"hotkey.key": "f13"})
        self.assertEqual(self.path.read_bytes(), b'[hotkey]\r\nkey = "f13"\r\n')


class LoadErrorTests(TmpCase):
    def test_friendly_error_has_line_number_and_is_short(self):
        self.path.write_text('[hotkey]\nkey = "pause"\n[llm]\napi_key = sk-secret-value\n', encoding="utf-8")
        with self.assertRaises(SystemExit) as cm:
            config.load(self.path)
        msg = str(cm.exception)
        self.assertIn("line 4", msg)
        self.assertLessEqual(len(msg), 200)
        self.assertNotIn("sk-secret-value", msg)
        self.assertNotIn("Traceback", msg)


class ApiKeyReprTests(unittest.TestCase):
    def test_api_key_not_in_repr(self):
        cfg = config.Config()
        cfg.llm.api_key = "sk-ant-very-secret"
        self.assertNotIn("very-secret", repr(cfg))
        self.assertNotIn("api_key", repr(cfg.llm))
        self.assertEqual(cfg.llm.api_key, "sk-ant-very-secret")

    def test_api_key_field_still_a_config_field(self):
        self.assertIn("api_key", {f.name for f in dataclasses.fields(config.LlmConfig)})


class ModelConfinementTests(TmpCase):
    def _cfg(self, file, repo="owner/name"):
        return config.LlmConfig(model_repo=repo, model_file=file)

    def test_hostile_model_file_rejected_before_any_download(self):
        bad = ["..\\x.gguf", "../x.gguf", "sub/x.gguf", "sub\\x.gguf", "C:\\x.gguf", "C:x.gguf", "/etc/x.gguf",
               "\\\\server\\share\\x.gguf", "x.gguf\0", "x.txt", "", "CON.gguf", "nul.gguf", "x.gguf ", "..\\..\\evil.gguf"]
        for f in bad:
            with self.subTest(file=f), mock.patch("parley.llama_server.download") as dl:
                with self.assertRaises(SystemExit):
                    llama_server.ensure_model(self._cfg(f), self.dir)
                dl.assert_not_called()

    def test_hostile_model_repo_rejected(self):
        for r in ["../x", "a/../b", "a/b/c", "a", "https://evil/x", "a\\b", "./a", "a/.."]:
            with self.subTest(repo=r), mock.patch("parley.llama_server.download") as dl:
                with self.assertRaises(SystemExit):
                    llama_server.ensure_model(self._cfg("m.gguf", r), self.dir)
                dl.assert_not_called()

    def test_valid_names_download_inside_models_llm(self):
        with mock.patch("parley.llama_server.download") as dl:
            dest = llama_server.ensure_model(self._cfg("Qwen3.5-9B-Q4_K_M.gguf", "unsloth/Qwen3.5-9B-GGUF"), self.dir)
        self.assertEqual(dest, self.dir / "models" / "llm" / "Qwen3.5-9B-Q4_K_M.gguf")
        dl.assert_called_once()

    def test_resolved_path_check_catches_a_link_out_of_the_folder(self):
        base = self.dir / "models" / "llm"
        base.mkdir(parents=True)
        outside = self.dir / "outside"
        outside.mkdir()
        try:
            os.symlink(outside / "m.gguf", base / "m.gguf")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not permitted here")
        with mock.patch("parley.llama_server.download") as dl:
            with self.assertRaises(SystemExit):
                llama_server.ensure_model(self._cfg("m.gguf"), self.dir)
        dl.assert_not_called()


if __name__ == "__main__":
    unittest.main()
