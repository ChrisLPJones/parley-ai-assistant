"""Hardware planning with synthetic adapter lists (stdlib unittest; no GPU needed).

Run: uv run python -m unittest discover -s tests -v
Intel cases are by design only: no Intel hardware was available to verify them.
"""
import unittest
from pathlib import Path
from unittest import mock

from parley import gpu, llama_server
from parley.config import LlmConfig

RTX = gpu.NvidiaGpu("NVIDIA GeForce RTX 5090", 32607, 617)
SMALL_RTX = gpu.NvidiaGpu("NVIDIA GeForce GTX 1650", 4096, 560)
RX = gpu.Adapter("AMD Radeon RX 7800 XT", "amd", 16368, 32000)
ARC = gpu.Adapter("Intel(R) Arc(TM) A770", "intel", 16000, 16000)  # unverified in reality
RADEON_IGPU = gpu.Adapter("AMD Radeon(TM) Graphics", "amd", 2021, 46783)
INTEL_IGPU = gpu.Adapter("Intel(R) UHD Graphics", "intel", 128, 16000)  # unverified in reality
NV_ADAPTER = gpu.Adapter(RTX.name, "nvidia", 32187, 46783)


class PlanTests(unittest.TestCase):
    def test_nvidia_only_uses_cuda_everywhere(self):
        p = gpu.plan([NV_ADAPTER], RTX)
        self.assertEqual((p.llm, p.tts, p.stt, p.preset), ("cuda", "cuda", "cuda", "small"))

    def test_small_nvidia_gets_tiny_preset(self):
        self.assertEqual(gpu.plan([], SMALL_RTX).preset, "tiny")

    def test_amd_discrete_uses_vulkan_llm_and_cpu_speech(self):
        p = gpu.plan([RX], None)
        self.assertEqual((p.llm, p.tts, p.stt, p.preset), ("vulkan", "cpu", "cpu", "small"))
        self.assertEqual(p.device, RX.name)

    def test_intel_discrete_follows_amd_path_unverified(self):
        p = gpu.plan([ARC], None)
        self.assertEqual((p.llm, p.tts, p.stt), ("vulkan", "cpu", "cpu"))

    def test_intel_igpu_only_is_all_cpu_unverified(self):
        p = gpu.plan([INTEL_IGPU], None)
        self.assertEqual((p.llm, p.tts, p.stt, p.preset), ("cpu", "cpu", "cpu", "tiny"))

    def test_amd_cpu_with_amd_igpu_is_all_cpu(self):
        p = gpu.plan([RADEON_IGPU], None)
        self.assertEqual((p.llm, p.tts, p.stt, p.preset), ("cpu", "cpu", "cpu", "tiny"))

    def test_no_gpu(self):
        p = gpu.plan([], None)
        self.assertEqual((p.llm, p.tts, p.stt, p.preset), ("cpu", "cpu", "cpu", "tiny"))

    def test_discrete_plus_igpu_picks_the_discrete_card(self):
        p = gpu.plan([RADEON_IGPU, RX], None)
        self.assertEqual((p.llm, p.device), ("vulkan", RX.name))

    def test_nvidia_wins_over_igpu(self):
        p = gpu.plan([NV_ADAPTER, RADEON_IGPU], RTX)
        self.assertEqual((p.llm, p.tts), ("cuda", "cuda"))

    def test_unknown_vendor_is_ignored(self):
        p = gpu.plan([gpu.Adapter("Mystery GPU", "other", 8192, 0)], None)
        self.assertEqual(p.llm, "cpu")

    def test_discrete_threshold_is_4gb(self):
        p = gpu.plan([gpu.Adapter("AMD Radeon RX 6500 XT", "amd", 4096, 0)], None)
        self.assertEqual((p.llm, p.preset), ("vulkan", "small"))
        p = gpu.plan([gpu.Adapter("Odd card", "amd", 3000, 0)], None)
        self.assertEqual(p.llm, "cpu")


class LlamaDeviceTests(unittest.TestCase):
    LISTING = (
        "ggml_vulkan: Found 2 Vulkan devices:\n"
        "Available devices:\n"
        "  Vulkan0: NVIDIA GeForce RTX 5090 (32109 MiB, 30000 MiB free)\n"
        "  Vulkan1: AMD Radeon(TM) Graphics (2048 MiB, 1900 MiB free)\n"
    )

    def test_parse(self):
        self.assertEqual(gpu.parse_llama_devices(self.LISTING),
                         [("Vulkan0", "NVIDIA GeForce RTX 5090"), ("Vulkan1", "AMD Radeon(TM) Graphics")])

    def test_pick_by_name(self):
        devices = gpu.parse_llama_devices(self.LISTING)
        self.assertEqual(gpu.pick_llama_device(devices, "AMD Radeon(TM) Graphics"), "Vulkan1")

    def test_unknown_or_garbage_omits_the_flag(self):
        self.assertIsNone(gpu.pick_llama_device(gpu.parse_llama_devices(self.LISTING), "Intel Arc"))
        self.assertEqual(gpu.parse_llama_devices("nothing useful here"), [])
        self.assertIsNone(gpu.pick_llama_device([], "AMD Radeon(TM) Graphics"))

    def test_two_identical_names_is_ambiguous(self):
        self.assertIsNone(gpu.pick_llama_device([("Vulkan0", "AMD Radeon"), ("Vulkan1", "AMD Radeon")], "AMD Radeon"))


class RuntimeTests(unittest.TestCase):
    def test_unknown_runtime_exits(self):
        with self.assertRaises(SystemExit):
            llama_server.resolve_runtime(LlmConfig(runtime="rocm"))

    def test_cuda_without_nvidia_exits(self):
        with mock.patch.object(gpu, "nvidia", return_value=None), self.assertRaises(SystemExit):
            llama_server.resolve_runtime(LlmConfig(runtime="cuda"))

    def test_forced_values_pass_through(self):
        with mock.patch.object(llama_server.sys, "platform", "win32"), mock.patch.object(gpu, "nvidia", return_value=RTX):
            for r in ("cuda", "vulkan", "cpu"):
                self.assertEqual(llama_server.resolve_runtime(LlmConfig(runtime=r)), r)

    def test_auto_follows_plan(self):
        with mock.patch.object(llama_server.sys, "platform", "win32"), \
                mock.patch.object(gpu, "nvidia", return_value=None), \
                mock.patch.object(gpu, "adapters", return_value=[RX]):
            self.assertEqual(llama_server.resolve_runtime(LlmConfig()), "vulkan")

    def test_cpu_and_vulkan_share_the_vulkan_download(self):
        with mock.patch.object(llama_server.sys, "platform", "win32"):
            self.assertEqual(llama_server._build_assets("cpu"), llama_server._build_assets("vulkan"))
            self.assertEqual(llama_server._build_assets("cpu")[0], "vulkan")


class FallbackTests(unittest.TestCase):
    """A GPU runtime that exits during startup is retried once on the CPU."""

    def test_gpu_failure_retries_on_cpu(self):
        launches = []

        def fake_launch(exe, model, cfg, runtime, log_path):
            launches.append(runtime)
            return mock.Mock(), "http://127.0.0.1:1"

        def fake_wait(proc, url, log_path):
            if launches[-1] != "cpu":
                raise RuntimeError("llama-server exited during startup:\n  vulkan boom")

        tmp = Path(__import__("tempfile").mkdtemp(prefix="parley_test_gpuplan_"))
        with mock.patch.object(llama_server, "resolve_runtime", return_value="vulkan"), \
                mock.patch.object(llama_server, "ensure_runtime", return_value=tmp / "llama-server.exe"), \
                mock.patch.object(llama_server, "ensure_model", return_value=tmp / "m.gguf"), \
                mock.patch.object(llama_server, "_launch", fake_launch), \
                mock.patch.object(llama_server, "_wait_ready", fake_wait):
            url, _ = llama_server.start(LlmConfig(), tmp)
        self.assertEqual(launches, ["vulkan", "cpu"])
        self.assertTrue(url.endswith("/v1"))

    def test_cpu_failure_is_raised(self):
        tmp = Path(__import__("tempfile").mkdtemp(prefix="parley_test_gpuplan_"))
        with mock.patch.object(llama_server, "resolve_runtime", return_value="cpu"), \
                mock.patch.object(llama_server, "ensure_runtime", return_value=tmp / "x"), \
                mock.patch.object(llama_server, "ensure_model", return_value=tmp / "m.gguf"), \
                mock.patch.object(llama_server, "_launch", return_value=(mock.Mock(), "u")), \
                mock.patch.object(llama_server, "_wait_ready", side_effect=RuntimeError("x\n  boom")):
            with self.assertRaises(RuntimeError):
                llama_server.start(LlmConfig(), tmp)


class StopHandleTests(unittest.TestCase):
    """start() returns a URL that llama_server.stop() can later shut down (used by a language model reload)."""

    def start(self, proc):
        tmp = Path(__import__("tempfile").mkdtemp(prefix="parley_test_gpuplan_"))
        with mock.patch.object(llama_server, "resolve_runtime", return_value="cpu"),                 mock.patch.object(llama_server, "ensure_runtime", return_value=tmp / "x"),                 mock.patch.object(llama_server, "ensure_model", return_value=tmp / "m.gguf"),                 mock.patch.object(llama_server, "_launch", return_value=(proc, "http://127.0.0.1:5")),                 mock.patch.object(llama_server, "_wait_ready"):
            url, _ = llama_server.start(LlmConfig(), tmp)
        return url

    def test_stop_terminates_the_server_that_start_returned_once(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        url = self.start(proc)
        self.assertTrue(llama_server.stop(url))
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once()
        self.assertFalse(llama_server.stop(url))  # unknown now: nothing happens twice
        self.assertFalse(llama_server.stop("http://127.0.0.1:99/v1"))
        self.assertFalse(llama_server.stop(None))

    def test_a_server_that_ignores_terminate_is_killed(self):
        import subprocess

        proc = mock.Mock()
        proc.poll.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired("llama-server", 10)
        self.assertTrue(llama_server.stop(self.start(proc)))
        proc.kill.assert_called_once()

    def test_an_already_exited_server_is_not_touched(self):
        proc = mock.Mock()
        proc.poll.return_value = 1
        self.assertFalse(llama_server.stop(self.start(proc)))
        proc.terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
