"""Device lines and option sentences for the Advanced page, with fake hardware (nothing is probed).

Run: uv run python -m unittest discover -s tests -v
"""
import unittest

from parley import gpu, hardware_view as hv

NV = gpu.NvidiaGpu("NVIDIA GeForce RTX 5090", 32607, 570)
AMD = gpu.Adapter("AMD Radeon RX 7800 XT", "amd", 16384, 8192)
IGPU = gpu.Adapter("AMD Radeon(TM) Graphics", "amd", 512, 8192)


def facts(nvidia=None, *adapters):
    return hv.Facts(nvidia, list(adapters), "Intel(R) Core(TM) i7-12700H", 20)


class DeviceLineTests(unittest.TestCase):
    def test_nvidia_auto_cuda_and_cpu(self):
        f = facts(NV, gpu.Adapter(NV.name, "nvidia", 32607, 0))
        self.assertEqual(hv.device_line("stt", "auto", f), "Runs on: NVIDIA GeForce RTX 5090, 32 GB.")
        self.assertEqual(hv.device_line("tts", "cuda", f), "Runs on: NVIDIA GeForce RTX 5090, 32 GB.")
        self.assertEqual(hv.device_line("llm", "auto", f), "Runs on: NVIDIA GeForce RTX 5090, 32 GB.")
        self.assertEqual(hv.device_line("stt", "cpu", f), "Runs on: CPU: Intel(R) Core(TM) i7-12700H, 20 logical cores.")

    def test_cuda_without_nvidia_is_not_available_and_says_what_happens(self):
        f = facts(None)
        stt = hv.device_line("stt", "cuda", f)
        self.assertTrue(stt.startswith("Not available on this PC"))
        self.assertIn("CPU: Intel(R) Core(TM) i7-12700H", stt)  # speech falls back to the CPU
        self.assertIn("would not start", hv.device_line("llm", "cuda", f))  # the language model really refuses

    def test_amd_card_runs_the_llm_but_not_speech(self):
        f = facts(None, AMD)
        self.assertEqual(hv.device_line("llm", "auto", f), "Runs on: AMD Radeon RX 7800 XT, 16 GB.")
        self.assertEqual(hv.device_line("llm", "vulkan", f), "Runs on: AMD Radeon RX 7800 XT, 16 GB.")
        auto = hv.device_line("stt", "auto", f)
        self.assertIn("CPU: Intel", auto)
        self.assertIn("only use an NVIDIA graphics card", auto)

    def test_integrated_gpu_is_described_honestly(self):
        f = facts(None, IGPU)
        self.assertIn("CPU: Intel", hv.device_line("llm", "auto", f))
        self.assertIn("integrated", hv.device_line("llm", "vulkan", f))

    def test_vulkan_without_any_graphics_card(self):
        self.assertTrue(hv.device_line("llm", "vulkan", facts(None)).startswith("Not available on this PC"))

    def test_nothing_ever_mentions_drivers(self):
        for f in (facts(NV), facts(None, AMD), facts(None, IGPU), facts(None)):
            for stage in hv.STAGES:
                for choice in ("auto", "cuda", "vulkan", "cpu"):
                    self.assertNotIn("driver", hv.device_line(stage, choice, f).lower())
        for table in hv.OPTION_TEXT.values():
            for text in table.values():
                self.assertNotIn("driver", text.lower())

    def test_every_dropdown_option_has_a_sentence(self):
        from parley import settings_schema as S

        for path, table in hv.OPTION_TEXT.items():
            self.assertEqual(set(table), set(S.BY_PATH[path].choices), path)
        self.assertEqual(set(hv.STAGE_OF), {"stt.device", "tts.device", "llm.runtime"})


if __name__ == "__main__":
    unittest.main()
