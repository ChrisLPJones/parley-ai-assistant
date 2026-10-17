"""scripts/make_installer.py helpers and packaging/parley.iss (stdlib unittest; temp dirs only).

The compile test runs the real ISCC on a tiny fake tree and is skipped when Inno Setup is not installed.
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from parley import __version__, launcher

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("parley_make_installer", ROOT / "scripts" / "make_installer.py")
mk = importlib.util.module_from_spec(_spec)
sys.modules["parley_make_installer"] = mk
_spec.loader.exec_module(mk)

ISS = (ROOT / "packaging" / "parley.iss").read_text("utf-8")


def fake_tree(root: Path, version: str) -> Path:
    app = root / "app" / version
    app.mkdir(parents=True)
    (root / "Parley.exe").write_bytes(b"MZ")
    (app / launcher.CHILD_EXE).write_bytes(b"MZ")
    (app / launcher.INDEX_NAME).write_bytes(b"{}")
    launcher.write_pointer(root, launcher.Pointer(version), backup=False)
    return root


def iscc_or_none() -> Path | None:
    try:
        return mk.find_iscc()
    except mk.InstallerError:
        return None


class Helpers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="parley_test_installer_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_find_iscc_missing_gives_clear_error(self):
        with self.assertRaises(mk.InstallerError) as cm:
            mk.find_iscc({"PATH": str(self.root), "LOCALAPPDATA": str(self.root)})
        self.assertIn("Inno Setup", str(cm.exception))

    def test_find_iscc_per_user_default(self):
        exe = self.root / "Programs" / "Inno Setup 6" / "ISCC.exe"
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"")
        self.assertEqual(mk.find_iscc({"PATH": "", "LOCALAPPDATA": str(self.root)}), exe)

    def test_args_carry_version_and_paths(self):
        args = mk.iscc_args(Path("ISCC.exe"), "1.2.3", Path("src dir"), Path("out"), Path("x.iss"))
        self.assertIn("/DAppVersion=1.2.3", args)
        self.assertIn("/DSourceDir=src dir", args)
        self.assertEqual(args[-1], "x.iss")
        self.assertEqual(mk.installer_path(Path("out"), "1.2.3").name, "Parley-Setup-1.2.3.exe")

    def test_check_source_accepts_matching_build(self):
        mk.check_source(fake_tree(self.root, "1.2.3"), "1.2.3")

    def test_check_source_rejects_stale_or_incomplete_build(self):
        tree = fake_tree(self.root, "1.2.2")
        with self.assertRaises(mk.InstallerError):
            mk.check_source(tree, "1.2.3")  # pointer says another version
        (tree / "Parley.exe").unlink()
        with self.assertRaises(mk.InstallerError):
            mk.check_source(tree, "1.2.2")

    def test_check_source_rejects_bad_version_string(self):
        with self.assertRaises(mk.InstallerError):
            mk.check_source(self.root, "1.2.3\n#define X")


class Script(unittest.TestCase):
    def test_installer_policy(self):
        for needle in ("PrivilegesRequired=lowest", r"DefaultDirName={localappdata}\Programs\Parley",
                       "AppVersion={#AppVersion}", "OutputBaseFilename=Parley-Setup-{#AppVersion}",
                       "Compression=lzma2/max", "SolidCompression=yes", "skipifsilent", "ConfirmUninstall"):
            self.assertIn(needle, ISS)

    def test_signtool_hook_is_commented_out(self):
        lines = [ln for ln in ISS.splitlines() if "SignTool" in ln]
        self.assertTrue(lines)
        self.assertTrue(all(ln.lstrip().startswith(";") for ln in lines))

    def test_never_touches_the_parley_home(self):
        # Only {app} paths are ever deleted; the data folder is named only in the uninstall prompt.
        for ln in ISS.splitlines():
            if ln.startswith(("Type:", "Source:")):
                self.assertNotIn("parley\\", ln.replace("{app}", "").lower())
        self.assertIn("is NOT removed", ISS)

    @unittest.skipUnless(iscc_or_none(), "Inno Setup 6 not installed")
    def test_compiles_with_version_from_single_source(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_installer_") as d:
            tree = fake_tree(Path(d) / "install", __version__)
            exe = mk.make(tree, Path(d) / "out", __version__)
            self.assertEqual(exe.name, f"Parley-Setup-{__version__}.exe")
            self.assertGreater(exe.stat().st_size, 100_000)


if __name__ == "__main__":
    unittest.main()
