"""Keep portable releases independent of local dependencies and working data."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.main_eval import package


class PackageTests(unittest.TestCase):
    def test_source_selection_keeps_audit_app_and_excludes_local_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            included = {
                "construction/audit/interface/package.json",
                "construction/audit/interface/package-lock.json",
                "construction/audit/interface/.gitignore",
                "construction/audit/interface/public/favicon.svg",
                "construction/audit/interface/src/App.jsx",
                "construction/audit/interface/src/styles.css",
                "construction/audit/interface/server/index.js",
                "construction/audit/interface/scripts/freeze-sample.js",
                "construction/audit/interface/tests/audit.test.js",
                "construction/audit/frozen/main150-v1/sample.json",
                "construction/audit/analysis/analyze.mjs",
            }
            excluded = {
                "construction/audit/interface/node_modules/example/package.json",
                "construction/audit/interface/dist/index.html",
                "construction/audit/interface/.vite/metadata.json",
                "construction/audit/responses/main150-v1/A1.submitted.json",
                "construction/audit/analysis/summary.json",
                "construction/audit/analysis/report.html",
                "construction/audit/A1.draft.json",
                "construction/audit/outputs/request.json",
                "construction/audit/__pycache__/module.pyc",
            }
            for name in included | excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")
            with patch.object(package, "ROOT", root):
                selected = {p.relative_to(root).as_posix()
                            for p in package.source_files(root / "construction/audit")}
            self.assertEqual(selected, included)


if __name__ == "__main__":
    unittest.main()
