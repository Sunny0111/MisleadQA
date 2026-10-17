"""Exercise Slurm wrappers with local command doubles; no jobs or models are run."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from experiments.main_eval.protocol import ROOT

def find_bash():
    if os.environ.get("BASH_EXECUTABLE"):
        return shutil.which(os.environ["BASH_EXECUTABLE"])
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            candidate = Path(git).resolve().parent.parent / "bin/bash.exe"
            if candidate.is_file():
                return str(candidate)
        return None
    return shutil.which("bash")


BASH = find_bash()


def shell_path(path):
    value = Path(path).resolve().as_posix()
    return "/" + value[0].lower() + value[2:] if os.name == "nt" else value


@unittest.skipUnless(BASH and Path(BASH).is_file(), "Bash is required for shell integration tests")
class SlurmTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.scripts = self.root / "experiments/option_order"
        self.scripts.mkdir(parents=True)
        for name in ("slurm.sh", "submit-slurm.sh"):
            shutil.copyfile(ROOT / "experiments/option_order" / name, self.scripts / name)
        self.trace = self.root / "trace.txt"
        self.environment = dict(os.environ, TRACE_FILE=shell_path(self.trace),
            SLURM_JOB_ID="1234", SLURM_ARRAY_TASK_ID="5", SLURM_SUBMIT_DIR=shell_path(self.root),
            CUDA_VISIBLE_DEVICES="3", RUN_PREFIX="test-order")
        self.environment.pop("FAIL_PREFLIGHT", None)
        self.command("sbatch", 'printf "%s\\n" "$*" >> "$TRACE_FILE"\nprintf "2468\\n"\n')
        self.command("srun", 'shift\nexec "$@"\n')
        python = self.command("fake-python", 'printf "%s\\n" "$CUDA_VISIBLE_DEVICES|$PWD|$*" >> "$TRACE_FILE"\n'
            'if [[ "${FAIL_PREFLIGHT:-0}" == 1 && "$*" == *preflight.py* ]]; then exit 7; fi\n')
        self.environment.update(TF_PYTHON=shell_path(python), VLLM_PYTHON=shell_path(python))

    def command(self, name, body):
        path = self.bin / name
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body, encoding="utf-8", newline="\n")
        path.chmod(0o755)
        return path

    def invoke(self, script, *arguments):
        return subprocess.run([BASH, "--noprofile", "--norc", "-c",
            'export PATH="$1:$PATH"; shift; exec bash "$@"', "test", shell_path(self.bin),
            shell_path(script), *arguments], cwd=self.root.parent, env=self.environment,
            capture_output=True, text=True, timeout=30)

    def test_submission_creates_logs_and_preserves_cli_overrides(self):
        result = self.invoke(self.scripts / "submit-slurm.sh", "check", "--account=test", "--time=00:20:00")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "2468")
        self.assertTrue((self.root / "logs").is_dir())
        arguments = self.trace.read_text(encoding="utf-8")
        self.assertIn("--array=0,5%1", arguments)
        self.assertIn("--account=test", arguments)
        self.assertIn("--chdir=" + shell_path(self.root), arguments)
        self.assertIn("experiments/option_order/slurm.sh check", arguments)

    def test_copied_batch_script_keeps_gpu_assignment_and_runs_all_orders(self):
        copied = self.root / "slurm_spool_script"
        shutil.copyfile(self.scripts / "slurm.sh", copied)
        result = self.invoke(copied, "full")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.trace.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(line.startswith("3|" + shell_path(self.root) + "|") for line in calls))
        self.assertIn("--backend vllm", calls[1])
        self.assertIn("--rerun-original --run-id test-order-smoke-fresh-v1 --smoke", calls[2])
        self.assertIn("--rerun-original --run-id test-order-full-fresh-v1", calls[3])
        self.assertIn("OPTION ORDER full COMPLETE: qwen3-32b-awq", result.stdout)

    def test_failed_environment_check_prevents_model_inference(self):
        self.environment["FAIL_PREFLIGHT"] = "1"
        result = self.invoke(self.scripts / "slurm.sh", "full")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(len(self.trace.read_text(encoding="utf-8").splitlines()), 2)
        self.assertNotIn("OPTION ORDER full COMPLETE", result.stdout)

    def test_default_environment_is_relative_to_submitted_project(self):
        self.environment.pop("TF_PYTHON")
        self.environment.pop("VLLM_PYTHON")
        python = self.root / ".venv-vllm/bin/python"
        python.parent.mkdir(parents=True)
        shutil.copyfile(self.bin / "fake-python", python)
        python.chmod(0o755)
        copied = self.root / "slurm_spool_script"
        shutil.copyfile(self.scripts / "slurm.sh", copied)
        result = self.invoke(copied, "check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Python=.venv-vllm/bin/python", result.stdout)
        self.assertIn("OPTION ORDER CHECK COMPLETE: vllm", result.stdout)


if __name__ == "__main__":
    unittest.main()
