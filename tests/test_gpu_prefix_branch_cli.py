import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from examples.train_gpu_prefix_branch import build_parser, parse_gpu_ids


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class GpuPrefixBranchCliTests(unittest.TestCase):
    def test_gpu_parser_accepts_cpu_single_gpu_and_multi_gpu(self):
        self.assertEqual(parse_gpu_ids("none"), ())
        self.assertEqual(parse_gpu_ids("0"), (0,))
        self.assertEqual(parse_gpu_ids("0,1"), (0, 1))

    def test_build_parser_accepts_model_output(self):
        parser = build_parser()
        args = parser.parse_args(["--device", "cpu", "--gpus", "none", "--model-out", "runs/policy.pt"])

        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.gpus, "none")
        self.assertEqual(args.model_out, Path("runs/policy.pt"))

    def test_script_help_runs_when_called_by_path(self):
        result = subprocess.run(
            [sys.executable, "-B", "examples/train_gpu_prefix_branch.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model-out", result.stdout)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_tiny_gpu_training_cli_writes_pt_and_summary(self):
        from examples.train_gpu_prefix_branch import run_from_args
        from poker_arena.bots import TorchPolicyBot

        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_out = Path(tmpdir) / "policy.pt"
            summary_out = Path(tmpdir) / "summary.json"
            args = parser.parse_args(
                [
                    "--iterations",
                    "1",
                    "--branch-width",
                    "2",
                    "--branch-depth",
                    "1",
                    "--integer-action-budget",
                    "4",
                    "--max-actions-per-episode",
                    "2",
                    "--starting-stack",
                    "200",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--device",
                    "cpu",
                    "--gpus",
                    "none",
                    "--model-out",
                    str(model_out),
                    "--summary-out",
                    str(summary_out),
                ]
            )

            summary = run_from_args(args, env={})

            self.assertTrue(model_out.exists())
            self.assertTrue(summary_out.exists())
            self.assertEqual(summary["model_out"], str(model_out))
            self.assertGreater(summary["training_samples"], 0)
            self.assertIsNotNone(TorchPolicyBot.from_checkpoint(model_out, device="cpu"))


if __name__ == "__main__":
    unittest.main()
