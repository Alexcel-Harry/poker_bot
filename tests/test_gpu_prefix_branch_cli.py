from contextlib import redirect_stderr
import io
import os
import subprocess
import sys
import unittest
from pathlib import Path

from examples.train_gpu_prefix_branch import (
    _require_single_cuda_gpu,
    build_parser,
    parse_hidden,
)


class GpuPrefixBranchCliTests(unittest.TestCase):
    def test_defaults_describe_a_cuda_tensorized_10k_run(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.gpus, "0")
        self.assertEqual(args.iterations, 10_000)
        self.assertEqual(args.parallel_hands, 1024)
        self.assertEqual(args.branch_width, 32)

    def test_cpu_and_no_gpu_are_rejected_by_the_parser_or_validator(self):
        parser = build_parser()
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            parser.parse_args(["--device", "cpu"])

        args = parser.parse_args(["--gpus", "none"])
        with self.assertRaises(ValueError):
            _require_single_cuda_gpu(args, {})

    def test_exactly_one_visible_gpu_is_configured_before_torch_import(self):
        args = build_parser().parse_args(["--gpus", "2"])
        environment: dict[str, str] = {}

        ids = _require_single_cuda_gpu(args, environment)

        self.assertEqual(ids, (2,))
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "2")

    def test_multiple_gpu_ids_are_rejected(self):
        args = build_parser().parse_args(["--gpus", "0,1"])
        with self.assertRaises(ValueError):
            _require_single_cuda_gpu(args, {})

    def test_hidden_parser_validates_widths(self):
        self.assertEqual(parse_hidden("64, 32"), (64, 32))
        with self.assertRaises(ValueError):
            parse_hidden("64,0")

    def test_script_help_does_not_initialize_cuda_or_start_training(self):
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-B", "examples/train_gpu_prefix_branch.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--parallel-hands", result.stdout)
        self.assertIn("CPU fallback is intentionally unavailable", result.stdout)


if __name__ == "__main__":
    unittest.main()
