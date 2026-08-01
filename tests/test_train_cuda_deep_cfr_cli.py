from contextlib import redirect_stderr
import io
import os
import subprocess
import sys
import unittest
from pathlib import Path

from examples.train_cuda_deep_cfr import build_parser, require_single_gpu, run_from_args


class TrainCudaDeepCFRCliTests(unittest.TestCase):
    def test_defaults_describe_level_synchronous_three_player_training(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.traversals_per_player, 4096)
        self.assertEqual(args.parallel_traversals, 256)
        self.assertEqual(args.starting_stack, 200)
        self.assertEqual(args.seats, 3)
        self.assertEqual(args.snapshot_every, 0)

        with self.assertRaises(ValueError):
            run_from_args(build_parser().parse_args(["--seats", "2"]), {})
        with self.assertRaises(ValueError):
            run_from_args(build_parser().parse_args(["--snapshot-every", "-1"]), {})

    def test_exactly_one_gpu_is_required(self):
        args = build_parser().parse_args(["--gpus", "2"])
        environment: dict[str, str] = {}

        self.assertEqual(require_single_gpu(args, environment), (2,))
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "2")
        with self.assertRaises(ValueError):
            require_single_gpu(build_parser().parse_args(["--gpus", "0,1"]), {})
        with self.assertRaises(ValueError):
            require_single_gpu(build_parser().parse_args(["--gpus", "none"]), {})

    def test_help_does_not_initialize_cuda(self):
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-B", "examples/train_cuda_deep_cfr.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("level-synchronous", result.stdout)


if __name__ == "__main__":
    unittest.main()
