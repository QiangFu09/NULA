import json
from pathlib import Path
import contextlib
import io
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from nula import Net, NULAOptimizer, EMNULAOptimizer

def invoke(main, arguments):
    with patch.object(sys, "argv", ["test", *arguments]):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main()


class TrainingTests(unittest.TestCase):
    def test_network_output_and_truncation(self):
        network = Net(N=4, input_dim=8, cut_off=10)
        with torch.no_grad():
            network.fc2.weight.fill_(20)
        output = network(torch.zeros(3, 8))
        self.assertEqual(output.shape, (3,))
        self.assertTrue(torch.isfinite(output).all())
        self.assertLessEqual(network.fc2.weight.max().item(), 10)

    def test_seed_controls_optimizer_randomness(self):
        for cls, momentum in [(NULAOptimizer, True), (NULAOptimizer, False), (EMNULAOptimizer, True)]:
            values = []
            for _ in range(2):
                torch.manual_seed(19)
                parameter = torch.nn.Parameter(torch.ones(3))
                optimizer = cls([parameter], N=4, momentum=momentum, gamma=1, noise_scale=0.001)
                for _ in range(3):
                    parameter.grad = parameter.detach().clone()
                    optimizer.step()
                values.append(parameter.detach().clone())
            torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)

    def test_training_plotting_and_overwrite_protection(self):
        from train import main as train
        from plot import main as plot
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            command = ["--particles", "4", "--seeds", "0", "--epochs", "3", "--dimension", "8", "--samples", "4", "--output-dir", str(output)]
            invoke(train, command)
            for method in ["nula", "nla", "em"]:
                result = json.loads((output / f"{method}_n4_seed0.json").read_text())
                self.assertEqual(result["configuration"]["dimension"], 8)
                self.assertEqual(len(result["train_loss"]), 3)
            with self.assertRaises(SystemExit):
                invoke(train, command)
            for figure in [1, 2]:
                destination = Path(temporary) / f"figure{figure}.pdf"
                invoke(plot, [ "--figure", str(figure), "--particles", "4", "--particle-counts", "4", "--seeds", "0", "--input-dir", str(output), "--output", str(destination)])
                self.assertTrue(destination.is_file())


if __name__ == "__main__":
    unittest.main()
