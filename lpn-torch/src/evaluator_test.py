import unittest

from src.models.lpn import LPN
from src.evaluator import Evaluator
from src.models.transformer import EncoderTransformer, DecoderTransformer
from src.models.utils import EncoderTransformerConfig, DecoderTransformerConfig


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        encoder = EncoderTransformer(EncoderTransformerConfig())
        decoder = DecoderTransformer(DecoderTransformerConfig())
        model = LPN(encoder=encoder, decoder=decoder)
        self.evaluator = Evaluator(model=model, inference_mode="mean", inference_mode_kwargs={})

    def test_evaluate_generations(self):
        generations = {
            "task_1": [{"attempt_1": [[0]], "attempt_2": [[1]]}],
            "task_2": [{"attempt_1": [[0]], "attempt_2": [[1, 1]]}],
            "task_3": [{"attempt_1": [[0]], "attempt_2": [[1, 1]]}],
            "task_4": [{"attempt_1": [[0]], "attempt_2": [[1, 1]]}],
        }
        solutions = {
            "task_1": [[[1]]],
            "task_2": [[[0]]],
            "task_3": [[[2]]],
            "task_4": [[[3, 3]]],
        }
        metrics = self.evaluator.evaluate_generations(generations, solutions)
        self.assertEqual(
            metrics,
            {
                "top_1_shape_accuracy": 0.75,
                "top_1_pixel_correctness": 0.25,
                "top_1_accuracy": 0.25,
                "top_2_shape_accuracy": 1.0,
                "top_2_pixel_correctness": 0.5,
                "top_2_accuracy": 0.5,
            },
        )


if __name__ == "__main__":
    unittest.main()
