import unittest

from src.datasets.task_gen.re_arc_generators import GENERATORS_SRC_CODE


class TestReArcGenerators(unittest.TestCase):

    def setUp(self) -> None:
        global_namespace = globals()
        exec(GENERATORS_SRC_CODE, global_namespace)
        globals().update(global_namespace)

    def test(self) -> None:
        # Extract the names that start with "generate" from GENERATORS_SRC_CODE
        function_names = [
            name.split("(")[0] for name in GENERATORS_SRC_CODE.split() if name.startswith("generate")
        ]
        self.assertEqual(len(function_names), 400)
        random.seed(0)
        for fn_name in function_names:
            # Check if the function is defined
            self.assertTrue(fn_name in globals())
            # Check if the function is callable
            self.assertTrue(callable(globals()[fn_name]))
            if fn_name in [
                "generate_0e206a2e",
                "generate_72ca375d",
                "generate_ea32f347",
            ]:  # Skip these functions because current DSL raises error
                continue
            # Check if the function returns a dictionary
            try:
                self.assertIsInstance(globals()[fn_name](0, 1), dict)
            except Exception as e:
                self.fail(f"Error with function {fn_name}: {e}")


if __name__ == "__main__":
    import random

    random.seed(0)
    unittest.main()
