import unittest

from project6_qwen_pipeline.qwen_runner import _strip_qwen_thinking


class QwenRunnerTests(unittest.TestCase):
    def test_leaves_plain_json_unchanged(self):
        response = '{"category":"Chair"}'
        self.assertEqual(_strip_qwen_thinking(response), response)

    def test_strips_thinking_prefix(self):
        response = '<think>internal reasoning</think>\n{"category":"Chair"}'
        self.assertEqual(_strip_qwen_thinking(response).strip(), '{"category":"Chair"}')


if __name__ == "__main__":
    unittest.main()
