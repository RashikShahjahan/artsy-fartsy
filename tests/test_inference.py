import unittest
from inference.server import parse_tools, template_messages


class ToolProtocolTest(unittest.TestCase):
    def test_literal_source_roundtrip(self):
        source = "from artcanvas import ArtCanvas\nprint('hi')"
        text, calls = parse_tools("<|tool_call_start|>[write_code(source=" + repr(source) + ")]<|tool_call_end|>")
        self.assertEqual(calls[0]["function"]["name"], "write_code")
        import json
        self.assertEqual(json.loads(calls[0]["function"]["arguments"])["source"], source)
        native = template_messages([{"role": "assistant", "content": None, "tool_calls": calls}])
        self.assertEqual(parse_tools(native[0]["content"])[1][0]["function"], calls[0]["function"])

    def test_parser_never_evaluates_expressions(self):
        with self.assertRaises(ValueError):
            parse_tools("<|tool_call_start|>[write_code(source=__import__('os').getcwd())]<|tool_call_end|>")

    def test_truncated_calls_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_tools("<|tool_call_start|>[submit()]")


if __name__ == "__main__":
    unittest.main()
