"""Synthetic binary fixtures for the bounded GGUF header inspector."""
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1] / "tools" / "inspect_gguf.py"
SPEC = importlib.util.spec_from_file_location("inspect_gguf", TOOL)
gguf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gguf)


def number(code, value):
    return struct.pack("<" + code, value)


def string(value):
    value = value.encode() if isinstance(value, str) else value
    return number("Q", len(value)) + value


def metadata(key, kind, payload):
    return string(key) + number("I", kind) + payload


def tensor(name, dimensions, kind, offset):
    return (string(name) + number("I", len(dimensions))
            + b"".join(number("Q", d) for d in dimensions)
            + number("I", kind) + number("Q", offset))


def fixture(entries=(), tensors=(), payload_bytes=256, alignment=32):
    header = (b"GGUF" + number("I", 3) + number("Q", len(tensors))
              + number("Q", len(entries)) + b"".join(entries) + b"".join(tensors))
    return header + b"\0" * (-len(header) % alignment) + b"\0" * payload_bytes


class InspectGGUFTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "model.gguf"

    def inspect(self, data, **kwargs):
        self.path.write_bytes(data)
        return gguf.inspect_gguf(self.path, **kwargs)

    def test_geometry_and_mixed_tensor_types_and_cli(self):
        data = fixture([
            metadata("general.architecture", 8, string("qwen3moe")),
            metadata("general.name", 8, string("Synthetic Sage")),
            metadata("general.file_type", 4, number("I", 15)),
            metadata("qwen3moe.embedding_length", 4, number("I", 2048)),
            metadata("qwen3moe.attention.head_count", 4, number("I", 32)),
            metadata("qwen3moe.expert_count", 4, number("I", 128)),
            metadata("tokenizer.ggml.tokens", 9,
                     number("I", 8) + number("Q", 3) + string("a") + string("b") + string("c")),
        ], [tensor("token_embd.weight", [2048, 151936], 12, 0),
            tensor("blk.0.ffn_down_exps.weight", [768, 2048, 128], 14, 32),
            tensor("output_norm.weight", [2048], 0, 64),
            tensor("blk.0.attn_norm.weight", [2048], 0, 96)])
        report = self.inspect(data, examples_per_type=1)
        self.assertEqual(report["architecture"], "qwen3moe")
        self.assertEqual(report["file_type_name"], "Q4_K_M")
        self.assertEqual(report["geometry"]["qwen3moe.expert_count"], 128)
        self.assertEqual(report["tensor_types"], {"F32": 2, "Q4_K": 1, "Q6_K": 1})
        self.assertEqual(report["representative_tensors"]["Q6_K"][0]["dimensions"], [768, 2048, 128])
        self.assertEqual(len(report["representative_tensors"]["F32"]), 1)
        self.assertEqual(report["array_summaries"]["tokenizer.ggml.tokens"],
                         {"element_type": "STRING", "count": 3})
        self.assertEqual(report["tensor_data_offset"] % 32, 0)
        self.assertEqual(report["model_bytes"], len(data))
        self.assertIn("unverified", report["validation"])
        run = subprocess.run([sys.executable, str(TOOL), str(self.path)],
                             text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(run.stdout)["tensor_types"], report["tensor_types"])

    def test_iq4_xs_recipe_is_distinct_from_tensor_type(self):
        data = fixture([
            metadata("general.architecture", 8, string("qwen35moe")),
            metadata("general.file_type", 4, number("I", 30)),
        ], [tensor("blk.0.ffn_gate_exps.weight", [2048, 512, 256], 23, 0)])
        report = self.inspect(data)
        self.assertEqual(report["file_type_name"], "IQ4_XS")
        self.assertEqual(report["tensor_types"], {"IQ4_XS": 1})
        self.assertEqual(report["representative_tensors"]["IQ4_XS"][0]["type_id"], 23)

    def test_all_scalar_types_and_nested_arrays(self):
        entries = [metadata("general.architecture", 8, string("test"))]
        scalar_values = [(0, "B", 255), (1, "b", -1), (2, "H", 65535),
                         (3, "h", -10), (4, "I", 42), (5, "i", -42),
                         (6, "f", 1.5), (7, "B", 1), (10, "Q", 2**40),
                         (11, "q", -(2**40)), (12, "d", 2.5)]
        for kind, code, value in scalar_values:
            entries.append(metadata(f"test.value_{kind}", kind, number(code, value)))
        nested = (number("I", 9) + number("Q", 1)
                  + number("I", 4) + number("Q", 2) + number("I", 1) + number("I", 2))
        entries.append(metadata("test.nested", 9, nested))
        report = self.inspect(fixture(entries))
        for kind, _, value in scalar_values:
            self.assertEqual(report["geometry"][f"test.value_{kind}"], value)
        self.assertEqual(report["array_summaries"]["test.nested"]["element_type"], "ARRAY")

    def test_numeric_arrays_are_skipped_without_materialization(self):
        payload = number("I", 4) + number("Q", 100_000) + b"\0" * 400_000
        report = self.inspect(fixture([metadata("tokenizer.ggml.ids", 9, payload)]))
        self.assertEqual(report["array_summaries"]["tokenizer.ggml.ids"]["count"], 100_000)
        self.assertLess(len(json.dumps(report)), 1500)

    def test_prefix_requires_explicit_model_size_and_all_descriptors(self):
        full = fixture([], [tensor("x", [32], 12, 64)])
        report = self.inspect(full)
        prefix = full[:report["header_bytes"]]
        with self.assertRaises(gguf.GGUFError):
            self.inspect(prefix)
        result = self.inspect(prefix, model_size=len(full))
        self.assertEqual(result["model_size_source"], "caller_supplied")
        self.assertEqual(result["local_bytes"], len(prefix))
        with self.assertRaisesRegex(gguf.GGUFError, "truncated"):
            self.inspect(prefix[:-1], model_size=len(full))

    def test_truncation_at_every_header_byte(self):
        full = fixture([metadata("general.architecture", 8, string("test")),
                        metadata("tokenizer.tokens", 9, number("I", 8) + number("Q", 1) + string("abc"))],
                       [tensor("x", [32, 64], 12, 0)])
        header_bytes = self.inspect(full)["header_bytes"]
        for position in range(header_bytes):
            with self.subTest(position=position), self.assertRaises(gguf.GGUFError):
                self.inspect(full[:position], model_size=len(full))

    def test_invalid_magic_version_types_counts_and_strings(self):
        cases = [
            (b"NOPE" + b"\0" * 20, "magic"),
            (b"GGUF" + number("I", 2) + b"\0" * 16, "version"),
            (b"GGUF" + number("I", 3) + number("Q", 1_000_001) + number("Q", 0), "count"),
            (b"GGUF" + number("I", 3) + number("Q", 0) + number("Q", 100_001), "count"),
            (fixture([metadata("x", 999, b"x")]), "unknown metadata"),
            (fixture([metadata("x", 9, number("I", 999) + number("Q", 0))]), "array element"),
            (fixture([metadata("x", 9, number("I", 4) + number("Q", 2**63))]), "item budget"),
            (fixture([metadata("x", 8, number("Q", 2**63))]), "string length"),
            (fixture([metadata("x", 8, string(b"\xff"))]), "UTF-8"),
            (fixture([metadata("x", 7, number("B", 2))]), "boolean"),
            (fixture([metadata("x", 6, number("f", float("nan")))]), "non-finite"),
        ]
        for data, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(gguf.GGUFError, message):
                self.inspect(data)

    def test_array_count_cannot_seek_past_eof_or_budget(self):
        data = fixture([metadata("x", 9, number("I", 12) + number("Q", 100))], payload_bytes=0)
        with self.assertRaisesRegex(gguf.GGUFError, "truncated"):
            self.inspect(data)
        data = fixture([metadata("x", 9, number("I", 4) + number("Q", 100) + b"\0" * 400)])
        with self.assertRaisesRegex(gguf.GGUFError, "byte budget"):
            self.inspect(data, max_header_bytes=100)
        entries = [metadata(str(i), 9, number("I", 0) + number("Q", 6) + b"\0" * 6) for i in range(2)]
        with self.assertRaisesRegex(gguf.GGUFError, "item budget"):
            self.inspect(fixture(entries), max_array_items=10)

    def test_nested_arrays_have_depth_limit(self):
        value = number("I", 0) + number("Q", 0)
        for _ in range(8):
            value = number("I", 9) + number("Q", 1) + value
        with self.assertRaisesRegex(gguf.GGUFError, "nesting"):
            self.inspect(fixture([metadata("x", 9, value)]))

    def test_invalid_tensor_descriptors_alignment_duplicates_and_offsets(self):
        cases = [
            (fixture([], [tensor("x", [], 0, 0)]), "dimension count"),
            (fixture([], [tensor("x", [1] * 5, 0, 0)]), "dimension count"),
            (fixture([], [tensor("x", [0], 0, 0)]), "nonzero"),
            (fixture([], [tensor("x", [1], 0, 1)]), "unaligned"),
            (fixture([], [tensor("x", [1], 0, 256)]), "payload bounds"),
            (fixture([], [tensor("x", [1], 0, 2**63)]), "payload bounds"),
            (fixture([], [tensor("x", [1], 0, 0), tensor("x", [1], 0, 32)]), "duplicate tensor"),
            (fixture([metadata("x", 0, b"\0"), metadata("x", 0, b"\0")]), "duplicate metadata"),
            (fixture([metadata("general.alignment", 4, number("I", 3))]), "power of two"),
            (fixture([metadata("general.alignment", 7, number("B", 1))]), "power of two"),
            (fixture([metadata("general.architecture", 4, number("I", 1))]), "must be a string"),
        ]
        for data, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(gguf.GGUFError, message):
                self.inspect(data)

    def test_unknown_tensor_type_is_reported_without_invented_payload_validation(self):
        report = self.inspect(fixture([], [tensor("future.weight", [64], 999, 0)]))
        self.assertEqual(report["tensor_types"], {"UNKNOWN_999": 1})

    def test_custom_alignment_and_invalid_parser_options(self):
        report = self.inspect(fixture([metadata("general.alignment", 4, number("I", 64))],
                                      [tensor("x", [32], 0, 64)], alignment=64))
        self.assertEqual(report["tensor_data_offset"] % 64, 0)
        data = fixture()
        for options in [{"model_size": 1}, {"model_size": -1},
                        {"max_header_bytes": 0}, {"examples_per_type": -1}]:
            with self.subTest(options=options), self.assertRaises(gguf.GGUFError):
                self.inspect(data, **options)


if __name__ == "__main__":
    unittest.main()
