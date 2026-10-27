"""Inspect a GGUF v3 header without loading model tensors.

The inventory checks header structure and tensor start offsets, not tensor payloads
or their hashes. A bounded downloaded header can be inspected with --model-size
set to the full file size; its reported payload is then explicitly unverified.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import struct
import sys


class GGUFError(ValueError):
    """Malformed, truncated, or over-budget GGUF input."""


# GGUF metadata value types; ARRAY and STRING are variable length.
VALUE_TYPES = {
    0: ("UINT8", "B"), 1: ("INT8", "b"), 2: ("UINT16", "H"),
    3: ("INT16", "h"), 4: ("UINT32", "I"), 5: ("INT32", "i"),
    6: ("FLOAT32", "f"), 7: ("BOOL", "B"), 8: ("STRING", None),
    9: ("ARRAY", None), 10: ("UINT64", "Q"), 11: ("INT64", "q"),
    12: ("FLOAT64", "d"),
}
TENSOR_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0",
    7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
    12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
    17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
    21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
    34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0",
}
# general.file_type describes the recipe, not the type of every tensor.
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0",
    8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
    13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
    17: "Q5_K_M", 18: "Q6_K",
}


class HeaderReader:
    def __init__(self, stream, available_bytes, max_header_bytes,
                 max_array_items, max_string_bytes):
        self.stream = stream
        self.available_bytes = available_bytes
        self.max_header_bytes = max_header_bytes
        self.array_items_left = max_array_items
        self.max_string_bytes = max_string_bytes

    @property
    def position(self):
        return self.stream.tell()

    def reserve(self, count):
        if count < 0 or count > self.max_header_bytes - self.position:
            raise GGUFError("header byte budget exceeded")
        if count > self.available_bytes - self.position:
            raise GGUFError(f"truncated header at byte {self.position}")

    def read(self, count):
        self.reserve(count)
        data = self.stream.read(count)
        if len(data) != count:
            raise GGUFError(f"truncated header at byte {self.position}")
        return data

    def skip(self, count):
        self.reserve(count)
        self.stream.seek(count, 1)

    def unpack(self, code):
        return struct.unpack("<" + code, self.read(struct.calcsize("<" + code)))[0]

    def string(self, capture=False, limit=None):
        length = self.unpack("Q")
        if length > self.max_string_bytes or (limit is not None and length > limit):
            raise GGUFError(f"string length {length} exceeds limit")
        if capture:
            try:
                return self.read(length).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise GGUFError("invalid UTF-8 in header string") from exc
        self.skip(length)
        return None

    def value(self, kind, capture=False, depth=0):
        if kind not in VALUE_TYPES:
            raise GGUFError(f"unknown metadata value type {kind}")
        if kind == 8:
            # Descriptions and tokenizer strings need not be allocated.
            return self.string(capture=capture)
        if kind == 9:
            if depth >= 8:
                raise GGUFError("metadata array nesting exceeds limit")
            element_kind = self.unpack("I")
            count = self.unpack("Q")
            if element_kind not in VALUE_TYPES:
                raise GGUFError(f"unknown array element type {element_kind}")
            if count > self.array_items_left:
                raise GGUFError("metadata array item budget exceeded")
            self.array_items_left -= count
            code = VALUE_TYPES[element_kind][1]
            if code is not None:
                self.skip(count * struct.calcsize("<" + code))
            else:
                for _ in range(count):
                    self.value(element_kind, depth=depth + 1)
            return {"element_type": VALUE_TYPES[element_kind][0], "count": count}
        value = self.unpack(VALUE_TYPES[kind][1])
        if kind == 7:
            if value not in (0, 1):
                raise GGUFError("invalid boolean metadata value")
            value = bool(value)
        if isinstance(value, float) and not math.isfinite(value):
            raise GGUFError("non-finite numeric metadata")
        return value if capture else None


def inspect_gguf(path, *, model_size=None, max_header_bytes=64 * 1024 * 1024,
                 max_metadata=100_000, max_tensors=1_000_000,
                 max_array_items=10_000_000, max_string_bytes=16 * 1024 * 1024,
                 examples_per_type=3):
    """Return a bounded header inventory. Payload bytes are never read.

    model_size allows a local header prefix to stand in for a larger file. It is
    a caller assertion, not a remotely authenticated or verified file size.
    """
    path = Path(path)
    available = path.stat().st_size
    full_size = available if model_size is None else model_size
    if not isinstance(full_size, int) or isinstance(full_size, bool) or full_size < available:
        raise GGUFError("model size must be an integer at least as large as the local file")
    if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in (
            max_header_bytes, max_metadata, max_tensors, max_array_items,
            max_string_bytes, examples_per_type)):
        raise GGUFError("all parser limits must be positive integers")
    with path.open("rb") as stream:
        reader = HeaderReader(stream, available, max_header_bytes,
                              max_array_items, max_string_bytes)
        if reader.read(4) != b"GGUF":
            raise GGUFError("expected little-endian GGUF magic")
        version = reader.unpack("I")
        if version != 3:
            raise GGUFError(f"unsupported GGUF version {version}; expected 3")
        tensor_count = reader.unpack("Q")
        metadata_count = reader.unpack("Q")
        if metadata_count > max_metadata or tensor_count > max_tensors:
            raise GGUFError("metadata or tensor count exceeds limit")
        # Even empty names/keys and scalar values occupy at least these bytes.
        reader.reserve(metadata_count * 13 + tensor_count * 32)
        metadata = {}
        arrays = {}
        keys = set()
        for _ in range(metadata_count):
            key = reader.string(capture=True, limit=4096)
            if key in keys:
                raise GGUFError(f"duplicate metadata key {key!r}")
            keys.add(key)
            kind = reader.unpack("I")
            value = reader.value(kind, capture=not key.startswith("tokenizer."))
            if kind == 9:
                arrays[key] = value
            elif value is not None:
                metadata[key] = value

        alignment = metadata.get("general.alignment", 32)
        if (not isinstance(alignment, int) or isinstance(alignment, bool)
                or alignment <= 0 or alignment > 1024 * 1024
                or alignment & (alignment - 1)):
            raise GGUFError("general.alignment must be a power of two up to 1 MiB")
        histogram = Counter()
        examples = defaultdict(list)
        largest_offset = None
        tensor_names = set()
        for _ in range(tensor_count):
            name = reader.string(capture=True, limit=4096)
            if name in tensor_names:
                raise GGUFError(f"duplicate tensor name {name!r}")
            tensor_names.add(name)
            n_dims = reader.unpack("I")
            if not 1 <= n_dims <= 4:
                raise GGUFError(f"invalid tensor dimension count {n_dims}")
            dimensions = [reader.unpack("Q") for _ in range(n_dims)]
            if any(d == 0 for d in dimensions):
                raise GGUFError("tensor dimensions must be nonzero")
            kind = reader.unpack("I")
            offset = reader.unpack("Q")
            if offset % alignment:
                raise GGUFError(f"unaligned tensor start offset for {name!r}")
            largest_offset = offset if largest_offset is None else max(largest_offset, offset)
            label = TENSOR_TYPES.get(kind, f"UNKNOWN_{kind}")
            histogram[label] += 1
            if len(examples[label]) < examples_per_type:
                examples[label].append({"name": name, "dimensions": dimensions,
                                        "relative_offset": offset, "type_id": kind})

        header_bytes = reader.position
        data_offset = (header_bytes + alignment - 1) // alignment * alignment
        if data_offset > full_size:
            raise GGUFError("aligned tensor data offset exceeds model size")
        if largest_offset is not None and largest_offset >= full_size - data_offset:
            raise GGUFError("tensor start offset exceeds model payload bounds")
        architecture = metadata.get("general.architecture")
        if architecture is not None and not isinstance(architecture, str):
            raise GGUFError("general.architecture must be a string")
        geometry = {key: value for key, value in metadata.items()
                    if architecture and key.startswith(architecture + ".")}
        return {
            "format": "GGUF", "version": version, "architecture": architecture,
            "name": metadata.get("general.name"), "geometry": geometry,
            "file_type": metadata.get("general.file_type"),
            "file_type_name": FILE_TYPES.get(metadata.get("general.file_type"), "UNKNOWN"),
            "metadata_count": metadata_count, "array_summaries": arrays,
            "tensor_count": tensor_count, "tensor_types": dict(sorted(histogram.items())),
            "representative_tensors": dict(sorted(examples.items())),
            "header_bytes": header_bytes, "alignment": alignment,
            "tensor_data_offset": data_offset, "model_bytes": full_size,
            "local_bytes": available, "model_size_source": (
                "local_file" if model_size is None else "caller_supplied"),
            "validation": "Header structure and tensor start offsets only; tensor payloads and hashes unverified.",
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--model-size", type=int,
                        help="full model bytes when MODEL is only a downloaded header prefix")
    parser.add_argument("--max-header-bytes", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args(argv)
    try:
        inventory = inspect_gguf(args.model, model_size=args.model_size,
                                 max_header_bytes=args.max_header_bytes)
    except (OSError, GGUFError) as exc:
        parser.exit(2, f"GGUF inspection failed: {exc}\n")
    json.dump(inventory, sys.stdout, indent=2, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
