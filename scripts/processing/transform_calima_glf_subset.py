#!/usr/bin/env python3
"""Deterministic bounded CALIMA-GLF morphology_record transform dry run.

Processes a section-complete subset of the verified morphology database: the
first two non-marker records from every source section, or all records when a
section contains fewer than two. Writes a temporary Parquet and a no-source-text
validation manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

SNAPSHOT_ID = "SNP-000003"
RESOURCE_ID = "RES-000003"
ARTIFACT_ID = "ART-000497"
SOURCE_ARCHIVE = "morphology_db_calima-glf-01-0.1.0.zip"
SOURCE_OBJECT = "morphology.db"
EXPECTED_ARCHIVE_SHA256 = "385a29aa4737335d6431768546aaaf7ecf848dfd4bef460c7e3358168c73c9b3"
EXPECTED_SOURCE_SHA256 = "0b88b55d09eda8edc2f0009cb3f46d4b2ad8176cf5dc7a17a7b65439f7aaae7d"
EXPECTED_SOURCE_BYTES = 7976670
ROWS_PER_SECTION = 2
SECTION_COUNTS = {
    "DEFINES": 25,
    "DEFAULTS": 37,
    "ORDER": 1,
    "TOKENIZATIONS": 1,
    "STEMBACKOFF": 2,
    "PREFIXES": 132,
    "SUFFIXES": 251,
    "STEMS": 22911,
    "TABLE AB": 82867,
    "TABLE BC": 102242,
    "TABLE AC": 865,
}
SECTION_MARKERS = {f"###{name}###": name for name in SECTION_COUNTS}
EXPECTED_SUBSET_COUNTS = {
    section: min(ROWS_PER_SECTION, total) for section, total in SECTION_COUNTS.items()
}
EXPECTED_SUBSET_ROWS = sum(EXPECTED_SUBSET_COUNTS.values())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def foundation_id(physical_line_number: int, raw_line: str) -> str:
    preimage = SNAPSHOT_ID + ARTIFACT_ID + str(physical_line_number) + raw_line
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def parse_key_value_tokens(tokens: list[str], *, context: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for token in tokens:
        if not token:
            continue
        pieces = token.split(":")
        if len(pieces) < 2:
            raise RuntimeError(f"{context}: invalid key:value token {token!r}")
        parsed[pieces[0]] = ":".join(pieces[1:])
    return parsed


def parse_section_record(section: str, raw_line: str) -> dict:
    parsed = {
        "surface_form": None,
        "source_category": None,
        "source_analysis_attributes": None,
        "source_feature_name": None,
        "source_feature_values": None,
        "source_default_attributes": None,
        "source_order_features": None,
        "source_tokenization_features": None,
        "source_backoff_name": None,
        "source_backoff_categories": None,
        "source_category_left": None,
        "source_category_right": None,
    }

    if section == "DEFINES":
        tokens = raw_line.strip().split(" ")
        if len(tokens) < 3 or tokens[0] != "DEFINE":
            raise RuntimeError("invalid DEFINES record")
        feature = tokens[1]
        values = []
        for token in tokens[2:]:
            if not token:
                continue
            pieces = token.split(":")
            if len(pieces) != 2 or pieces[0] != feature:
                raise RuntimeError(f"invalid DEFINES token {token!r}")
            values.append(pieces[1])
        if not values:
            raise RuntimeError("DEFINES record has no values")
        parsed["source_feature_name"] = feature
        parsed["source_feature_values"] = values

    elif section == "DEFAULTS":
        tokens = raw_line.strip().split(" ")
        if len(tokens) < 2 or tokens[0] != "DEFAULT":
            raise RuntimeError("invalid DEFAULTS record")
        attrs = parse_key_value_tokens(tokens[1:], context="DEFAULTS")
        if not attrs:
            raise RuntimeError("DEFAULTS record has no attributes")
        parsed["source_default_attributes"] = attrs

    elif section == "ORDER":
        tokens = raw_line.strip().split(" ")
        if len(tokens) < 2 or tokens[0] != "ORDER":
            raise RuntimeError("invalid ORDER record")
        parsed["source_order_features"] = [t for t in tokens[1:] if t]

    elif section == "TOKENIZATIONS":
        tokens = raw_line.strip().split(" ")
        if len(tokens) < 2 or tokens[0] != "TOKENIZATION":
            raise RuntimeError("invalid TOKENIZATIONS record")
        parsed["source_tokenization_features"] = [t for t in tokens[1:] if t]

    elif section == "STEMBACKOFF":
        tokens = [t for t in raw_line.strip().split(" ") if t]
        if len(tokens) < 3 or tokens[0] != "STEMBACKOFF":
            raise RuntimeError("invalid STEMBACKOFF record")
        parsed["source_backoff_name"] = tokens[1]
        parsed["source_backoff_categories"] = tokens[2:]

    elif section in {"PREFIXES", "SUFFIXES", "STEMS"}:
        line_for_parse = raw_line if section in {"PREFIXES", "SUFFIXES"} else raw_line.strip()
        parts = line_for_parse.split("\t")
        if len(parts) != 3:
            raise RuntimeError(f"invalid {section} record: expected 3 tab fields")
        surface = parts[0].strip() if section in {"PREFIXES", "SUFFIXES"} else parts[0]
        category = parts[1]
        analysis_text = parts[2].strip() if section in {"PREFIXES", "SUFFIXES"} else parts[2]
        attrs = parse_key_value_tokens(analysis_text.split(" "), context=section)
        parsed["surface_form"] = surface
        parsed["source_category"] = category
        parsed["source_analysis_attributes"] = attrs

    elif section in {"TABLE AB", "TABLE BC", "TABLE AC"}:
        tokens = raw_line.strip().split()
        if len(tokens) != 2:
            raise RuntimeError(f"invalid {section} record: expected 2 category tokens")
        parsed["source_category_left"] = tokens[0]
        parsed["source_category_right"] = tokens[1]

    else:
        raise RuntimeError(f"unsupported source section {section!r}")

    return parsed


def read_subset(archive: Path) -> tuple[list[dict], dict]:
    archive_sha = sha256_file(archive)
    if archive_sha != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(f"archive sha256 mismatch: {archive_sha}")

    with zipfile.ZipFile(archive) as zf:
        try:
            info = zf.getinfo(SOURCE_OBJECT)
        except KeyError as exc:
            raise RuntimeError(f"missing {SOURCE_OBJECT}") from exc
        if info.file_size != EXPECTED_SOURCE_BYTES:
            raise RuntimeError(
                f"{SOURCE_OBJECT}: byte-size mismatch {info.file_size} != {EXPECTED_SOURCE_BYTES}"
            )
        source_bytes = zf.read(SOURCE_OBJECT)

    source_sha = sha256_bytes(source_bytes)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"{SOURCE_OBJECT}: sha256 mismatch: {source_sha}")

    text = source_bytes.decode("utf-8-sig", errors="strict")
    rows: list[dict] = []
    seen_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    active_section: str | None = None

    for physical_line_number, raw_line in enumerate(text.splitlines(), start=1):
        marker = raw_line.strip()
        if marker in SECTION_MARKERS:
            active_section = SECTION_MARKERS[marker]
            continue
        if active_section is None:
            if raw_line.strip():
                raise RuntimeError(
                    f"nonempty data before first section marker at line {physical_line_number}"
                )
            continue
        if not raw_line:
            raise RuntimeError(
                f"unexpected blank morphology record in {active_section} at line {physical_line_number}"
            )

        seen_counts[active_section] += 1
        section_record_index = seen_counts[active_section]
        if selected_counts[active_section] >= EXPECTED_SUBSET_COUNTS[active_section]:
            continue

        fields = parse_section_record(active_section, raw_line)
        rows.append(
            {
                "foundation_record_id": foundation_id(physical_line_number, raw_line),
                "snapshot_id": SNAPSHOT_ID,
                "resource_id": RESOURCE_ID,
                "artifact_id": ARTIFACT_ID,
                "source_object": SOURCE_OBJECT,
                "source_record_locator": (
                    f"zip://{SOURCE_ARCHIVE}/{SOURCE_OBJECT}#line={physical_line_number}"
                ),
                "source_split": None,
                "source_record_id": f"{SOURCE_OBJECT}:{physical_line_number}",
                "record_family": "morphology_record",
                "source_attributes": {
                    "physical_line_number": str(physical_line_number),
                    "source_section_record_index": str(section_record_index),
                },
                "raw_record": raw_line,
                "source_section": active_section,
                "source_section_record_index": section_record_index,
                **fields,
            }
        )
        selected_counts[active_section] += 1

    if dict(seen_counts) != SECTION_COUNTS:
        raise RuntimeError(f"full source section accounting mismatch: {dict(seen_counts)}")
    if dict(selected_counts) != EXPECTED_SUBSET_COUNTS:
        raise RuntimeError(f"subset section accounting mismatch: {dict(selected_counts)}")
    if len(rows) != EXPECTED_SUBSET_ROWS:
        raise RuntimeError(f"subset row count mismatch: {len(rows)}")

    return rows, {
        "archive_sha256": archive_sha,
        "source_object_sha256": source_sha,
        "source_object_bytes": len(source_bytes),
        "source_section_counts": dict(seen_counts),
    }


def _map_items(value: dict[str, str] | None):
    return None if value is None else list(value.items())


def write_parquet(rows: list[dict], output: Path) -> None:
    string_map = pa.map_(pa.string(), pa.string())
    schema = pa.schema(
        [
            pa.field("foundation_record_id", pa.string(), nullable=False),
            pa.field("snapshot_id", pa.string(), nullable=False),
            pa.field("resource_id", pa.string(), nullable=False),
            pa.field("artifact_id", pa.string(), nullable=False),
            pa.field("source_object", pa.string(), nullable=False),
            pa.field("source_record_locator", pa.string(), nullable=False),
            pa.field("source_split", pa.string(), nullable=True),
            pa.field("source_record_id", pa.string(), nullable=False),
            pa.field("record_family", pa.string(), nullable=False),
            pa.field("source_attributes", string_map, nullable=False),
            pa.field("raw_record", pa.string(), nullable=False),
            pa.field("source_section", pa.string(), nullable=False),
            pa.field("source_section_record_index", pa.int64(), nullable=False),
            pa.field("surface_form", pa.string(), nullable=True),
            pa.field("source_category", pa.string(), nullable=True),
            pa.field("source_analysis_attributes", string_map, nullable=True),
            pa.field("source_feature_name", pa.string(), nullable=True),
            pa.field("source_feature_values", pa.list_(pa.string()), nullable=True),
            pa.field("source_default_attributes", string_map, nullable=True),
            pa.field("source_order_features", pa.list_(pa.string()), nullable=True),
            pa.field("source_tokenization_features", pa.list_(pa.string()), nullable=True),
            pa.field("source_backoff_name", pa.string(), nullable=True),
            pa.field("source_backoff_categories", pa.list_(pa.string()), nullable=True),
            pa.field("source_category_left", pa.string(), nullable=True),
            pa.field("source_category_right", pa.string(), nullable=True),
        ]
    )
    normalized = []
    for row in rows:
        item = dict(row)
        item["source_attributes"] = list(item["source_attributes"].items())
        item["source_analysis_attributes"] = _map_items(item["source_analysis_attributes"])
        item["source_default_attributes"] = _map_items(item["source_default_attributes"])
        normalized.append(item)
    table = pa.Table.from_pylist(normalized, schema=schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output, compression="zstd", use_dictionary=True)


def validate(rows: list[dict], output: Path, source_meta: dict) -> dict:
    table = pq.read_table(output)
    out = table.to_pylist()
    if table.num_rows != EXPECTED_SUBSET_ROWS:
        raise RuntimeError(f"expected {EXPECTED_SUBSET_ROWS} rows, found {table.num_rows}")
    if len({row["foundation_record_id"] for row in out}) != EXPECTED_SUBSET_ROWS:
        raise RuntimeError("foundation IDs are not unique")

    output_counts = dict(sorted(Counter(row["source_section"] for row in out).items()))
    if output_counts != dict(sorted(EXPECTED_SUBSET_COUNTS.items())):
        raise RuntimeError(f"section row accounting mismatch: {output_counts}")

    original_raw = "\n".join(row["raw_record"] for row in rows)
    output_raw = "\n".join(row["raw_record"] for row in out)
    if original_raw != output_raw:
        raise RuntimeError("raw morphology record preservation mismatch")

    original_locators = "\n".join(row["source_record_locator"] for row in rows)
    output_locators = "\n".join(row["source_record_locator"] for row in out)
    if original_locators != output_locators:
        raise RuntimeError("source locator preservation mismatch")

    for original, emitted in zip(rows, out, strict=True):
        checks = [
            ("foundation_record_id", original["foundation_record_id"], emitted["foundation_record_id"]),
            ("source_section_record_index", original["source_section_record_index"], emitted["source_section_record_index"]),
            ("surface_form", original["surface_form"], emitted["surface_form"]),
            ("source_category", original["source_category"], emitted["source_category"]),
            ("source_feature_values", original["source_feature_values"], emitted["source_feature_values"]),
            ("source_order_features", original["source_order_features"], emitted["source_order_features"]),
            ("source_tokenization_features", original["source_tokenization_features"], emitted["source_tokenization_features"]),
            ("source_backoff_categories", original["source_backoff_categories"], emitted["source_backoff_categories"]),
            ("source_category_left", original["source_category_left"], emitted["source_category_left"]),
            ("source_category_right", original["source_category_right"], emitted["source_category_right"]),
        ]
        for field, expected, actual in checks:
            if expected != actual:
                raise RuntimeError(f"{field} changed after Parquet round trip")
        for field in ("source_analysis_attributes", "source_default_attributes"):
            actual = emitted[field]
            actual = None if actual is None else dict(actual)
            if original[field] != actual:
                raise RuntimeError(f"{field} changed after Parquet round trip")

    schema_text = str(table.schema)
    return {
        "checkpoint": "8C-1E",
        "source_snapshot": SNAPSHOT_ID,
        "record_family": "morphology_record",
        "subset_rule": (
            "morphology.db only; first two non-marker records from each source section, "
            "or all records for sections with fewer than two"
        ),
        "canonical_row_count": EXPECTED_SUBSET_ROWS,
        "source_section_row_counts": output_counts,
        "full_source_section_counts_verified": source_meta["source_section_counts"],
        "unique_foundation_record_ids": EXPECTED_SUBSET_ROWS,
        "identity_preimage_rule": (
            "sha256(snapshot_id + artifact_id + physical_line_number + raw_line)"
        ),
        "source_archive_sha256": source_meta["archive_sha256"],
        "source_object_sha256": source_meta["source_object_sha256"],
        "source_object_bytes": source_meta["source_object_bytes"],
        "schema_sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
        "output_parquet_sha256": sha256_file(output),
        "raw_record_collection_sha256": hashlib.sha256(output_raw.encode("utf-8")).hexdigest(),
        "source_locator_collection_sha256": hashlib.sha256(
            output_locators.encode("utf-8")
        ).hexdigest(),
        "compression": "zstd",
        "checks": {
            "schema_readable": True,
            "section_complete_bounded_subset": True,
            "full_source_section_accounting": True,
            "stable_unique_record_ids": True,
            "source_provenance": True,
            "raw_record_preservation": True,
            "conditional_section_fields_round_trip": True,
            "archive_sha256": True,
            "source_object_sha256": True,
        },
        "raw_source_text_emitted_in_manifest": False,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output-parquet", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()

    rows, source_meta = read_subset(args.archive)
    write_parquet(rows, args.output_parquet)
    manifest = validate(rows, args.output_parquet, source_meta)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
