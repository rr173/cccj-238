"""Generate deterministic, re-runnable fixtures for the recovery tool.

Each fixture is written into its own empty directory under the target root.
The fixtures are untrusted *data* (byte strings), never code.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

COLS_2 = [
    {"name": "id", "required": True, "regex": r"[A-Za-z0-9]+(;[A-Za-z0-9]+)*"},
    {"name": "note", "required": False},
]


def _write(root: str, name: str, doc: Dict[str, Any]) -> str:
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    for fn in os.listdir(d):
        os.remove(os.path.join(d, fn))
    path = os.path.join(d, "input.json")
    with open(path, "w", encoding="ascii") as fh:
        json.dump(doc, fh, indent=2)
    return path


def make_multiline_ok(root: str) -> str:
    # Intact file incl. an embedded newline inside a quoted field.
    data = 'K1,"line one\nline two"\nK2,plain\nK3,"say ""hi"""\n'
    doc = {
        "data": data,
        "columns": COLS_2,
        "repair_cap": 0,
        "anchors": [
            {"byte_offset": 0, "line": 1, "expect_first_column": "K1"},
            {"byte_offset": len(data.encode("ascii")), "sentinel": True},
        ],
    }
    return _write(root, "01_multiline_ok", doc)


def make_repair_ok(root: str) -> str:
    # The closing quote after 'hello\nworld' was dropped: one inserted quote
    # before the record newline fully restores the table.
    data = 'A1,"hello\nworld\nA2,second\nA3,third\n'
    doc = {
        "data": data,
        "columns": COLS_2,
        "repair_cap": 1,
        "anchors": [
            {"byte_offset": 0, "line": 1, "expect_first_column": "A1"},
            {"byte_offset": len(data.encode("ascii")), "sentinel": True},
        ],
    }
    return _write(root, "02_repair_ok", doc)


def make_anchor_mismatch(root: str) -> str:
    # The bytes are structurally intact and within the cap, but the start
    # anchor demands first column B1 while every parse begins with A1.  The
    # segment cannot satisfy its anchor contract and must be isolated with
    # reason first_column_mismatch (not a structural or cap failure).
    data = 'A1,"line one\nline two"\nA2,second\n'
    doc = {
        "data": data,
        "columns": COLS_2,
        "repair_cap": 2,
        "anchors": [
            {"byte_offset": 0, "line": 1, "expect_first_column": "B1"},
            {"byte_offset": len(data.encode("ascii")), "line": 3},
        ],
    }
    return _write(root, "03_anchor_firstcol_wrong", doc)


def make_cap_exceeded(root: str) -> str:
    # Two independently damaged quoted fields but the per-segment cap allows
    # zero repairs, so the segment is isolated with reason repair_cap_exceeded;
    # the diagnostic re-run reports the unbounded minimum (2 repairs).
    data = 'C1,"aaa\nbbb\nC2,"ccc\nddd\nC3,ok\n'
    doc = {
        "data": data,
        "columns": COLS_2,
        "repair_cap": 0,
        "anchors": [
            {"byte_offset": 0, "line": 1, "expect_first_column": "C1"},
            {"byte_offset": len(data.encode("ascii")), "line": 4},
        ],
    }
    return _write(root, "04_cap_exceeded", doc)


def make_cap_satisfied_after_relief(root: str) -> str:
    # Same bytes as 04 but a cap that covers both repairs: the segment then
    # parses fully, demonstrating the cap (not the bytes) decides isolation.
    data = 'C1,"aaa\nbbb\nC2,"ccc\nddd\nC3,ok\n'
    doc = {
        "data": data,
        "columns": COLS_2,
        "repair_cap": 2,
        "anchors": [
            {"byte_offset": 0, "line": 1, "expect_first_column": "C1"},
            {"byte_offset": len(data.encode("ascii")), "line": 4},
        ],
    }
    return _write(root, "04b_cap_relieved", doc)


def make_ambiguous(root: str) -> str:
    # One damaged open quote spanning two candidate record newlines.  The
    # tail sentinel carries no line number, so a 1-record parse (repair at the
    # final newline) and a 2-record parse (repair at the earlier newline plus
    # the intact second record) tie at exactly one repair.  Every line is
    # therefore ambiguous; line 2 is additionally "absent" under one parse.
    data = 'D1,"aaa\nbbb\nD2,ccc\n'
    doc = {
        "data": data,
        "columns": COLS_2,
        "repair_cap": 2,
        "anchors": [
            {"byte_offset": 0, "line": 1, "expect_first_column": "D1"},
            {"byte_offset": len(data.encode("ascii"))},
        ],
    }
    return _write(root, "05_ambiguous", doc)


def make_ambiguous_with_determined_edge(root: str) -> str:
    # The middle segment is genuinely ambiguous (two equal-cost parses that
    # differ in record count because the tail boundary is a byte-only anchor
    # whose end side carries no line number), while the anchored prefix and
    # suffix segments are intact and stay DETERMINED.  This proves determined
    # lines survive right next to ambiguous ones.
    head = 'E1,safe1\nE2,safe2\n'
    mid = 'D1,"aa\nbb\nD2,cc\nD3,ee\n'
    tail = 'E7,safe7\n'
    data = head + mid + tail
    b1 = len(head.encode("ascii"))
    b2 = b1 + len(mid.encode("ascii"))
    doc = {
        "data": data,
        "columns": COLS_2,
        "repair_cap": 3,
        "anchors": [
            {"byte_offset": 0, "line": 1, "expect_first_column": "E1"},
            {"byte_offset": b1, "line": 3, "expect_first_column": "D1"},
            # Internal boundary anchor: its byte position delimits segment 1,
            # but it gives no line on segment 1's end side, leaving the
            # damaged segment with two equal-cost record counts.
            {"byte_offset": b2, "expect_first_column": "E7"},
            {"byte_offset": len(data.encode("ascii")), "line": 7},
        ],
    }
    return _write(root, "06_ambiguous_edge_determined", doc)


def make_isolated_neighbour_ok(root: str) -> str:
    # Segment 0 has a hard error (quote inside an unquoted field); segment 1
    # is intact and must still be delivered.
    bad = 'F1,brok"en\nF2,x\n'
    good = 'G1,"multi\nline"\nG2,plain\n'
    data = bad + good
    b1 = len(bad.encode("ascii"))
    doc = {
        "data": data,
        "columns": COLS_2,
        "repair_cap": 1,
        "anchors": [
            {"byte_offset": 0, "line": 1, "expect_first_column": "F1"},
            {"byte_offset": b1, "line": 3, "expect_first_column": "G1"},
            {"byte_offset": len(data.encode("ascii")), "line": 5},
        ],
    }
    return _write(root, "07_isolated_neighbour_ok", doc)


def make_scale(root: str, blocks: int = 400) -> str:
    # Each block is one damaged quoted field whose repair is forced by the
    # next block's anchored line-start.  The number of independent binary
    # repair choices is large, but record boundaries are fixed.
    parts: List[str] = []
    offsets = [0]
    for i in range(blocks):
        parts.append(f'H{i},"v{i}\nw{i}\n')
        offsets.append(offsets[-1] + len(parts[-1].encode("ascii")))
    data = "".join(parts)
    anchors = [{"byte_offset": 0, "line": 1, "expect_first_column": "H0"}]
    for i in range(1, blocks):
        anchors.append({
            "byte_offset": offsets[i],
            "line": i + 1,
            "expect_first_column": f"H{i}",
        })
    anchors.append({"byte_offset": len(data.encode("ascii")),
                    "line": blocks + 1})
    doc = {
        "data": data,
        "columns": COLS_2,
        "repair_cap": blocks,
        "anchors": anchors,
    }
    return _write(root, f"08_scale_{blocks}", doc)


def make_all(root: str) -> Dict[str, str]:
    return {
        "01": make_multiline_ok(root),
        "02": make_repair_ok(root),
        "03": make_anchor_mismatch(root),
        "04": make_cap_exceeded(root),
        "04b": make_cap_satisfied_after_relief(root),
        "05": make_ambiguous(root),
        "06": make_ambiguous_with_determined_edge(root),
        "07": make_isolated_neighbour_ok(root),
        "08": make_scale(root),
    }


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "fixtures")
    os.makedirs(target, exist_ok=True)
    for key, path in sorted(make_all(target).items()):
        print(key, path)
