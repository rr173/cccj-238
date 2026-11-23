"""Re-runnable, assertion-based tests for the quote-recovery tool.

Run with:  python3 -I tests/test_recover.py
The fixtures are regenerated into a throwaway directory on every run, so the
suite is fully deterministic and never depends on stale on-disk state.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csv_recover.recover import (  # noqa: E402
    InputContractError,
    analyze,
    parse_quoted_field,
    record_outcomes,
)
from tests import make_fixtures  # noqa: E402

COLS = make_fixtures.COLS_2


def _load(path: str):
    with open(path, "r", encoding="ascii") as fh:
        return json.load(fh)


def _segments(report):
    return report["segments"]


def _line(report, line_no):
    for seg in _segments(report):
        for ln in seg["lines"]:
            if ln.get("line") == line_no:
                return ln
    raise KeyError(line_no)


class FixtureBackedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="csv_recover_fx_")
        cls.paths = make_fixtures.make_all(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_01_intact_multiline_is_fully_determined(self):
        report = analyze(_load(self.paths["01"]))
        seg = _segments(report)[0]
        self.assertEqual(seg["status"], "ok")
        self.assertEqual(seg["optimal_repairs"], 0)
        self.assertEqual({ln["line"]: ln["status"] for ln in seg["lines"]},
                         {1: "determined", 2: "determined", 3: "determined"})
        l1 = _line(report, 1)["variants"][0]
        # Embedded newline preserved verbatim inside the quoted field.
        self.assertEqual(l1["fields"][1]["value"], "line one\nline two")
        # Escaped quote unescaped exactly once.
        self.assertEqual(_line(report, 3)["variants"][0]["fields"][1]["value"],
                         'say "hi"')

    def test_02_single_repair_recovers_table(self):
        report = analyze(_load(self.paths["02"]))
        seg = _segments(report)[0]
        self.assertEqual(seg["status"], "ok")
        self.assertEqual(seg["optimal_repairs"], 1)
        # The minimum repair is the early newline; the full 3-record parse is
        # one of the optimal interpretations.
        witness_lens = {len(w) for w in seg["witness_interpretations"]}
        self.assertIn(3, witness_lens)

    def test_03_wrong_first_column_is_isolated_with_reason(self):
        report = analyze(_load(self.paths["03"]))
        seg = _segments(report)[0]
        self.assertEqual(seg["status"], "isolated")
        self.assertEqual(seg["reason"], "first_column_mismatch")
        self.assertEqual(seg["diagnostics"]["expected_first_column"], "B1")
        self.assertIn("A1", seg["diagnostics"]["available_first_columns"])

    def test_04_cap_exceeded_then_relieved_control(self):
        blocked = analyze(_load(self.paths["04"]))["segments"][0]
        self.assertEqual(blocked["status"], "isolated")
        self.assertEqual(blocked["reason"], "repair_cap_exceeded")
        # The unbounded re-run tells the operator how many repairs are needed.
        self.assertEqual(blocked["diagnostics"]["unbounded_minimum_repairs"], 2)

        relieved = analyze(_load(self.paths["04b"]))["segments"][0]
        self.assertEqual(relieved["status"], "ok")
        self.assertEqual(relieved["optimal_repairs"], 2)
        self.assertTrue(
            all(ln["status"] == "determined" for ln in relieved["lines"]))

    def test_05_genuine_ambiguity_two_distinct_witnesses(self):
        report = analyze(_load(self.paths["05"]))
        seg = _segments(report)[0]
        self.assertEqual(seg["status"], "ok")
        self.assertEqual(seg["optimal_repairs"], 1)
        # Two structurally different full parses, both costing one repair.
        self.assertEqual(len(seg["witness_interpretations"]), 2)
        w1, w2 = seg["witness_interpretations"]
        self.assertNotEqual(
            [tuple(r["byte_range"]) for r in w1],
            [tuple(r["byte_range"]) for r in w2])
        # Line 1 has two distinct local results with full-segment evidence.
        l1 = _line(report, 1)
        self.assertEqual(l1["status"], "ambiguous")
        self.assertEqual(l1["variant_count"], 2)
        for v in l1["variants"]:
            ev = v["evidence"]
            self.assertEqual(
                ev["prefix_repairs"] + ev["this_record_repairs"]
                + ev["suffix_repairs"],
                ev["segment_total_repairs"])
        # Line 2 is ambiguous between "present" and "absent".
        l2 = _line(report, 2)
        self.assertEqual(l2["status"], "ambiguous")
        presence = {v.get("present", True) for v in l2["variants"]}
        self.assertIn(False, presence)

    def test_06_determined_lines_survive_beside_ambiguity(self):
        report = analyze(_load(self.paths["06"]))
        segs = _segments(report)
        # Prefix and suffix segments are untouched and fully determined.
        self.assertTrue(
            all(ln["status"] == "determined" for ln in segs[0]["lines"]))
        self.assertTrue(
            all(ln["status"] == "determined" for ln in segs[2]["lines"]))
        # The middle segment is ambiguous and yields two distinct witnesses.
        mid = segs[1]
        self.assertTrue(any(ln["status"] == "ambiguous" for ln in mid["lines"]))
        self.assertEqual(len(mid["witness_interpretations"]), 2)

    def test_07_hard_error_isolated_but_neighbour_delivered(self):
        report = analyze(_load(self.paths["07"]))
        segs = _segments(report)
        self.assertEqual(segs[0]["status"], "isolated")
        self.assertEqual(segs[0]["reason"], "structural_hard_error")
        self.assertTrue(segs[0]["diagnostics"]["first_structural_error_offsets"])
        # The intact neighbouring segment is still delivered.
        self.assertEqual(segs[1]["status"], "ok")
        self.assertTrue(
            all(ln["status"] == "determined" for ln in segs[1]["lines"]))

    def test_08_scale_runs_without_enumerating_interpretations(self):
        doc = _load(self.paths["08"])
        n_blocks = len(doc["anchors"]) - 1
        start = time.perf_counter()
        report = analyze(doc)
        elapsed = time.perf_counter() - start
        # 400 independent repair decisions would mean 2**400 full parses; the
        # DP must finish in well under a second and never enumerate them.
        self.assertLess(elapsed, 5.0)
        self.assertEqual(len(report["segments"]), n_blocks)
        self.assertTrue(
            all(s["status"] == "ok" for s in report["segments"]))
        total_states = sum(s["diagnostics"]["forward_states"]
                           for s in report["segments"])
        # State count grows linearly in the data, never exponentially.
        self.assertLess(total_states, 10 * n_blocks + 50)


class GrammarUnitTests(unittest.TestCase):
    def test_quoted_field_offers_every_repair_point(self):
        data = b'D1,"aa\nbb\nD2,cc\n'
        outs = parse_quoted_field(data, 3, {})
        repairs = sorted(o.repair for o in outs if o.cost == 1)
        # One repair candidate before every delimiter met while the quote is
        # open: two interior newlines, the comma, and the record newline.
        self.assertEqual(repairs, [6, 9, 12, 15])

    def test_escaped_quote_pair_is_content_not_close(self):
        data = b'"say ""hi"""\n'
        outs = parse_quoted_field(data, 0, {})
        intact = [o for o in outs if o.cost == 0]
        self.assertEqual(len(intact), 1)
        self.assertEqual(intact[0].value, 'say "hi"')

    def test_quote_inside_unquoted_field_is_hard_error(self):
        data = b'abc"def,xx\n'
        ros = record_outcomes(data, 0, 2, COLS, 9, {}, {})
        self.assertEqual(ros, ())

    def test_bytes_are_never_modified(self):
        doc = _load(make_fixtures.make_repair_ok(
            tempfile.mkdtemp(prefix="csv_recover_bytes_")))
        original = doc["data"].encode("ascii")
        analyze(doc)
        # The input document itself is untouched; repairs are offsets only.
        self.assertEqual(doc["data"].encode("ascii"), original)


class ContractTests(unittest.TestCase):
    def _doc(self, **over):
        base = {
            "data": "A,b\n",
            "columns": COLS,
            "repair_cap": 0,
            "anchors": [
                {"byte_offset": 0, "line": 1},
                {"byte_offset": 4},
            ],
        }
        base.update(over)
        return base

    def test_first_anchor_must_be_at_zero(self):
        doc = self._doc(anchors=[
            {"byte_offset": 1, "line": 1}, {"byte_offset": 4}])
        with self.assertRaises(InputContractError):
            analyze(doc)

    def test_last_anchor_must_be_tail_sentinel(self):
        doc = self._doc(anchors=[
            {"byte_offset": 0, "line": 1}, {"byte_offset": 3}])
        with self.assertRaises(InputContractError):
            analyze(doc)

    def test_anchor_lines_must_increase(self):
        doc = self._doc(anchors=[
            {"byte_offset": 0, "line": 2}, {"byte_offset": 4, "line": 1}])
        with self.assertRaises(InputContractError):
            analyze(doc)

    def test_non_ascii_is_rejected(self):
        with self.assertRaises(InputContractError):
            analyze(self._doc(data="A,é\n"))


class SegmentPolicyTests(unittest.TestCase):
    def test_per_segment_caps_are_independent(self):
        s0 = 'A1,"x\ny\nA2,z\n'
        s1 = 'B1,"p\nq\nB2,r\n'
        b1 = len(s0.encode("ascii"))
        n = len((s0 + s1).encode("ascii"))
        doc = {
            "data": s0 + s1,
            "columns": COLS,
            "repair_cap": 0,
            "segment_repair_caps": {"0": 0, "1": 1},
            "anchors": [
                {"byte_offset": 0, "line": 1, "expect_first_column": "A1"},
                {"byte_offset": b1, "line": 3, "expect_first_column": "B1"},
                {"byte_offset": n, "line": 5},
            ],
        }
        segs = analyze(doc)["segments"]
        self.assertEqual(segs[0]["status"], "isolated")
        self.assertEqual(segs[0]["reason"], "repair_cap_exceeded")
        self.assertEqual(segs[1]["status"], "ok")
        self.assertEqual(segs[1]["optimal_repairs"], 1)

    def test_coincident_anchors_are_contract_errors(self):
        # Two anchors sharing a byte offset cannot define a segment and are
        # rejected up front, so zero-width segments never reach the solver.
        s0 = "G1,a\n"          # 5 bytes
        s1 = "G2,b\n"          # 5 bytes
        b1 = len(s0.encode("ascii"))
        n = len((s0 + s1).encode("ascii"))
        doc = {
            "data": s0 + s1,
            "columns": COLS,
            "repair_cap": 0,
            "anchors": [
                {"byte_offset": 0, "line": 1, "expect_first_column": "G1"},
                {"byte_offset": b1, "line": 2},
                {"byte_offset": b1, "line": 2},   # coincident -> invalid
                {"byte_offset": n, "line": 3},
            ],
        }
        with self.assertRaises(InputContractError):
            analyze(doc)

    def test_determined_fact_requires_range_value_and_repair_agreement(self):
        # A segment with a unique minimum interpretation is fully determined.
        data = 'A1,"x\ny"\nA2,z\n'
        doc = {
            "data": data,
            "columns": COLS,
            "repair_cap": 0,
            "anchors": [
                {"byte_offset": 0, "line": 1, "expect_first_column": "A1"},
                {"byte_offset": len(data.encode("ascii")), "line": 3},
            ],
        }
        seg = analyze(doc)["segments"][0]
        self.assertEqual(seg["status"], "ok")
        for ln in seg["lines"]:
            self.assertEqual(ln["status"], "determined")
            self.assertEqual(len(ln["variants"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
