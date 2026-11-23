"""Offline recovery tool for delimited tables with dropped closing quotes.

Grammar supported (explicit, intentionally narrow):

    record   = field ("," field)*
    field    = quoted | unquoted
    quoted   = DQUOTE quoted-char* DQUOTE
    quoted-char = safe byte / DQUOTE DQUOTE     # "" is an escaped quote
    unquoted = byte other than COMMA / LF / DQUOTE
    record separator = LF (0x0A)

Only four byte classes have meaning: comma (0x2C), LF (0x0A), double quote
(0x2D ... 0x22) and data bytes.  CR is an ordinary data byte.  The only
permitted mutation is *inserting* one missing closing double quote immediately
before a candidate comma or a candidate record newline while a quoted field is
open.  Bytes are never deleted or modified and field contents are never
guessed.  A quoted field still open at end of segment cannot be repaired
(there is no delimiter to insert before).

The solver never enumerates complete interpretations.  It runs a shortest-cost
dynamic program over record-boundary states, then uses forward/backward
optimal distances to decide, per logical line, whether every minimum-cost
interpretation agrees.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Any, Dict, List, Optional, Sequence, Tuple

COMMA = 0x2C
LF = 0x0A
DQUOTE = 0x22
DELIMS = (COMMA, LF)

# Safety bounds for the diagnostic "unbounded" re-run.
DIAG_CAP = 10**9
DIAG_VARIANT_CAP = 10**6


class InputContractError(ValueError):
    """The anchor/table description itself is malformed (a usage error)."""


@dataclass(frozen=True)
class FieldOutcome:
    """One way a field starting at ``pos`` can end.

    end    - position just past the delimiter consumed (== len(data) at EOF)
    cost   - number of inserted closing quotes (0 or 1)
    value  - decoded field value (``""`` already unescaped to ``"``)
    repair - local byte offset immediately before which the quote is inserted,
             or None for an intact close
    """

    end: int
    cost: int
    value: str
    repair: Optional[int]


@dataclass(frozen=True)
class RecordOutcome:
    """One complete, column-count-valid interpretation of one record."""

    end: int
    cost: int
    repairs: Tuple[int, ...]
    fields: Tuple[Dict[str, Any], ...]


# --------------------------------------------------------------------------- #
# Field level parsing
# --------------------------------------------------------------------------- #

def parse_unquoted_field(data: bytes, i: int) -> Tuple[FieldOutcome, ...]:
    """Parse an unquoted field starting at ``i``.

    A double quote inside an unquoted field is a hard structural error under
    the supported grammar: no insertion rule can repair it, so no outcome is
    returned and every state reaching it dies.
    """
    n = len(data)
    j = i
    while j < n and data[j] not in DELIMS:
        if data[j] == DQUOTE:
            return ()
        j += 1
    value = data[i:j].decode("ascii")
    end = n if j == n else j + 1
    return (FieldOutcome(end=end, cost=0, value=value, repair=None),)


def parse_quoted_field(
    data: bytes, i: int, memo: Dict[int, Tuple[FieldOutcome, ...]]
) -> Tuple[FieldOutcome, ...]:
    """Parse a quoted field whose opening quote is at ``i``.

    While the field is open:

    * ``""``               -> one escaped quote, scanning continues;
    * lone ``"`` followed by comma/LF/EOF -> the field closes there.  The same
      delimiter is simultaneously a *repair* candidate (pretend the quote is
      field content and insert a missing close right before the delimiter);
    * lone ``"`` followed by another byte -> ordinary swallowed content;
    * comma/LF            -> repair candidate: insert one closing quote before
      the delimiter, then consume the delimiter;
    * EOF                 -> the field stays open and cannot be repaired.
    """
    cached = memo.get(i)
    if cached is not None:
        return cached
    n = len(data)
    outcomes: Dict[Tuple[int, int, str, Optional[int]], FieldOutcome] = {}
    j = i + 1
    while j < n:
        c = data[j]
        if c == DQUOTE:
            if j + 1 < n and data[j + 1] == DQUOTE:
                j += 2  # escaped quote pair
                continue
            k = j + 1
            if k == n or data[k] in DELIMS:
                # Intact close: quote at j, delimiter at k.  The field ends
                # just past the delimiter (or at EOF), like every other field.
                end = n if k == n else k + 1
                value = data[i + 1 : j].replace(b'""', b'"').decode("ascii")
                fo = FieldOutcome(end=end, cost=0, value=value, repair=None)
                outcomes[(fo.end, fo.cost, fo.value, fo.repair)] = fo
                # Repair at the same delimiter: the quote at j is treated as
                # content and an inserted close sits before the delimiter.
                value_r = data[i + 1 : k].replace(b'""', b'"').decode("ascii")
                fr = FieldOutcome(end=end, cost=1, value=value_r, repair=k)
                outcomes[(fr.end, fr.cost, fr.value, fr.repair)] = fr
                break  # a valid close cannot simultaneously be content
            # lone quote followed by an ordinary byte: swallowed content
            j += 1
            continue
        if c in DELIMS:
            # Repair candidate: close the open quoted field right before this
            # delimiter.  Keep scanning afterwards -- the same open field may
            # offer several equally valid repair points, which is exactly what
            # produces ambiguous lines.
            value = data[i + 1 : j].replace(b'""', b'"').decode("ascii")
            fo = FieldOutcome(end=j + 1, cost=1, value=value, repair=j)
            outcomes[(fo.end, fo.cost, fo.value, fo.repair)] = fo
            j += 1
            continue
        j += 1
    result = tuple(outcomes.values())
    memo[i] = result
    return result


def field_outcomes(
    data: bytes, i: int, qmemo: Dict[int, Tuple[FieldOutcome, ...]]
) -> Tuple[FieldOutcome, ...]:
    if i < len(data) and data[i] == DQUOTE:
        return parse_quoted_field(data, i, qmemo)
    return parse_unquoted_field(data, i)


# --------------------------------------------------------------------------- #
# Record level parsing (memoized per start offset)
# --------------------------------------------------------------------------- #

def _field_constraint_ok(
    columns: Sequence[Dict[str, Any]], fidx: int, value: str
) -> bool:
    if fidx >= len(columns):
        return False
    col = columns[fidx]
    if col.get("required") and value == "":
        return False
    pattern = col.get("regex")
    if pattern is not None and re.fullmatch(pattern, value) is None:
        return False
    return True


def record_outcomes(
    data: bytes,
    start: int,
    ncols: int,
    columns: Sequence[Dict[str, Any]],
    repair_cap: int,
    qmemo: Dict[int, Tuple[FieldOutcome, ...]],
    rmemo: Dict[int, Tuple[RecordOutcome, ...]],
) -> Tuple[RecordOutcome, ...]:
    """All column-count-valid records that can begin at ``start``.

    Every structurally distinct record interpretation is retained.  The
    segment-level repair cap is *not* applied here: a locally pricier record
    may still belong to a globally minimum interpretation depending on the
    prefix chosen.  The cap is enforced once, on whole segments, by the
    forward DP in :func:`solve_segment`.
    """
    cached = rmemo.get(start)
    if cached is not None:
        return cached
    n = len(data)
    # states[(fields_done, pos)] maps a content signature to its bundle.
    states: Dict[Tuple[int, int], Dict[Tuple[Any, ...], Tuple[int, Tuple[int, ...], Tuple[Dict[str, Any], ...]]]] = {
        (0, start): {(): (0, (), ())}
    }
    finished: Dict[Tuple[Any, ...], RecordOutcome] = {}
    for fidx in range(ncols):
        nxt: Dict[Tuple[int, int], Dict[Tuple[Any, ...], Tuple[int, Tuple[int, ...], Tuple[Dict[str, Any], ...]]]] = {}
        last_field = fidx + 1 == ncols
        for (done, pos), paths in states.items():
            for fo in field_outcomes(data, pos, qmemo):
                ends_record = fo.end == n or data[fo.end - 1] == LF
                if last_field:
                    if not ends_record:
                        continue  # last column must terminate the record
                elif ends_record:
                    continue  # newline before all columns are consumed
                if not _field_constraint_ok(columns, fidx, fo.value):
                    continue
                field_end = fo.end if fo.end == n else fo.end - 1  # drop delimiter
                entry = {
                    "index": fidx,
                    "value": fo.value,
                    "raw_range": [pos, field_end],
                    "repair_at": fo.repair,
                }
                add_repair = (fo.repair,) if fo.repair is not None else ()
                for _old_sig, (cost, repairs, entries) in paths.items():
                    repairs2 = repairs + add_repair
                    entries2 = entries + (entry,)
                    c2 = cost + fo.cost
                    sig = tuple(
                        (e["value"], e["repair_at"], e["raw_range"][0], e["raw_range"][1])
                        for e in entries2
                    )
                    nxt.setdefault((fidx + 1, fo.end), {})[sig] = (c2, repairs2, entries2)
        states = nxt
    for (done, pos), paths in states.items():
        if done != ncols:
            continue
        for sig, (cost, repairs, entries) in paths.items():
            finished[sig] = RecordOutcome(end=pos, cost=cost, repairs=repairs, fields=entries)
    result = tuple(sorted(finished.values(), key=lambda r: (r.cost, r.end)))
    rmemo[start] = result
    return result


# --------------------------------------------------------------------------- #
# Anchor contract
# --------------------------------------------------------------------------- #

def validate_anchors(anchors: Sequence[Dict[str, Any]], length: int) -> List[Dict[str, Any]]:
    """Validate and normalize anchors.  Bad anchor definitions are fatal."""
    if not anchors:
        raise InputContractError("at least one anchor is required (file start)")
    norm = []
    for idx, a in enumerate(anchors):
        if not isinstance(a, dict) or "byte_offset" not in a:
            raise InputContractError(f"anchor #{idx} needs a byte_offset")
        off = a["byte_offset"]
        if not isinstance(off, int) or off < 0 or off > length:
            raise InputContractError(f"anchor #{idx}: byte_offset out of range")
        line = a.get("line")
        if line is not None and (not isinstance(line, int) or line < 1):
            raise InputContractError(f"anchor #{idx}: logical line must be a >=1 integer")
        norm.append({
            "byte_offset": off,
            "line": line,
            "expect_first_column": a.get("expect_first_column"),
            "sentinel": bool(a.get("sentinel", False)),
        })
    norm.sort(key=lambda a: a["byte_offset"])
    for a, b in zip(norm, norm[1:]):
        if a["byte_offset"] == b["byte_offset"]:
            raise InputContractError("two anchors share a byte_offset")
        if a["line"] is not None and b["line"] is not None and b["line"] <= a["line"]:
            raise InputContractError("anchor logical lines must be strictly increasing")
    if norm[0]["byte_offset"] != 0:
        raise InputContractError("the first anchor must sit at byte offset 0 (logical line 1)")
    if norm[-1]["byte_offset"] != length:
        raise InputContractError("the last anchor must be the tail sentinel at end-of-file")
    # Line numbers, where present, must be strictly increasing.  An internal
    # anchor may omit a line number: it then fixes only a byte boundary and a
    # first-column expectation, leaving the preceding segment's record count
    # unconstrained (a source of genuine ambiguity).
    numbered = [a["line"] for a in norm if a["line"] is not None]
    if any(b <= a for a, b in zip(numbered, numbered[1:])):
        raise InputContractError("anchor logical lines must be strictly increasing")
    # The tail sentinel is the boundary after the last record.  It carries no
    # first-column expectation; a logical line number, when given, fixes the
    # expected record count of the final segment.
    if norm[-1]["expect_first_column"] is not None:
        raise InputContractError("the tail sentinel cannot require a first-column value")
    norm[-1]["sentinel"] = True
    if norm[0]["line"] is None:
        norm[0]["line"] = 1
    return norm


def _fill_anchor_lines(anchors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign display line numbers to anchors that omitted one.

    An internal anchor without a line number still fixes a byte boundary and a
    first-column expectation; only the output numbering needs a value.  We fill
    such gaps by linear interpolation between the nearest numbered anchors and
    mark them ``line_interpolated``.  The original presence is preserved in
    ``line_was_given`` so :func:`solve_segment` never turns a display number
    into a fabricated record-count contract.  The tail sentinel is left
    untouched when it carries no line.
    """
    n = len(anchors)
    have = [i for i, a in enumerate(anchors) if a["line"] is not None]
    if len(have) == n:
        for a in anchors:
            a["line_was_given"] = True
        return anchors
    out = [dict(a) for a in anchors]
    for a in out:
        a["line_was_given"] = a["line"] is not None
    for li, ri in zip(have, have[1:]):
        if ri - li <= 1:
            continue
        span_lines = out[ri]["line"] - out[li]["line"]
        step = span_lines / (ri - li)
        for j in range(li + 1, ri):
            out[j]["line"] = out[li]["line"] + round(step * (j - li))
            out[j]["line_interpolated"] = True
    return out


# --------------------------------------------------------------------------- #
# Segment solver
# --------------------------------------------------------------------------- #

@dataclass
class SolveResult:
    status: str
    reason: Optional[str] = None
    optimal_cost: Optional[int] = None
    lines: List[Dict[str, Any]] = field(default_factory=list)
    witnesses: List[List[Dict[str, Any]]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _shift_record(ro: RecordOutcome, base: int) -> Dict[str, Any]:
    """Translate a segment-relative record outcome to absolute file offsets."""
    fields = []
    for e in ro.fields:
        fields.append({
            "index": e["index"],
            "value": e["value"],
            "raw_range": [base + e["raw_range"][0], base + e["raw_range"][1]],
            "repair_at": None if e["repair_at"] is None else base + e["repair_at"],
        })
    return {
        "byte_range": [base, base + ro.end],
        "repairs": tuple(base + r for r in ro.repairs),
        "fields": tuple(fields),
        "cost": ro.cost,
    }


def _digest(rec: Optional[Dict[str, Any]]) -> Any:
    if rec is None:
        return ("absent",)
    return (
        tuple(rec["byte_range"]),
        tuple((f["value"], tuple(f["raw_range"]), f["repair_at"]) for f in rec["fields"]),
        tuple(rec["repairs"]),
    )


def solve_segment(
    seg: bytes,
    base: int,
    columns: Sequence[Dict[str, Any]],
    start_anchor: Dict[str, Any],
    end_anchor: Dict[str, Any],
    repair_cap: int,
    variant_cap: int,
) -> SolveResult:
    ncols = len(columns)
    n = len(seg)
    if n == 0:
        return SolveResult(status="isolated", reason="empty_segment")

    qmemo: Dict[int, Tuple[FieldOutcome, ...]] = {}
    rmemo: Dict[int, Tuple[RecordOutcome, ...]] = {}

    edge_cache: Dict[int, Tuple[RecordOutcome, ...]] = {}
    truncated_edges = False

    def edges_at(pos: int) -> Tuple[RecordOutcome, ...]:
        nonlocal truncated_edges
        if pos in edge_cache:
            return edge_cache[pos]
        ros = record_outcomes(seg, pos, ncols, columns, repair_cap, qmemo, rmemo)
        if start_anchor["expect_first_column"] is not None and pos == 0:
            expect = start_anchor["expect_first_column"]
            ros = tuple(r for r in ros if r.fields and r.fields[0]["value"] == expect)
        if len(ros) > variant_cap:
            truncated_edges = True
            ros = ros[:variant_cap]
        edge_cache[pos] = ros
        return ros

    # Forward DP over (byte_pos, records_done).  Record-boundary positions are
    # only 0, EOF, or one past a newline, so the state space stays small.
    dist: Dict[Tuple[int, int], int] = {(0, 0): 0}
    heap: List[Tuple[int, int, int]] = [(0, 0, 0)]
    while heap:
        cost, pos, k = heappop(heap)
        if dist.get((pos, k)) != cost:
            continue
        for ro in edges_at(pos):
            c2 = cost + ro.cost
            if c2 > repair_cap:
                continue
            state = (ro.end, k + 1)
            if c2 < dist.get(state, 10**18):
                dist[state] = c2
                heappush(heap, (c2, ro.end, k + 1))

    # Edge relation built after the distances settle so that *every* edge
    # whose cost matches the optimal distance to its target is retained,
    # including equal-cost alternatives discovered from different parents.
    out_edges: Dict[Tuple[int, int], List[Tuple[RecordOutcome, int, int]]] = {}
    for (pos, k), d in dist.items():
        for ro in edges_at(pos):
            to = (ro.end, k + 1)
            if d + ro.cost == dist.get(to) and d + ro.cost <= repair_cap:
                out_edges.setdefault((pos, k), []).append((ro, ro.end, k + 1))

    r0 = start_anchor["line"]
    r1 = end_anchor["line"]
    # The end anchor is the boundary *after* the last record (the tail
    # sentinel, or the start of the next segment's record).  Hence a segment
    # contains r1 - r0 records, not r1 - r0 + 1.  The count contract only
    # exists when both boundary anchors originally carried logical line
    # numbers; an interpolated display number (or an omitted tail line) leaves
    # the record count deliberately open and permits ambiguity.
    given0 = start_anchor.get("line_was_given", r0 is not None)
    given1 = end_anchor.get("line_was_given", r1 is not None)
    expected_k = r1 - r0 if (given0 and given1 and r0 is not None and r1 is not None) else None
    # Every state that consumes the whole segment is a structural terminal;
    # the contract terminals additionally satisfy the anchored record count.
    all_terminal_states = [s for s in dist if s[0] == n]
    if not all_terminal_states or (
        expected_k is not None
        and not any(s[1] == expected_k for s in all_terminal_states)
    ):
        return _isolate(seg, base, columns, start_anchor, end_anchor,
                        repair_cap, variant_cap, expected_k, truncated_edges,
                        dist)

    # Reverse edge relation for backward shortest-cost propagation.
    rev: Dict[Tuple[int, int], List[Tuple[int, Tuple[int, int], RecordOutcome]]] = {}
    for (pos, k), lst in out_edges.items():
        for ro, epos, ek in lst:
            rev.setdefault((epos, ek), []).append((ro.cost, (pos, k), ro))

    # Backward DP *per terminal record count*.  bdist_for_count[K][state] is
    # the cheapest suffix cost from ``state`` to an EOF terminal holding
    # exactly K records.  Keeping counts separate is essential: a generic
    # suffix-to-any-EOF distance would either drop edges that can only reach
    # the wrong count (hiding ambiguity) or admit edges that cannot satisfy
    # the anchor contract at all.
    bdist_for_count: Dict[int, Dict[Tuple[int, int], int]] = {}
    for K in sorted({s[1] for s in all_terminal_states}):
        bk: Dict[Tuple[int, int], int] = {}
        bheap = []
        for s in all_terminal_states:
            if s[1] == K:
                bk[s] = 0
                bheap.append((0, s[0], s[1]))
        while bheap:
            cost, pos, k = heappop(bheap)
            if bk.get((pos, k)) != cost:
                continue
            for ecost, prev, _ro in rev.get((pos, k), ()):
                c2 = cost + ecost
                if c2 < bk.get(prev, 10**18):
                    bk[prev] = c2
                    heappush(bheap, (c2, prev[0], prev[1]))
        bdist_for_count[K] = bk

    # Contract: the anchor fixes the terminal record count when it carries a
    # line number; otherwise every structural terminal is admissible.
    if expected_k is not None:
        admissible_counts = [expected_k]
        admissible_terminals = [s for s in all_terminal_states if s[1] == expected_k]
    else:
        admissible_counts = sorted(bdist_for_count)
        admissible_terminals = list(all_terminal_states)
    total = min(dist[s] for s in admissible_terminals)

    def admissible_suffix(to_state: Tuple[int, int]) -> Optional[int]:
        """Cheapest suffix from ``to_state`` to any contract terminal."""
        best: Optional[int] = None
        for K in admissible_counts:
            v = bdist_for_count[K].get(to_state)
            if v is not None and (best is None or v < best):
                best = v
        return best

    # An edge belongs to some minimum-cost contract-satisfying interpretation
    # iff dist[from] + edge.cost + admissible_suffix(to) == total.
    line_variants: Dict[int, Dict[Any, Dict[str, Any]]] = {}
    optimal_edges: List[Tuple[Tuple[int, int], Tuple[int, int], RecordOutcome]] = []
    for (pos, k), lst in out_edges.items():
        d = dist.get((pos, k))
        if d is None:
            continue
        for ro, epos, ek in lst:
            to = (epos, ek)
            bsuffix = admissible_suffix(to)
            if bsuffix is None:
                continue
            if d + ro.cost + bsuffix != total:
                continue
            optimal_edges.append(((pos, k), to, ro))
            line_no = r0 + k
            rec = _shift_record(ro, base)
            rec["evidence"] = {
                "prefix_repairs": d,
                "this_record_repairs": ro.cost,
                "suffix_repairs": bsuffix,
                "segment_total_repairs": total,
            }
            line_variants.setdefault(line_no, {})
            line_variants[line_no].setdefault(_digest(rec), rec)

    # Lines that some optimal interpretation skips (merged into a neighbour):
    # a logical line is "absent" under an interpretation when fewer records
    # than the anchor span exist.  Only possible for the tail sentinel span
    # where expected_k is None.
    terminal_counts = {s[1] for s in admissible_terminals if dist[s] == total}
    max_k = max(terminal_counts)
    min_k = min(terminal_counts)
    if expected_k is None and min_k < max_k:
        for missing_k in range(min_k + 1, max_k + 1):
            line_no = r0 + missing_k - 1
            absent = {
                "present": False,
                "evidence": {
                    "note": "a minimum-cost interpretation ends the segment "
                            "with fewer records, so this logical line does not "
                            "exist under that interpretation",
                    "segment_total_repairs": total,
                },
            }
            line_variants.setdefault(line_no, {})
            line_variants[line_no].setdefault(_digest(None), absent)

    # Two full witness interpretations (for ambiguous segments), reconstructed
    # by walking optimal edges -- no enumeration of all interpretations.
    # Reconstruct up to two distinct contract-satisfying witnesses by edge
    # walking.  The admissible terminal set enforces the record-count contract;
    # a global signature guards against returning the same parse twice.
    witnesses = _witnesses(
        optimal_edges, (0, 0), base, r0,
        admissible_terminals=set(admissible_terminals),
    )

    lines_out: List[Dict[str, Any]] = []
    for line_no in sorted(line_variants):
        variants = list(line_variants[line_no].values())
        if len(variants) == 1:
            lines_out.append({"line": line_no, "status": "determined", "variants": variants})
        else:
            lines_out.append({
                "line": line_no,
                "status": "ambiguous",
                "variant_count": len(variants),
                "variants": variants[:2],
            })

    diag: Dict[str, Any] = {
        "candidate_record_boundaries": len(edge_cache),
        "forward_states": len(dist),
        "variant_cap_truncated_edges": truncated_edges,
        "optimal_interpretation_record_count": sorted(terminal_counts),
    }
    return SolveResult(
        status="ok",
        optimal_cost=total,
        lines=lines_out,
        witnesses=witnesses,
        diagnostics=diag,
    )


def _witnesses(
    optimal_edges: List[Tuple[Tuple[int, int], Tuple[int, int], RecordOutcome]],
    start: Tuple[int, int],
    base: int,
    r0: int,
    admissible_terminals: set,
) -> List[List[Dict[str, Any]]]:
    """Reconstruct up to two distinct full optimal interpretations.

    Each DFS branch owns its complete record path, so witnesses never share
    suffixes.  A path only counts when it ends in an admissible terminal (the
    anchored record count, when one exists), and duplicate parses are filtered
    by a global signature.  The walk stops as soon as two distinct parses are
    collected -- it never enumerates the whole interpretation space.
    """
    adj: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], RecordOutcome]]] = {}
    for frm, to, ro in optimal_edges:
        adj.setdefault(frm, []).append((to, ro))
    witnesses: List[List[Dict[str, Any]]] = []
    seen_sigs: set = set()

    def dfs(state: Tuple[int, int], acc: List[Dict[str, Any]]) -> None:
        if len(witnesses) >= 2:
            return
        if state not in adj:
            if acc and state in admissible_terminals:
                sig = tuple((tuple(r["byte_range"]), tuple(r["repairs"]),
                             tuple((f["value"], f["repair_at"]) for f in r["fields"]))
                            for r in acc)
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    witnesses.append(acc)
            return
        for to, ro in adj[state]:
            rec = _shift_record(ro, base)
            rec["line"] = r0 + state[1]
            dfs(to, acc + [rec])
            if len(witnesses) >= 2:
                return

    dfs(start, [])
    return witnesses


def _isolate(
    seg: bytes,
    base: int,
    columns: Sequence[Dict[str, Any]],
    start_anchor: Dict[str, Any],
    end_anchor: Dict[str, Any],
    repair_cap: int,
    variant_cap: int,
    expected_k: Optional[int],
    truncated: bool,
    dist: Dict[Tuple[int, int], int],
) -> SolveResult:
    """Re-run without caps to distinguish cap failure from structural failure."""
    n = len(seg)
    qmemo: Dict[int, Tuple[FieldOutcome, ...]] = {}
    rmemo: Dict[int, Tuple[RecordOutcome, ...]] = {}
    d2: Dict[Tuple[int, int], int] = {(0, 0): 0}
    heap = [(0, 0, 0)]
    # Track whether ANY record interpretation exists at the segment start
    # whose first column matches the anchor expectation.
    start_records = record_outcomes(seg, 0, len(columns), columns, DIAG_CAP, qmemo, rmemo)
    expect = start_anchor["expect_first_column"]
    matching_start = [r for r in start_records
                      if expect is None or (r.fields and r.fields[0]["value"] == expect)]
    while heap:
        cost, pos, k = heappop(heap)
        if d2.get((pos, k)) != cost:
            continue
        for ro in record_outcomes(seg, pos, len(columns), columns, DIAG_CAP, qmemo, rmemo)[:DIAG_VARIANT_CAP]:
            if pos == 0 and expect is not None:
                if not (ro.fields and ro.fields[0]["value"] == expect):
                    continue
            state = (ro.end, k + 1)
            c2 = cost + ro.cost
            if c2 < d2.get(state, 10**18):
                d2[state] = c2
                heappush(heap, (c2, ro.end, k + 1))
    diag_terms = [(s, c) for s, c in d2.items() if s[0] == n and
                  (expected_k is None or s[1] == expected_k)]
    diagnostics: Dict[str, Any] = {
        "repair_cap": repair_cap,
        "variant_cap": variant_cap,
        "variant_cap_truncated_edges": truncated,
        "reachable_states_under_cap": len(dist),
    }
    if diag_terms:
        unbounded_cost = min(c for _, c in diag_terms)
        diagnostics["unbounded_minimum_repairs"] = unbounded_cost
        if unbounded_cost > repair_cap:
            # Even an untruncated search needs more insertions than the
            # per-segment cap allows.
            reason = "repair_cap_exceeded"
        else:
            # A feasible, within-cap parse exists but the bounded forward pass
            # never reached it, so the protective variant cap is the cause.
            reason = "variant_cap_exceeded"
    else:
        hard_errors = _first_hard_errors(seg)
        if hard_errors:
            # A byte the grammar cannot repair dominates: the segment is
            # unusable regardless of what its first column looks like.
            reason = "structural_hard_error"
            diagnostics["first_structural_error_offsets"] = [
                base + o for o in hard_errors[:5]]
        elif expect is not None and not matching_start:
            # Structurally parseable, but no interpretation starts with the
            # anchored first-column value.
            reason = "first_column_mismatch"
            diagnostics["expected_first_column"] = expect
            diagnostics["available_first_columns"] = sorted(
                {r.fields[0]["value"] for r in start_records if r.fields})[:10]
        else:
            reason = "unsatisfiable_contract"
        diagnostics["unbounded_reachable_states"] = len(d2)
    r0, r1 = start_anchor["line"], end_anchor["line"]
    span = {"from_line": r0, "to_line": r1}
    return SolveResult(status="isolated", reason=reason,
                       lines=[{"line_span": span, "byte_range": [base, base + n]}],
                       diagnostics=diagnostics)


def _first_hard_errors(seg: bytes) -> List[int]:
    """Locate bytes that are unrepairable under the supported grammar.

    A double quote occurring *inside* an unquoted field (i.e. not at its first
    byte) cannot be fixed by inserting a closing quote, so it is a hard
    structural error.  A quoted field still open at end of segment is also
    hard (there is no delimiter to insert before).  Everything else is either
    valid or repairable.
    """
    out: List[int] = []
    i, n = 0, len(seg)
    at_field_start = True
    in_quote = False
    while i < n:
        c = seg[i]
        if in_quote:
            if c == DQUOTE:
                if i + 1 < n and seg[i + 1] == DQUOTE:
                    i += 2  # escaped pair
                    continue
                in_quote = False  # a close quote
            i += 1
            continue
        if at_field_start and c == DQUOTE:
            in_quote = True
            at_field_start = False
            i += 1
            continue
        if c == COMMA or c == LF:
            at_field_start = True
        elif c == DQUOTE:
            out.append(i)  # quote inside an unquoted field
            at_field_start = False
        else:
            at_field_start = False
        i += 1
    if in_quote:
        out.append(n)  # quoted field open at end of segment
    return out


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #

def analyze(doc: Dict[str, Any], variant_cap: int = 128) -> Dict[str, Any]:
    raw = doc["data"]
    try:
        if isinstance(raw, str):
            data = raw.encode("ascii")
        else:
            data = bytes(raw)
            data.decode("ascii")
    except (UnicodeEncodeError, UnicodeDecodeError) as exc:
        raise InputContractError(f"input data is not ASCII: {exc}")
    columns = doc["columns"]
    if not isinstance(columns, list) or not columns:
        raise InputContractError("columns must be a non-empty list")
    anchors = validate_anchors(doc["anchors"], len(data))
    anchors = _fill_anchor_lines(anchors)
    default_cap = int(doc.get("repair_cap", 0))
    segments_out = []
    for idx, (a, b) in enumerate(zip(anchors, anchors[1:])):
        b0, b1 = a["byte_offset"], b["byte_offset"]
        seg = data[b0:b1]
        cap = default_cap
        if isinstance(doc.get("segment_repair_caps"), dict):
            cap = int(doc["segment_repair_caps"].get(str(idx), cap))
        res = solve_segment(seg, b0, columns, a, b, cap, variant_cap)
        segments_out.append({
            "segment_index": idx,
            "byte_range": [b0, b1],
            "start_anchor_line": a["line"],
            "end_anchor_line": b["line"],
            "start_line_interpolated": bool(a.get("line_interpolated")),
            "end_line_interpolated": bool(b.get("line_interpolated")),
            "record_count_contract_enforced": res.status == "ok" and (
                a.get("line_was_given", True) and b.get("line_was_given", True)),
            "repair_cap": cap,
            "status": res.status,
            "reason": res.reason,
            "optimal_repairs": res.optimal_cost,
            "lines": res.lines,
            "witness_interpretations": res.witnesses,
            "diagnostics": res.diagnostics,
        })
    return {
        "file_size": len(data),
        "column_count": len(columns),
        "segments": segments_out,
    }


# --------------------------------------------------------------------------- #
# Rendering / CLI
# --------------------------------------------------------------------------- #

def _visible(value: str) -> str:
    """Make embedded newlines/quotes visible in human-readable output."""
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def render_text(report: Dict[str, Any]) -> str:
    out: List[str] = []
    out.append(f"file_size={report['file_size']} columns={report['column_count']}")
    for seg in report["segments"]:
        out.append("")
        out.append(
            f"[segment {seg['segment_index']}] bytes {seg['byte_range'][0]}..{seg['byte_range'][1]} "
            f"lines {seg['start_anchor_line']}..{seg['end_anchor_line']} "
            f"cap={seg['repair_cap']} status={seg['status']}"
            + (f" reason={seg['reason']}" if seg["reason"] else "")
        )
        if seg["status"] == "isolated":
            for ln in seg["lines"]:
                if "line_span" in ln:
                    out.append(f"  ISOLATED line span {ln['line_span']} bytes {ln['byte_range']}")
            continue
        out.append(f"  minimum repairs = {seg['optimal_repairs']}")
        for ln in seg["lines"]:
            if ln["status"] == "determined":
                v = ln["variants"][0]
                if v.get("present") is False:
                    out.append(f"  line {ln['line']}: DETERMINED ABSENT "
                               f"(no minimum-cost interpretation contains this line)")
                    continue
                repairs = ",".join(map(str, v["repairs"])) or "-"
                vals = " | ".join(_visible(f["value"]) for f in v["fields"])
                out.append(f"  line {ln['line']}: DETERMINED bytes={v['byte_range']} "
                           f"repairs@[{repairs}] fields=[{vals}]")
            else:
                out.append(f"  line {ln['line']}: AMBIGUOUS ({ln['variant_count']} distinct local results)")
                for vi, v in enumerate(ln["variants"], 1):
                    if v.get("present") is False:
                        out.append(f"    interpretation {vi}: line absent "
                                   f"({v['evidence']['note']})")
                        continue
                    repairs = ",".join(map(str, v["repairs"])) or "-"
                    vals = " | ".join(_visible(f["value"]) for f in v["fields"])
                    ev = v["evidence"]
                    out.append(
                        f"    interpretation {vi}: bytes={v['byte_range']} repairs@[{repairs}] "
                        f"fields=[{vals}]"
                    )
                    out.append(
                        f"      evidence: prefix={ev['prefix_repairs']} + here={ev['this_record_repairs']} "
                        f"+ suffix={ev['suffix_repairs']} = {ev['segment_total_repairs']} (segment minimum)"
                    )
        if seg.get("witness_interpretations"):
            out.append("  witness interpretations (full segment-compatible parses):")
            for wi, w in enumerate(seg["witness_interpretations"], 1):
                desc = "; ".join(
                    f"L{r['line']}=" + ",".join(_visible(f["value"]) for f in r["fields"])
                    + (f"+repairs[{','.join(map(str, r['repairs']))}]" if r["repairs"] else "")
                    for r in w
                )
                out.append(f"    witness {wi}: {desc}")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Recover delimited tables where closing quotes were dropped.")
    ap.add_argument("input", help="JSON input file (data + columns + anchors)")
    ap.add_argument("-o", "--output", help="write JSON report here (default: stdout text)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--variant-cap", type=int, default=128,
                    help="per-state distinct record-interpretation cap (default 128)")
    args = ap.parse_args(argv)
    with open(args.input, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    try:
        report = analyze(doc, variant_cap=args.variant_cap)
    except InputContractError as exc:
        print(f"input contract error: {exc}", file=sys.stderr)
        return 2
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
    elif args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_text(report))
    # Non-zero exit if any segment was isolated, useful in pipelines.
    return 1 if any(s["status"] == "isolated" for s in report["segments"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
