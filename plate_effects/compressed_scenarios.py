"""Replayable compressed-stream examples for the demo and tests.

Each example builds a small synthetic plate whose used wells are encoded as a
command stream.  The example emits:

* the *published layout* (exactly the existing product format),
* the original command stream (as text and as records),
* the command boundary directory (layout digest, ranges, memory states),
* the handheld load, quota, chosen snapshot boundaries, per-query recovery
  paths and costs, and the replayed query results,
* the minimal :class:`~plate_effects.retrieval.ShippedPackage` the device would
  receive (original stream + range index + selected snapshots only).

Calling :func:`replay_example` on the JSON payload reconstructs a
:class:`LookupViewer` from the shipped package alone and answers every query by
replaying the original stream -- demonstrating that nothing server-side is
needed to recover a value.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping

from .compressed import CommandDirectory
from .design import PlateDesign
from .rational import fmt_rational
from .retrieval import LookupViewer, ShippedPackage, allocate, ship


def _plate(plate_id: str, cols: int) -> PlateDesign:
    """A 1 x ``cols`` plate where every well is a blank control."""

    design = PlateDesign(plate_id, rows=1, cols=cols, min_replicates=1, max_replicates=1)
    for col in range(cols):
        design.set_blank([f"A{col + 1}"])
    return design


def _example_payload(
    name: str,
    description: str,
    *,
    cols: int,
    stream: list[object],
    frequencies: Mapping[str, int],
    max_pages: int,
    layout_id: str | None = None,
) -> dict:
    design = _plate(layout_id or f"plate-{name}", cols)
    layout = design.publish()
    directory = CommandDirectory.build(layout, stream)
    plan = allocate(directory, frequencies, max_pages)
    package = ship(directory, plan)

    viewer = LookupViewer(frequencies=frequencies, max_pages=max_pages)
    viewer.receive_package(package)
    # Re-derive every answer from the shipped package alone (original stream +
    # range index + snapshots), never from the server-side directory.
    replay = {
        "ok": all(
            viewer.query(well).value == value
            for well, value in directory.decoded
        )
        and all(viewer.query(well).missing for well in directory.missing),
        "queries": [
            {
                "well": well,
                "result": viewer.query(well).to_dict(),
            }
            for well in sorted(frequencies)
        ],
    }

    return {
        "name": name,
        "description": description,
        "layout": layout.to_dict(),
        "stream": {
            "text": [command.text() for command in directory.commands],
            "records": [command.to_dict() for command in directory.commands],
        },
        "directory": directory.to_dict(),
        "load": {well: count for well, count in sorted(frequencies.items())},
        "max_pages": max_pages,
        "allocation": plan.to_dict(),
        "shipped_package": package.to_dict(),
        "replay": replay,
    }


# ---------------------------------------------------------------------------
# The four required worked examples
# ---------------------------------------------------------------------------


def example_basic() -> dict:
    """ABS(5), DELTA(1), DELTA(0), DELTA(0); A2 and A4 once; one page.

    Optimal: one page, boundary 4.  A2 reads commands 1-2 (2 reads), A4 reads
    command 4 (1 read), total weighted cost 3.
    """

    return _example_payload(
        "basic",
        "Four reads ABS(5),DELTA(1),DELTA(0),DELTA(0). Query A2 and A4 once "
        "each with one page. The page goes before command 4; A2 reads 2 "
        "commands and A4 reads 1.",
        cols=4,
        stream=["ABS(5)", "DELTA(1)", "DELTA(0)", "DELTA(0)"],
        frequencies={"A2": 1, "A4": 1},
        max_pages=1,
    )


def example_heavy_a2() -> dict:
    """Same stream; A2 frequency 3, A4 once; one page.

    Optimal: page before command 2, weighted cost 3*1 + 1*3 = 6.
    """

    return _example_payload(
        "heavy_a2",
        "Same four reads, but A2 is queried three times. With one page the "
        "optimum moves to boundary 2: A2 reads 1 command (3 reads total) and "
        "A4 reads 3 commands, weighted cost 6.",
        cols=4,
        stream=["ABS(5)", "DELTA(1)", "DELTA(0)", "DELTA(0)"],
        frequencies={"A2": 3, "A4": 1},
        max_pages=1,
    )


def example_repeat() -> dict:
    """ABS(5), DELTA(1), REPEAT(2); A2 and A4 once; one page.

    A4 is produced by command 3, whose range also covers A3, so the whole
    command is read.  Optimal cost 3; a tie between boundaries 2 and 3 is
    broken lexicographically to boundary 2.
    """

    return _example_payload(
        "repeat",
        "Four reads via ABS(5),DELTA(1),REPEAT(2). Query A2 and A4 once each "
        "with one page. Cost 3; the tie between boundaries 2 and 3 is broken "
        "lexicographically to boundary 2.",
        cols=4,
        stream=["ABS(5)", "DELTA(1)", "REPEAT(2)"],
        frequencies={"A2": 1, "A4": 1},
        max_pages=1,
    )


def example_gap_recovery() -> dict:
    """GAP clears memory; a later DELTA must recover from a fresh ABS.

    Stream: ABS(1), GAP(1), ABS(9), DELTA(1), REPEAT(1) over five wells.
    A2 is MISSING (inside GAP); A5 reads 10.  One page goes before command 5.
    """

    return _example_payload(
        "gap_recovery",
        "ABS(1),GAP(1),ABS(9),DELTA(1),REPEAT(1). A2 falls in the GAP and is "
        "MISSING; the DELTA after the GAP recovers from the fresh ABS(9). A "
        "single page before command 5 lets A2 read commands 1-2 and A5 read "
        "command 5.",
        cols=5,
        stream=["ABS(1)", "GAP(1)", "ABS(9)", "DELTA(1)", "REPEAT(1)"],
        frequencies={"A2": 1, "A5": 1},
        max_pages=1,
    )


def example_multi_well_command() -> dict:
    """A single REPEAT covers three wells; querying any of them reads the command.

    Stream: ABS(7), REPEAT(3).  Even the middle well A3 must read the whole
    REPEAT (command 2), so the middle is never a free start.
    """

    return _example_payload(
        "multi_well_command",
        "ABS(7),REPEAT(3) covers A2,A3,A4 in one command. Query the middle well "
        "A3 and the last well A4: both must read the whole command 2, so the "
        "middle of a REPEAT is never a free start.",
        cols=4,
        stream=["ABS(7)", "REPEAT(3)"],
        frequencies={"A3": 1, "A4": 1},
        max_pages=1,
    )


def example_zero_pages() -> dict:
    """With no page budget every query starts at the free stream head."""

    return _example_payload(
        "zero_pages",
        "ABS(5),DELTA(1),DELTA(0),DELTA(0) with a zero page quota. Only the "
        "free head boundary 1 is available, so A2 reads 2 commands and A4 "
        "reads 4, weighted cost 6.",
        cols=4,
        stream=["ABS(5)", "DELTA(1)", "DELTA(0)", "DELTA(0)"],
        frequencies={"A2": 1, "A4": 1},
        max_pages=0,
    )


def build_examples() -> dict[str, dict]:
    builders = (
        example_basic,
        example_heavy_a2,
        example_repeat,
        example_gap_recovery,
        example_multi_well_command,
        example_zero_pages,
    )
    examples = [builder() for builder in builders]
    return {example["name"]: example for example in examples}


# ---------------------------------------------------------------------------
# Replay from a file payload
# ---------------------------------------------------------------------------


def replay_example(payload: Mapping) -> dict:
    """Reconstruct the handheld from a shipped-package payload and answer all.

    The payload's ``shipped_package`` is the only thing the device needs; the
    query wells come from ``load``.  This proves the answers are re-derived by
    replaying the original stream, not read from the directory.
    """

    load = payload["load"]
    package = ShippedPackage.from_dict(payload["shipped_package"])
    viewer = LookupViewer(frequencies=load, max_pages=int(payload.get("max_pages", 0)))
    viewer.receive_package(package)
    results = []
    for well in sorted(load):
        result = viewer.query(well)
        results.append(result.to_dict())
    return {
        "wells": results,
        "stream_digest": package.stream_digest,
        "snapshot_boundaries": list(package.snapshot_boundaries),
    }
