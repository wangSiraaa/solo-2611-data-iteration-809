"""Tests for compressed command streams, the boundary directory and the
snapshot-allocating lookup device."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from plate_effects.compressed import (
    Command,
    CommandDirectory,
    CommandSyntaxError,
    CompressedStreamError,
    StreamDecodeError,
    parse_command,
    parse_commands,
)
from plate_effects.compressed_scenarios import (
    build_examples,
    example_basic,
    example_gap_recovery,
    example_heavy_a2,
    example_multi_well_command,
    example_repeat,
    example_zero_pages,
    replay_example,
)
from plate_effects.design import PlateDesign
from plate_effects.retrieval import (
    AllocationPlan,
    LookupViewer,
    RecoveryError,
    RetrievalError,
    ShippedPackage,
    StaleSnapshotError,
    allocate,
    ship,
)
from plate_effects.warehouse import ReadoutWarehouse

F = Fraction


def blank_layout(plate_id: str, cols: int) -> "object":
    design = PlateDesign(plate_id, rows=1, cols=cols, min_replicates=1, max_replicates=1)
    for col in range(cols):
        design.set_blank([f"A{col + 1}"])
    return design.publish()


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------


def test_parse_command_forms():
    assert parse_command("ABS(5)") == Command("ABS", value=F(5))
    assert parse_command("DELTA(-3/4)") == Command("DELTA", value=F(-3, 4))
    assert parse_command("REPEAT(3)") == Command("REPEAT", count=3)
    assert parse_command("GAP(2)") == Command("GAP", count=2)
    assert parse_command({"op": "ABS", "value": "1/2"}) == Command("ABS", value=F(1, 2))
    assert parse_command(("DELTA", 2)) == Command("DELTA", value=F(2))
    assert parse_command(("REPEAT", "3")) == Command("REPEAT", count=3)


@pytest.mark.parametrize(
    "token",
    [
        "ABS()",  # no value
        "REPEAT(0)",  # non-positive
        "REPEAT(-1)",
        "GAP(0)",
        "FROB(1)",  # unknown op
        "ABS(1) trailing",
        "REPEAT(1.5)",
        "REPEAT(x)",
        ("REPEAT", True),
    ],
)
def test_parse_command_rejects_bad_tokens(token):
    with pytest.raises(CommandSyntaxError):
        parse_command(token)


def test_parse_commands_rejects_string_input():
    with pytest.raises(CommandSyntaxError):
        parse_commands("ABS(1)")


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def test_basic_decode_order_is_row_major():
    layout = blank_layout("P", 4)
    directory = CommandDirectory.build(
        layout, ["ABS(5)", "DELTA(1)", "DELTA(0)", "DELTA(0)"]
    )
    assert directory.used_wells == ("A1", "A2", "A3", "A4")
    assert directory.to_measurements() == {"A1": F(5), "A2": F(6), "A3": F(6), "A4": F(6)}
    assert [r.wells for r in directory.ranges] == [("A1",), ("A2",), ("A3",), ("A4",)]
    assert directory.stream_digest == directory.to_dict()["stream_digest"]


def test_repeat_produces_multiple_wells_in_one_command():
    layout = blank_layout("P", 4)
    directory = CommandDirectory.build(layout, ["ABS(7)", "REPEAT(3)"])
    assert directory.to_measurements() == {"A1": F(7), "A2": F(7), "A3": F(7), "A4": F(7)}
    repeat_range = directory.ranges[1]
    assert repeat_range.op == "REPEAT"
    assert repeat_range.start == 1 and repeat_range.end == 4
    assert repeat_range.wells == ("A2", "A3", "A4")


def test_gap_marks_missing_and_clears_memory():
    layout = blank_layout("P", 5)
    directory = CommandDirectory.build(
        layout, ["ABS(1)", "GAP(1)", "ABS(9)", "DELTA(1)", "REPEAT(1)"]
    )
    assert directory.missing == ("A2",)
    assert "A2" not in directory.to_measurements()
    assert directory.to_measurements() == {"A1": F(1), "A3": F(9), "A4": F(10), "A5": F(10)}
    # Memory is cleared by GAP, then rebuilt by ABS before the DELTA.
    assert directory.boundary_memory[2] is None  # before ABS(9)
    assert directory.memory_after[1] is None  # after GAP
    assert directory.boundary_memory[3] == F(9)  # before DELTA(1)


@pytest.mark.parametrize(
    "stream",
    [
        ["DELTA(1)"],  # no initial memory
        ["REPEAT(2)"],  # no initial memory
        ["ABS(1)", "GAP(1)", "DELTA(1)", "ABS(1)", "ABS(1)"],  # DELTA after GAP
        ["ABS(1)", "GAP(1)", "REPEAT(1)", "ABS(1)", "ABS(1)"],  # REPEAT after GAP
    ],
)
def test_malformed_streams_reject_no_memory(stream):
    layout = blank_layout("P", 5)
    with pytest.raises(StreamDecodeError):
        CommandDirectory.build(layout, stream)


def test_out_of_bounds_command_rejected():
    layout = blank_layout("P", 3)
    with pytest.raises(StreamDecodeError):
        CommandDirectory.build(layout, ["ABS(1)", "REPEAT(5)"])


def test_too_few_wells_rejected():
    layout = blank_layout("P", 4)
    with pytest.raises(StreamDecodeError):
        CommandDirectory.build(layout, ["ABS(1)", "DELTA(1)"])


def test_trailing_commands_rejected():
    layout = blank_layout("P", 2)
    with pytest.raises(StreamDecodeError):
        CommandDirectory.build(layout, ["ABS(1)", "DELTA(1)", "DELTA(1)"])


def test_empty_stream_rejected_when_wells_used():
    layout = blank_layout("P", 2)
    with pytest.raises(StreamDecodeError):
        CommandDirectory.build(layout, [])


def test_only_unused_wells_needs_empty_stream():
    design = PlateDesign("P", rows=1, cols=2)  # all unused
    layout = design.publish()
    directory = CommandDirectory.build(layout, [])
    assert directory.used_wells == ()
    assert directory.commands == ()
    # An empty stream ships a valid, empty package and answers no queries.
    viewer = LookupViewer(frequencies={}, max_pages=2)
    plan = viewer.receive(directory)
    assert plan.total_weighted_cost == 0
    assert plan.pages_used == 0
    assert viewer.package.snapshot_boundaries == ()
    # Any well is not a used well of this stream.
    with pytest.raises(RetrievalError):
        LookupViewer(frequencies={"A1": 1}, max_pages=2).receive(directory)


# ---------------------------------------------------------------------------
# Warehouse integration: decoded run reuses the coordinate import path
# ---------------------------------------------------------------------------


def test_compressed_import_flows_through_warehouse():
    layout = blank_layout("P", 4)
    warehouse = ReadoutWarehouse()
    run = warehouse.import_compressed(
        layout, ["ABS(5)", "DELTA(1)", "DELTA(0)", "DELTA(0)"], run_id="c1"
    )
    report = warehouse.fit(run)
    assert report.verification["ok"]
    assert run.measurements == {"A1": F(5), "A2": F(6), "A3": F(6), "A4": F(6)}
    directory = warehouse.directory_for(run)
    assert directory.stream_digest == CommandDirectory.build(
        layout, ["ABS(5)", "DELTA(1)", "DELTA(0)", "DELTA(0)"]
    ).stream_digest


def test_gap_wells_become_missing_in_warehouse():
    layout = blank_layout("P", 5)
    warehouse = ReadoutWarehouse()
    run = warehouse.import_compressed(
        layout, ["ABS(1)", "GAP(1)", "ABS(9)", "DELTA(1)", "REPEAT(1)"], run_id="g"
    )
    report = warehouse.fit(run)
    assert report.well("A2").status == "MISSING"
    assert run.missing_used_wells == ("A2",)


def test_warehouse_rejects_malformed_compressed_stream():
    layout = blank_layout("P", 4)
    warehouse = ReadoutWarehouse()
    with pytest.raises(CompressedStreamError):
        warehouse.import_compressed(layout, ["DELTA(1)"])
    # A rejected decode consumes no run id.
    assert warehouse.runs() == ()


# ---------------------------------------------------------------------------
# Required worked examples
# ---------------------------------------------------------------------------


def test_example_basic_one_page_before_command_four():
    payload = example_basic()
    assert payload["allocation"]["selected_boundaries"] == [1, 4]
    assert payload["allocation"]["pages_used"] == 1
    assert payload["allocation"]["total_weighted_cost"] == 3
    by_well = {q["well"]: q for q in payload["allocation"]["queries"]}
    assert by_well["A2"]["commands"] == [1, 2]
    assert by_well["A2"]["commands_read"] == 2
    assert by_well["A4"]["commands"] == [4]
    assert by_well["A4"]["commands_read"] == 1


def test_example_heavy_a2_moves_page_and_costs_six():
    payload = example_heavy_a2()
    assert payload["allocation"]["selected_boundaries"] == [1, 2]
    assert payload["allocation"]["total_weighted_cost"] == 6
    by_well = {q["well"]: q for q in payload["allocation"]["queries"]}
    assert by_well["A2"]["commands"] == [2]
    assert by_well["A2"]["weighted_cost"] == 3
    assert by_well["A4"]["commands"] == [2, 3, 4]
    assert by_well["A4"]["weighted_cost"] == 3


def test_example_repeat_cost_three_tie_to_boundary_two():
    payload = example_repeat()
    assert payload["allocation"]["selected_boundaries"] == [1, 2]
    assert payload["allocation"]["pages_used"] == 1
    assert payload["allocation"]["total_weighted_cost"] == 3
    by_well = {q["well"]: q for q in payload["allocation"]["queries"]}
    # A4 is the last well of command 3 and must read the whole command.
    assert by_well["A4"]["covering_command"] == 3
    assert by_well["A4"]["commands"] == [2, 3]


def test_example_gap_recovery():
    payload = example_gap_recovery()
    by_well = {q["well"]: q for q in payload["allocation"]["queries"]}
    assert by_well["A2"]["missing"] is True
    assert by_well["A2"]["value"] is None
    assert by_well["A5"]["value"]["exact"] == "10"


def test_example_multi_well_command_middle_not_free():
    payload = example_multi_well_command()
    allocation = allocate(
        CommandDirectory.build(blank_layout("P-multi", 4), ["ABS(7)", "REPEAT(3)"]),
        {"A3": 1, "A4": 1},
        1,
    )
    # Both covering command 2 reads the whole command.
    for query in allocation.queries:
        assert query.covering_command == 2
        assert query.commands == (2,)
        assert query.commands_read == 1
    # A middle-of-REPEAT "free" start at boundary 3 is impossible: no page
    # inside a command can be selected, and querying A3 still reads command 2.
    directory = CommandDirectory.build(blank_layout("P-multi", 4), ["ABS(7)", "REPEAT(3)"])
    plan = allocate(directory, {"A3": 1}, 1)
    package = ship(directory, plan)
    viewer = LookupViewer(frequencies={"A3": 1}, max_pages=1)
    viewer.receive_package(package)
    with pytest.raises(RecoveryError):
        viewer.query("A3", from_boundary=3)  # no snapshot inside the command


def test_example_zero_pages_starts_at_free_head():
    payload = example_zero_pages()
    assert payload["allocation"]["selected_boundaries"] == [1]
    assert payload["allocation"]["pages_used"] == 0
    assert payload["allocation"]["total_weighted_cost"] == 6
    by_well = {q["well"]: q for q in payload["allocation"]["queries"]}
    assert by_well["A2"]["commands"] == [1, 2]
    assert by_well["A4"]["commands"] == [1, 2, 3, 4]
    # Every query starts at the free head boundary 1.
    for query in payload["allocation"]["queries"]:
        assert query["start_boundary"] == 1


# ---------------------------------------------------------------------------
# Viewer behaviour
# ---------------------------------------------------------------------------


def test_viewer_replays_and_matches_plan():
    layout = blank_layout("P", 4)
    directory = CommandDirectory.build(
        layout, ["ABS(5)", "DELTA(1)", "DELTA(0)", "DELTA(0)"]
    )
    viewer = LookupViewer(frequencies={"A2": 1, "A4": 1}, max_pages=1)
    plan = viewer.receive(directory)
    assert plan.total_weighted_cost == 3
    assert viewer.verify()["ok"]
    a2 = viewer.query("A2")
    assert a2.value == F(6) and a2.commands == (1, 2)
    a4 = viewer.query("A4")
    assert a4.value == F(6) and a4.commands == (4,)


def test_viewer_rejects_unknown_well():
    layout = blank_layout("P", 4)
    directory = CommandDirectory.build(
        layout, ["ABS(5)", "DELTA(1)", "DELTA(0)", "DELTA(0)"]
    )
    viewer = LookupViewer(frequencies={"A2": 1}, max_pages=1)
    viewer.receive(directory)
    with pytest.raises(RetrievalError):
        viewer.query("B9")


def test_frequency_validation():
    for bad in ({"A1": 0}, {"A1": -2}, {"A1": 1.5}, {"A1": True}, {1: 2}):
        with pytest.raises(RetrievalError):
            LookupViewer(frequencies=bad, max_pages=1)


def test_page_quota_validation():
    for bad in (-1, 1.5, True, "2"):
        with pytest.raises(RetrievalError):
            LookupViewer(frequencies={"A1": 1}, max_pages=bad)


def test_changing_load_reoptimises():
    layout = blank_layout("P", 4)
    directory = CommandDirectory.build(
        layout, ["ABS(5)", "DELTA(1)", "DELTA(0)", "DELTA(0)"]
    )
    viewer = LookupViewer(frequencies={"A2": 1, "A4": 1}, max_pages=1)
    plan1 = viewer.receive(directory)
    assert plan1.selected_boundaries == (1, 4)
    viewer.update_frequencies({"A2": 3, "A4": 1})
    plan2 = viewer.receive(directory)
    assert plan2.selected_boundaries == (1, 2)
    assert plan2.total_weighted_cost == 6


def test_different_directory_reoptimises_independently():
    layout = blank_layout("P", 4)
    cd1 = CommandDirectory.build(layout, ["ABS(5)", "DELTA(1)", "DELTA(0)", "DELTA(0)"])
    cd2 = CommandDirectory.build(layout, ["ABS(5)", "DELTA(1)", "REPEAT(2)"])
    viewer = LookupViewer(frequencies={"A2": 1, "A4": 1}, max_pages=1)
    assert viewer.receive(cd1).selected_boundaries == (1, 4)
    plan2 = viewer.receive(cd2)
    assert plan2.selected_boundaries == (1, 2)
    assert plan2.total_weighted_cost == 3


def test_query_beyond_snapshot_uses_latest_available():
    layout = blank_layout("P", 4)
    directory = CommandDirectory.build(
        layout, ["ABS(5)", "DELTA(1)", "DELTA(0)", "DELTA(0)"]
    )
    viewer = LookupViewer(frequencies={"A2": 1, "A4": 1}, max_pages=1)
    viewer.receive(directory)
    # Boundary 4 is dispensed; explicitly recovering from boundary 4 for A4.
    result = viewer.query("A4", from_boundary=4)
    assert result.start_boundary == 4 and result.commands == (4,)
    with pytest.raises(RecoveryError):
        viewer.query("A2", from_boundary=4)  # recovery point after target
    with pytest.raises(RecoveryError):
        viewer.query("A2", from_boundary=2)  # boundary 2 not dispensed


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------


def test_stale_snapshot_from_another_stream_rejected():
    layout = blank_layout("P", 4)
    cd1 = CommandDirectory.build(layout, ["ABS(5)", "DELTA(1)", "DELTA(0)", "DELTA(0)"])
    cd2 = CommandDirectory.build(layout, ["ABS(5)", "DELTA(1)", "DELTA(1)", "DELTA(1)"])
    plan1 = allocate(cd1, {"A2": 1, "A4": 1}, 1)
    with pytest.raises(StaleSnapshotError):
        ship(cd2, plan1)  # plan belongs to cd1


def test_package_digest_binding():
    layout = blank_layout("P", 4)
    cd1 = CommandDirectory.build(layout, ["ABS(5)", "DELTA(1)", "DELTA(0)", "DELTA(0)"])
    cd2 = CommandDirectory.build(layout, ["ABS(5)", "DELTA(1)", "DELTA(1)", "DELTA(1)"])
    plan1 = allocate(cd1, {"A2": 1, "A4": 1}, 1)
    package = ship(cd1, plan1)
    # A package can never be paired with a different stream's plan.
    assert package.stream_digest == cd1.stream_digest != cd2.stream_digest
    viewer = LookupViewer(frequencies={"A2": 1}, max_pages=1)
    viewer.receive_package(ShippedPackage.from_dict(package.to_dict()))
    assert viewer.query("A2").value == F(6)


def test_shipped_package_carries_no_decoded_values():
    layout = blank_layout("P", 4)
    directory = CommandDirectory.build(
        layout, ["ABS(5)", "DELTA(1)", "DELTA(0)", "DELTA(0)"]
    )
    plan = allocate(directory, {"A2": 1, "A4": 1}, 1)
    package = ship(directory, plan)
    payload = package.to_dict()
    assert set(payload) == {
        "layout_digest",
        "stream_digest",
        "commands",
        "ranges",
        "snapshots",
    }
    # No well->value mapping is shipped; the device must replay.
    assert "decoded" not in json.dumps(payload)
    # Only the free head plus the selected paid page are shipped.
    assert package.snapshot_boundaries == (1, 4)


# ---------------------------------------------------------------------------
# Allocation optimality / tie-breaking
# ---------------------------------------------------------------------------


def brute_force_cost(directory, frequencies, max_pages):
    """Exhaustive best ``(cost, pages, selected_boundaries)`` over all plans."""

    import itertools

    n = len(directory.commands)
    if n == 0:
        return (0, 0, ())
    command_of = {
        well: entry.command_index for entry in directory.ranges for well in entry.wells
    }
    load = {well: count for well, count in frequencies.items()}
    best = None
    for pages in range(0, min(max_pages, n - 1) + 1):
        for extra in itertools.combinations(range(2, n + 1), pages):
            selected = (1, *extra)
            total = 0
            for well, count in load.items():
                covering = command_of[well]
                start = max(b for b in selected if b <= covering)
                total += count * (covering - start + 1)
            # Rank by cost, then pages, then lexicographic selected list.
            key = (total, pages, selected)
            if best is None or key < best:
                best = key
    return best


@pytest.mark.parametrize("max_pages", [0, 1, 2, 3])
def test_allocation_matches_brute_force(max_pages):
    layout = blank_layout("P", 6)
    directory = CommandDirectory.build(
        layout,
        ["ABS(1)", "DELTA(1)", "REPEAT(2)", "GAP(1)", "ABS(3)"],
    )
    frequencies = {"A1": 2, "A2": 1, "A4": 3, "A5": 2, "A6": 1}
    plan = allocate(directory, frequencies, max_pages)
    best_cost, best_pages, best_selected = brute_force_cost(
        directory, frequencies, max_pages
    )
    assert plan.total_weighted_cost == best_cost
    assert plan.pages_used == best_pages
    assert plan.selected_boundaries == best_selected
    # Pages never exceed the quota or the number of selectable boundaries.
    assert plan.pages_used <= max_pages


@pytest.mark.parametrize("max_pages", [0, 1, 2, 3, 4, 5])
def test_allocation_lexicographic_tiebreak_matches_brute_force(max_pages):
    layout = blank_layout("P", 6)
    # Many equal-weight commands produce frequent exact ties.
    directory = CommandDirectory.build(
        layout, ["ABS(0)"] + ["DELTA(0)"] * 5
    )
    frequencies = {"A1": 1, "A2": 1, "A3": 1, "A4": 1, "A5": 1, "A6": 1}
    plan = allocate(directory, frequencies, max_pages)
    best_cost, best_pages, best_selected = brute_force_cost(
        directory, frequencies, max_pages
    )
    assert plan.total_weighted_cost == best_cost
    assert plan.pages_used == best_pages
    assert plan.selected_boundaries == best_selected


def test_tie_breaking_fewest_pages_then_lexicographic():
    # Every query is on command 1, so no paid page helps: the optimum is zero
    # pages, regardless of a generous quota.
    layout = blank_layout("P", 4)
    directory = CommandDirectory.build(
        layout, ["ABS(5)", "DELTA(1)", "DELTA(0)", "DELTA(0)"]
    )
    plan = allocate(directory, {"A1": 5}, 3)
    assert plan.pages_used == 0
    assert plan.selected_boundaries == (1,)

    # A genuine tie between boundaries 2 and 3: choose the lexicographically
    # smaller boundary list [1, 2].
    cd_repeat = CommandDirectory.build(layout, ["ABS(5)", "DELTA(1)", "REPEAT(2)"])
    plan2 = allocate(cd_repeat, {"A2": 1, "A4": 1}, 1)
    assert plan2.selected_boundaries == (1, 2)
    assert plan2.total_weighted_cost == 3


def test_multiple_pages_improve_cost():
    layout = blank_layout("P", 5)
    # Distinct values through five commands.
    directory = CommandDirectory.build(
        layout, ["ABS(1)", "DELTA(1)", "DELTA(1)", "DELTA(1)", "DELTA(1)"]
    )
    frequencies = {"A2": 1, "A3": 1, "A4": 1, "A5": 1}
    one_page = allocate(directory, frequencies, 1)
    two_page = allocate(directory, frequencies, 2)
    assert two_page.total_weighted_cost <= one_page.total_weighted_cost
    assert two_page.pages_used <= 2
    # With enough pages each well starts at its own covering command.
    ample = allocate(directory, frequencies, 10)
    for query in ample.queries:
        assert query.commands_read == 1
    assert ample.total_weighted_cost == 4


# ---------------------------------------------------------------------------
# Replayable file examples
# ---------------------------------------------------------------------------


def test_all_examples_roundtrip_and_replay():
    examples = build_examples()
    assert set(examples) == {
        "basic",
        "heavy_a2",
        "repeat",
        "gap_recovery",
        "multi_well_command",
        "zero_pages",
    }
    for name, payload in examples.items():
        restored = json.loads(json.dumps(payload))
        replay = replay_example(restored)
        assert replay["stream_digest"] == restored["directory"]["stream_digest"]
        planned = {q["well"]: q for q in restored["allocation"]["queries"]}
        for result in replay["wells"]:
            expected = planned[result["well"]]
            assert result["start_boundary"] == expected["start_boundary"]
            assert result["commands"] == expected["commands"]
            assert result["missing"] == expected["missing"]
            assert result["value"] == expected["value"]


def test_directory_roundtrip_matches_decode():
    layout = blank_layout("P", 5)
    directory = CommandDirectory.build(
        layout, ["ABS(1)", "GAP(1)", "ABS(9)", "DELTA(1)", "REPEAT(1)"]
    )
    payload = directory.to_dict()
    restored = json.loads(json.dumps(payload))
    assert restored["decoded"] == {
        well: _exact(value) for well, value in directory.to_measurements().items()
    }
    assert restored["missing"] == list(directory.missing)
    assert restored["stream_digest"] == directory.stream_digest


def _exact(value):
    from plate_effects.rational import fmt_rational

    return fmt_rational(value)


def test_supports_up_to_96_wells():
    design = PlateDesign("P-96", rows=8, cols=12, min_replicates=1, max_replicates=1)
    for row in range(8):
        for col in range(12):
            design.set_blank([f"{chr(ord('A') + row)}{col + 1}"])
    layout = design.publish()
    tokens = ["ABS(0)"] + ["DELTA(1)"] * 95  # exactly 96 single-well commands
    directory = CommandDirectory.build(layout, tokens)
    assert len(directory.used_wells) == 96
    frequencies = {
        f"{chr(ord('A') + row)}{col + 1}": (row * col % 5) + 1
        for row in range(8)
        for col in range(12)
    }
    viewer = LookupViewer(frequencies=frequencies, max_pages=10)
    plan = viewer.receive(directory)
    assert plan.pages_used <= 10
    assert viewer.verify()["ok"]
