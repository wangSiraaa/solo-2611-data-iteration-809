"""Snapshot allocation and the handheld lookup device.

Model
-----
A :class:`~plate_effects.compressed.CommandDirectory` indexes the original
command stream by *command boundary*: boundary ``b`` means "before command
``b``" (1-based).  The stream head (boundary 1) is always available and free.
Each other boundary may optionally hold one fixed-size recovery snapshot that
records the memory state immediately before that command; at most ``max_pages``
snapshots may be taken.

A query is for one well.  The well is covered by exactly one command ``q`` (a
``REPEAT`` can cover several wells).  An independent query starts at the latest
dispensed boundary ``b <= q`` and reads commands ``b, b+1, ..., q`` -- the whole
covering command is always read, so a point *inside* a ``REPEAT`` is never a
free start.  Its cost is the number of commands actually read, ``q - b + 1``.
The load is a table of positive integer access frequencies; the objective is to
minimise the sum of ``frequency * commands_read`` over all queries.

Allocation minimises weighted cost first, then (among optimal plans) the number
of pages used, then the ascending list of selected 1-based command indices
lexicographically.  The complete provenance (decoded values, memory states,
etc.) stays at the dispatch end: :func:`ship` builds a
:class:`ShippedPackage` containing only the original stream, the command -> well
range index and the selected snapshots.  A :class:`LookupViewer` must actually
replay the original stream from the designated recovery point.

Every snapshot and package carries the stream digest, so a snapshot minted for
one stream can never be used to answer a query against another stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

from .compressed import (
    OP_ABS,
    OP_DELTA,
    OP_GAP,
    OP_REPEAT,
    Command,
    CommandDirectory,
    CommandRange,
)
from .rational import fmt_rational

_INF = 10**18


class RetrievalError(ValueError):
    """Base class for allocation / lookup errors."""


class StaleSnapshotError(RetrievalError):
    """A snapshot or package belongs to a different stream."""


class RecoveryError(RetrievalError):
    """The original stream could not be replayed from a recovery point."""


# ---------------------------------------------------------------------------
# Frequencies and quota validation
# ---------------------------------------------------------------------------


def _normalise_frequencies(frequencies: Mapping[str, object]) -> dict[str, int]:
    if not isinstance(frequencies, Mapping):
        raise RetrievalError("access frequencies must be a mapping well -> positive int")
    result: dict[str, int] = {}
    for well, count in frequencies.items():
        if not isinstance(well, str) or not well:
            raise RetrievalError(f"invalid well key {well!r} in access frequencies")
        if isinstance(count, bool) or not isinstance(count, int):
            raise RetrievalError(
                f"access frequency for {well} must be a positive integer, got {count!r}"
            )
        if count < 1:
            raise RetrievalError(
                f"access frequency for {well} must be positive, got {count}"
            )
        result[well] = count
    return result


def _normalise_pages(max_pages: object) -> int:
    if isinstance(max_pages, bool) or not isinstance(max_pages, int):
        raise RetrievalError("page quota must be a non-negative integer")
    if max_pages < 0:
        raise RetrievalError("page quota must be non-negative")
    return max_pages


# ---------------------------------------------------------------------------
# Plan records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryPlan:
    """Predicted recovery path and cost for one queried well."""

    well: str
    frequency: int
    covering_command: int
    start_boundary: int
    commands: tuple[int, ...]
    commands_read: int
    weighted_cost: int
    value: Fraction | None
    missing: bool

    def to_dict(self) -> dict:
        return {
            "well": self.well,
            "frequency": self.frequency,
            "covering_command": self.covering_command,
            "start_boundary": self.start_boundary,
            "commands": list(self.commands),
            "commands_read": self.commands_read,
            "weighted_cost": self.weighted_cost,
            "missing": self.missing,
            "value": None
            if self.value is None
            else {"exact": fmt_rational(self.value), "float": float(self.value)},
        }


@dataclass(frozen=True)
class AllocationPlan:
    """Server-side optimum for one (directory, load, quota) triple."""

    layout_digest: str
    stream_digest: str
    max_pages: int
    selected_boundaries: tuple[int, ...]  # includes the free boundary 1
    paid_boundaries: tuple[int, ...]  # selected boundaries excluding 1
    pages_used: int
    total_weighted_cost: int
    command_weights: tuple[tuple[int, int], ...]  # (command index, weight)
    queries: tuple[QueryPlan, ...]

    def to_dict(self) -> dict:
        return {
            "layout_digest": self.layout_digest,
            "stream_digest": self.stream_digest,
            "max_pages": self.max_pages,
            "selected_boundaries": list(self.selected_boundaries),
            "paid_boundaries": list(self.paid_boundaries),
            "pages_used": self.pages_used,
            "total_weighted_cost": self.total_weighted_cost,
            "command_weights": [
                {"command": index, "weight": weight}
                for index, weight in self.command_weights
            ],
            "queries": [query.to_dict() for query in self.queries],
        }


# ---------------------------------------------------------------------------
# Allocation (optimal segmentation of the command sequence)
# ---------------------------------------------------------------------------


def _command_weights(
    directory: CommandDirectory, frequencies: Mapping[str, int]
) -> list[int]:
    n_commands = len(directory.commands)
    weights = [0] * (n_commands + 1)  # 1-based
    # Map each well to its covering command via the range index.
    command_of: dict[str, int] = {}
    for entry in directory.ranges:
        for well in entry.wells:
            command_of[well] = entry.command_index
    for well, count in frequencies.items():
        if well not in command_of:
            raise RetrievalError(
                f"queried well {well!r} is not a used well of this stream"
            )
        weights[command_of[well]] += count
    return weights


def _make_cost_fn(weights: Sequence[int], n_commands: int):
    # f(a, b) = sum_{q=a..b} weights[q] * (q - a + 1), 0 when a > b.
    prefix0 = [0] * (n_commands + 1)
    prefix1 = [0] * (n_commands + 1)
    for q in range(1, n_commands + 1):
        prefix0[q] = prefix0[q - 1] + weights[q]
        prefix1[q] = prefix1[q - 1] + weights[q] * (q + 1)

    def f(a: int, b: int) -> int:
        if a > b:
            return 0
        total_weight = prefix0[b] - prefix0[a - 1]
        total_weighted = prefix1[b] - prefix1[a - 1]
        return total_weighted - a * total_weight

    return f


def _allocation_interval(
    directory: CommandDirectory,
    frequencies: Mapping[str, int],
    max_pages: int,
) -> tuple[int, tuple[int, ...]]:
    """Return ``(total_cost, selected_boundaries)`` for the optimum."""

    n_commands = len(directory.commands)
    if n_commands == 0:
        return 0, ()
    weights = _command_weights(directory, frequencies)
    f = _make_cost_fn(weights, n_commands)

    # Most segments we would ever need: one per command, and one more than the
    # page budget.  The free head counts as a segment but not a page.
    max_segments = min(max_pages + 1, n_commands)

    # cover[k][i] = minimal cost covering commands 1..i with exactly k segments.
    cover: list[list[int]] = [[_INF] * (n_commands + 1) for _ in range(max_segments + 1)]
    for i in range(1, n_commands + 1):
        cover[1][i] = f(1, i)
    for k in range(2, max_segments + 1):
        for i in range(k, n_commands + 1):
            best = _INF
            for s in range(k, i + 1):
                prev = cover[k - 1][s - 1]
                if prev >= _INF:
                    continue
                candidate = prev + f(s, i)
                if candidate < best:
                    best = candidate
            cover[k][i] = best

    best_cost = _INF
    best_k = 1
    for k in range(1, max_segments + 1):
        cost = cover[k][n_commands]
        if cost < best_cost or (cost == best_cost and k < best_k):
            best_cost = cost
            best_k = k
    if best_cost >= _INF:  # pragma: no cover - infeasible only on empty input
        raise RetrievalError("no feasible allocation")

    # Suffix costs for lexicographic reconstruction.
    # suf[t][s] = minimal cost covering commands s..m with t segments, first
    # starting at s.
    suf = [[_INF] * (n_commands + 1) for _ in range(best_k + 1)]
    for s in range(1, n_commands + 1):
        suf[1][s] = f(s, n_commands)
    for t in range(2, best_k + 1):
        for s in range(1, n_commands + 1):
            best = _INF
            max_end = n_commands - (t - 1)
            for e in range(s, max_end + 1):
                tail = suf[t - 1][e + 1] if e + 1 <= n_commands else _INF
                if tail >= _INF:
                    continue
                candidate = f(s, e) + tail
                if candidate < best:
                    best = candidate
            suf[t][s] = best

    selected = [1]
    if best_k > 1:
        need = best_cost
        prev = 1
        for t_segments in range(best_k, 1, -1):
            chosen = None
            for b in range(prev + 1, n_commands + 1):
                tail = suf[t_segments - 1][b]
                if tail >= _INF:
                    continue
                if f(prev, b - 1) + tail == need:
                    chosen = b
                    break
            if chosen is None:  # pragma: no cover - would be an internal error
                raise RetrievalError("failed to reconstruct optimal allocation")
            selected.append(chosen)
            need -= f(prev, chosen - 1)
            prev = chosen
        if f(prev, n_commands) != need:  # pragma: no cover - internal check
            raise RetrievalError("reconstruction cost mismatch")

    return best_cost, tuple(selected)


def allocate(
    directory: CommandDirectory,
    frequencies: Mapping[str, object],
    max_pages: int,
) -> AllocationPlan:
    """Compute the optimal snapshot allocation for one load.

    Feeding a different directory or a different load simply recomputes the
    optimum for that input; no state is shared between calls.
    """

    frequency_table = _normalise_frequencies(frequencies)
    pages = _normalise_pages(max_pages)
    total_cost, selected = _allocation_interval(directory, frequency_table, pages)

    weights = _command_weights(directory, frequency_table) if directory.commands else [0]
    paid = tuple(b for b in selected if b != 1)
    command_of: dict[str, int] = {}
    for entry in directory.ranges:
        for well in entry.wells:
            command_of[well] = entry.command_index
    for well in frequency_table:
        if well not in command_of:
            raise RetrievalError(
                f"queried well {well!r} is not a used well of this stream"
            )
    decoded = dict(directory.decoded)
    selected_set = sorted(selected)

    queries: list[QueryPlan] = []
    for well in sorted(frequency_table):
        frequency = frequency_table[well]
        covering = command_of[well]
        start = max(b for b in selected_set if b <= covering)
        path = tuple(range(start, covering + 1))
        value = decoded.get(well)
        missing = well in directory.missing
        if missing:
            value = None
        queries.append(
            QueryPlan(
                well=well,
                frequency=frequency,
                covering_command=covering,
                start_boundary=start,
                commands=path,
                commands_read=len(path),
                weighted_cost=frequency * len(path),
                value=value,
                missing=missing,
            )
        )

    return AllocationPlan(
        layout_digest=directory.layout_digest,
        stream_digest=directory.stream_digest,
        max_pages=pages,
        selected_boundaries=tuple(selected_set),
        paid_boundaries=paid,
        pages_used=len(paid),
        total_weighted_cost=total_cost,
        command_weights=tuple(
            (index, weights[index])
            for index in range(1, len(weights))
            if weights[index]
        ),
        queries=tuple(queries),
    )


# ---------------------------------------------------------------------------
# Shipped package (everything the device is allowed to hold)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    boundary: int  # 1-based command index
    memory: Fraction | None
    stream_digest: str

    def to_dict(self) -> dict:
        return {
            "boundary": self.boundary,
            "memory": None if self.memory is None else fmt_rational(self.memory),
            "stream_digest": self.stream_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "Snapshot":
        memory = payload.get("memory")
        return cls(
            boundary=int(payload["boundary"]),
            memory=None if memory is None else Fraction(str(memory)),
            stream_digest=str(payload["stream_digest"]),
        )


@dataclass(frozen=True)
class ShippedPackage:
    """Only the original stream, the range index and the selected snapshots."""

    layout_digest: str
    stream_digest: str
    commands: tuple[Command, ...]
    ranges: tuple[CommandRange, ...]
    snapshots: tuple[Snapshot, ...]

    def __post_init__(self) -> None:
        n = len(self.commands)
        if n == 0:
            # An all-unused plate has no stream and no queryable well; shipping
            # an empty package is valid and the head rule is vacuous.
            if self.snapshots or self.ranges:
                raise StaleSnapshotError(
                    "an empty stream must ship no snapshots and no ranges"
                )
            return
        if not self.snapshots:
            raise StaleSnapshotError("package must contain at least the free head snapshot")
        boundaries: set[int] = set()
        for snapshot in self.snapshots:
            if snapshot.stream_digest != self.stream_digest:
                raise StaleSnapshotError(
                    "snapshot stream digest does not match package stream digest"
                )
            if not (1 <= snapshot.boundary <= n):
                raise StaleSnapshotError(
                    f"snapshot boundary {snapshot.boundary} outside 1..{n}"
                )
            if snapshot.boundary in boundaries:
                raise StaleSnapshotError(f"duplicate snapshot boundary {snapshot.boundary}")
            boundaries.add(snapshot.boundary)
        if 1 not in boundaries:
            raise StaleSnapshotError("package is missing the free stream-head snapshot")
        head = next(s for s in self.snapshots if s.boundary == 1)
        if head.memory is not None:
            raise StaleSnapshotError("the stream head has no remembered value")
        for entry in self.ranges:
            if not (1 <= entry.command_index <= n):
                raise StaleSnapshotError("range index references an unknown command")

    @property
    def snapshot_boundaries(self) -> tuple[int, ...]:
        return tuple(sorted(s.boundary for s in self.snapshots))

    def snapshot_at_or_before(self, command: int) -> Snapshot:
        candidates = [s for s in self.snapshots if s.boundary <= command]
        if not candidates:
            raise RecoveryError(f"no recovery point at or before command {command}")
        return max(candidates, key=lambda s: s.boundary)

    def to_dict(self) -> dict:
        return {
            "layout_digest": self.layout_digest,
            "stream_digest": self.stream_digest,
            "commands": [c.to_dict() for c in self.commands],
            "ranges": [r.to_dict() for r in self.ranges],
            "snapshots": [s.to_dict() for s in self.snapshots],
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "ShippedPackage":
        from .compressed import Command, CommandRange

        return cls(
            layout_digest=str(payload["layout_digest"]),
            stream_digest=str(payload["stream_digest"]),
            commands=tuple(Command.from_dict(c) for c in payload["commands"]),
            ranges=tuple(CommandRange.from_dict(r) for r in payload["ranges"]),
            snapshots=tuple(Snapshot.from_dict(s) for s in payload["snapshots"]),
        )


def ship(directory: CommandDirectory, plan: AllocationPlan) -> ShippedPackage:
    """Build the minimal package for ``plan``; rejects stale plans/snapshots."""

    if plan.stream_digest != directory.stream_digest:
        raise StaleSnapshotError("plan was built for a different stream")
    if plan.layout_digest != directory.layout_digest:
        raise StaleSnapshotError("plan was built for a different layout")
    snapshots: list[Snapshot] = []
    for boundary in plan.selected_boundaries:
        if not (1 <= boundary <= len(directory.commands)):
            raise StaleSnapshotError(
                f"selected boundary {boundary} outside stream bounds"
            )
        memory = directory.boundary_memory[boundary - 1]
        snapshots.append(
            Snapshot(boundary=boundary, memory=memory, stream_digest=directory.stream_digest)
        )
    return ShippedPackage(
        layout_digest=directory.layout_digest,
        stream_digest=directory.stream_digest,
        commands=directory.commands,
        ranges=directory.ranges,
        snapshots=tuple(snapshots),
    )


# ---------------------------------------------------------------------------
# Handheld lookup device
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryResult:
    well: str
    frequency: int | None
    covering_command: int
    start_boundary: int
    commands: tuple[int, ...]
    commands_read: int
    weighted_cost: int | None
    value: Fraction | None
    missing: bool

    def to_dict(self) -> dict:
        return {
            "well": self.well,
            "frequency": self.frequency,
            "covering_command": self.covering_command,
            "start_boundary": self.start_boundary,
            "commands": list(self.commands),
            "commands_read": self.commands_read,
            "weighted_cost": self.weighted_cost,
            "missing": self.missing,
            "value": None
            if self.value is None
            else {"exact": fmt_rational(self.value), "float": float(self.value)},
        }


class LookupViewer:
    """A handheld device owning its own access frequencies and page quota.

    ``receive(directory)`` asks the dispatch end for the optimal plan under the
    device's load and quota and accepts the shipped package.  ``query(well)``
    then replays the *original stream* from the latest dispensed boundary and
    returns the value or ``MISSING`` together with the commands it actually
    read.
    """

    def __init__(self, *, frequencies: Mapping[str, object], max_pages: int) -> None:
        self._frequencies = _normalise_frequencies(frequencies)
        self._max_pages = _normalise_pages(max_pages)
        self._package: ShippedPackage | None = None
        self._plan: AllocationPlan | None = None
        self._command_of: dict[str, int] = {}

    # -- load management -------------------------------------------------------
    @property
    def frequencies(self) -> dict[str, int]:
        return dict(self._frequencies)

    @property
    def max_pages(self) -> int:
        return self._max_pages

    @property
    def plan(self) -> AllocationPlan | None:
        return self._plan

    @property
    def package(self) -> ShippedPackage | None:
        return self._package

    def update_frequencies(self, frequencies: Mapping[str, object]) -> None:
        """Change the load; the next :meth:`receive` re-optimises for it."""

        self._frequencies = _normalise_frequencies(frequencies)

    def update_max_pages(self, max_pages: int) -> None:
        self._max_pages = _normalise_pages(max_pages)

    # -- dispatch handshake ----------------------------------------------------
    def plan_for(self, directory: CommandDirectory) -> AllocationPlan:
        return allocate(directory, self._frequencies, self._max_pages)

    def receive(self, directory: CommandDirectory) -> AllocationPlan:
        plan = self.plan_for(directory)
        package = ship(directory, plan)
        self._package = package
        self._plan = plan
        self._command_of = {
            well: entry.command_index
            for entry in package.ranges
            for well in entry.wells
        }
        return plan

    def receive_package(self, package: ShippedPackage) -> None:
        """Load a package restored from a file (e.g. on the handheld device).

        The device can answer queries but has no plan prediction; use
        :meth:`query` directly.  Stale packages (built for another stream) are
        still rejected by :class:`ShippedPackage`'s own digest checks.
        """

        if not isinstance(package, ShippedPackage):
            raise RetrievalError("receive_package expects a ShippedPackage")
        self._package = package
        self._plan = None
        self._command_of = {
            well: entry.command_index
            for entry in package.ranges
            for well in entry.wells
        }

    # -- queries ---------------------------------------------------------------
    def _require_package(self) -> ShippedPackage:
        if self._package is None:
            raise RecoveryError("no package received; call receive(directory) first")
        return self._package

    def query(self, well: str, *, from_boundary: int | None = None) -> QueryResult:
        package = self._require_package()
        if not isinstance(well, str) or well not in self._command_of:
            raise RetrievalError(f"well {well!r} is not a used well of this stream")
        covering = self._command_of[well]
        if from_boundary is None:
            snapshot = package.snapshot_at_or_before(covering)
        else:
            snapshot = self._snapshot_at(package, from_boundary)
            if snapshot.boundary > covering:
                raise RecoveryError(
                    f"recovery point {snapshot.boundary} is after covering command {covering}"
                )
        start = snapshot.boundary
        memory = snapshot.memory
        path: list[int] = []
        last_op = None
        for index in range(start, covering + 1):
            command = package.commands[index - 1]
            path.append(index)
            last_op = command.op
            if command.op == OP_ABS:
                memory = command.value
            elif command.op == OP_DELTA:
                if memory is None:
                    raise RecoveryError(
                        f"command {index} is DELTA with no remembered value"
                    )
                memory = memory + command.value
            elif command.op == OP_REPEAT:
                if memory is None:
                    raise RecoveryError(
                        f"command {index} is REPEAT with no remembered value"
                    )
                # memory is unchanged; the emitted value is the remembered one
            elif command.op == OP_GAP:
                memory = None
            else:  # pragma: no cover - Command guards this
                raise RecoveryError(f"unknown command op {command.op!r}")

        missing = last_op == OP_GAP
        value = None if missing else memory
        frequency = self._frequencies.get(well)
        return QueryResult(
            well=well,
            frequency=frequency,
            covering_command=covering,
            start_boundary=start,
            commands=tuple(path),
            commands_read=len(path),
            weighted_cost=None if frequency is None else frequency * len(path),
            value=value,
            missing=missing,
        )

    @staticmethod
    def _snapshot_at(package: ShippedPackage, boundary: int) -> Snapshot:
        if isinstance(boundary, bool) or not isinstance(boundary, int):
            raise RecoveryError("recovery point must be an integer command index")
        for snapshot in package.snapshots:
            if snapshot.boundary == boundary:
                return snapshot
        raise RecoveryError(
            f"no snapshot at boundary {boundary}; only command boundaries with a "
            "dispensed snapshot can be used as a recovery point"
        )

    # -- verification ----------------------------------------------------------
    def verify(self) -> dict:
        """Replay every queried well and compare with the plan's prediction."""

        if self._plan is None or self._package is None:
            raise RecoveryError("no plan/package to verify")
        checks = []
        ok = True
        for query in self._plan.queries:
            result = self.query(query.well)
            match = (
                result.start_boundary == query.start_boundary
                and result.commands == query.commands
                and result.commands_read == query.commands_read
                and result.missing == query.missing
                and result.value == query.value
            )
            ok = ok and match
            checks.append(
                {
                    "well": query.well,
                    "match": match,
                    "planned": query.to_dict(),
                    "replayed": result.to_dict(),
                }
            )
        return {
            "ok": ok,
            "stream_digest": self._package.stream_digest,
            "queries": checks,
        }
