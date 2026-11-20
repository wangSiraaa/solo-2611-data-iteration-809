"""Compressed command-stream input and the boundary directory.

A synthetic scanner encodes the *used* wells of one plate, in row-major order,
as a sequence of variable-length commands::

    ABS(x)     emit and remember the rational value x
    DELTA(d)   add d to the remembered value, then emit the new value
    REPEAT(k)  emit the remembered value k times in a row
    GAP(k)     mark k consecutive measurements missing, then clear memory

``k`` is always a positive integer.  There is no memory at the start of a
stream and ``GAP`` clears it, so ``DELTA`` or ``REPEAT`` with no memory is a
malformed stream.  A command is an *indivisible read unit*: decoding assigns
emitted values to the layout's used wells in row-major order and must consume
the stream exactly.  A command that would run past the last used well (out of
bounds), a stream that stops before every used well has been assigned (too few
wells decoded), and commands left over after the last used well has been
assigned (trailing commands) are all rejected.

Complete decoding yields a ``well -> Fraction`` mapping with ``GAP`` wells
simply absent.  That mapping is what the existing
:class:`~plate_effects.warehouse.ReadoutWarehouse` already accepts through its
normal coordinate import; the original coordinate path is untouched.

:class:`CommandDirectory` is the server-side, fully provenanced boundary
catalogue.  It binds the layout digest, the original stream, the expanded well
order, the command -> well range index and every boundary's memory state.  A
lookup device only receives :class:`~plate_effects.retrieval.ShippedPackage`,
which contains the original stream, the command -> well range index and the
selected snapshots -- never the decoded values.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

from .design import MAX_WELLS, ROLE_UNUSED, PublishedLayout
from .rational import RationalError, fmt_rational, parse_rational

OP_ABS = "ABS"
OP_DELTA = "DELTA"
OP_REPEAT = "REPEAT"
OP_GAP = "GAP"
OPS = (OP_ABS, OP_DELTA, OP_REPEAT, OP_GAP)

_COMMAND_RE = re.compile(r"^\s*([A-Za-z]+)\s*\(\s*([^()]*?)\s*\)\s*$")
_POSITIVE_INT_RE = re.compile(r"^\+?[0-9]+$")


class CompressedStreamError(ValueError):
    """Base class for compressed-stream errors."""


class CommandSyntaxError(CompressedStreamError):
    """A command token cannot be parsed."""


class StreamDecodeError(CompressedStreamError):
    """A stream does not decode against the used-well sequence."""


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Command:
    """One indivisible read unit.

    ``value`` is set for ``ABS``/``DELTA``; ``count`` is set for
    ``REPEAT``/``GAP``.
    """

    op: str
    value: Fraction | None = None
    count: int | None = None

    def __post_init__(self) -> None:
        if self.op not in OPS:
            raise CommandSyntaxError(f"unknown command op {self.op!r}")
        if self.op in (OP_ABS, OP_DELTA):
            if self.value is None:
                raise CommandSyntaxError(f"{self.op} requires a rational value")
            object.__setattr__(self, "value", parse_rational(self.value, field=self.op))
            if self.count is not None:
                raise CommandSyntaxError(f"{self.op} must not carry a count")
        else:
            if self.count is None:
                raise CommandSyntaxError(f"{self.op} requires a positive integer count")
            if isinstance(self.count, bool) or not isinstance(self.count, int):
                raise CommandSyntaxError(f"{self.op} count must be an integer")
            if self.count < 1:
                raise CommandSyntaxError(f"{self.op} count must be positive")
            if self.value is not None:
                raise CommandSyntaxError(f"{self.op} must not carry a value")

    @property
    def slots(self) -> int:
        """Number of used-well positions this command consumes."""

        if self.op in (OP_ABS, OP_DELTA):
            return 1
        return int(self.count)

    @property
    def emits_values(self) -> bool:
        return self.op != OP_GAP

    def text(self) -> str:
        if self.op in (OP_ABS, OP_DELTA):
            return f"{self.op}({fmt_rational(self.value)})"
        return f"{self.op}({self.count})"

    def to_dict(self) -> dict:
        if self.op in (OP_ABS, OP_DELTA):
            return {"op": self.op, "value": fmt_rational(self.value)}
        return {"op": self.op, "count": self.count}

    @classmethod
    def from_dict(cls, payload: Mapping) -> "Command":
        try:
            op = str(payload["op"]).upper()
        except (KeyError, TypeError) as exc:
            raise CommandSyntaxError(f"malformed command record {payload!r}") from exc
        if op in (OP_ABS, OP_DELTA):
            if "value" not in payload:
                raise CommandSyntaxError(f"{op} requires a 'value'")
            return cls(op, value=payload["value"])
        if op in (OP_REPEAT, OP_GAP):
            if "count" not in payload:
                raise CommandSyntaxError(f"{op} requires a 'count'")
            return cls(op, count=_coerce_positive_int(payload["count"], op))
        raise CommandSyntaxError(f"unknown command op {op!r}")


def _coerce_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise CommandSyntaxError(f"{field}: booleans are not counts")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        text = value.strip()
        if not _POSITIVE_INT_RE.match(text):
            raise CommandSyntaxError(f"{field}: invalid integer count {value!r}")
        result = int(text)
    else:
        raise CommandSyntaxError(f"{field}: invalid count type {type(value).__name__}")
    if result < 1:
        raise CommandSyntaxError(f"{field}: count must be positive, got {result}")
    return result


def parse_command(token: object) -> Command:
    """Parse one command from a string, mapping, pair or :class:`Command`."""

    if isinstance(token, Command):
        return token
    if isinstance(token, str):
        match = _COMMAND_RE.match(token)
        if not match:
            raise CommandSyntaxError(f"malformed command {token!r}")
        op = match.group(1).upper()
        arg = match.group(2)
        if op in (OP_ABS, OP_DELTA):
            try:
                return Command(op, value=parse_rational(arg, field=op))
            except RationalError as exc:
                raise CommandSyntaxError(str(exc)) from exc
        if op in (OP_REPEAT, OP_GAP):
            return Command(op, count=_coerce_positive_int(arg, op))
        raise CommandSyntaxError(f"unknown command op {op!r}")
    if isinstance(token, Mapping):
        return Command.from_dict(token)
    if isinstance(token, (list, tuple)) and len(token) == 2:
        op = str(token[0]).upper()
        if op in (OP_ABS, OP_DELTA):
            return Command(op, value=token[1])
        if op in (OP_REPEAT, OP_GAP):
            return Command(op, count=_coerce_positive_int(token[1], op))
        raise CommandSyntaxError(f"unknown command op {op!r}")
    raise CommandSyntaxError(f"unsupported command token {token!r}")


def parse_commands(tokens: Iterable[object]) -> tuple[Command, ...]:
    if isinstance(tokens, (str, bytes)):
        raise CommandSyntaxError("command stream must be a sequence of tokens")
    try:
        return tuple(parse_command(token) for token in tokens)
    except TypeError as exc:
        raise CommandSyntaxError("command stream must be iterable") from exc


# ---------------------------------------------------------------------------
# Command -> well range index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandRange:
    """The slice of the expanded well order covered by one command."""

    command_index: int  # 1-based
    op: str
    start: int  # 0-based, inclusive
    end: int  # 0-based, exclusive
    wells: tuple[str, ...]

    @property
    def missing(self) -> bool:
        return self.op == OP_GAP

    def to_dict(self) -> dict:
        return {
            "command_index": self.command_index,
            "op": self.op,
            "start": self.start,
            "end": self.end,
            "wells": list(self.wells),
            "missing": self.missing,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "CommandRange":
        return cls(
            command_index=int(payload["command_index"]),
            op=str(payload["op"]),
            start=int(payload["start"]),
            end=int(payload["end"]),
            wells=tuple(str(w) for w in payload["wells"]),
        )


# ---------------------------------------------------------------------------
# Stream digest (binds a directory, its snapshots and one original stream)
# ---------------------------------------------------------------------------


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_stream_digest(
    layout_digest: str,
    used_wells: Sequence[str],
    commands: Sequence[Command],
) -> str:
    """Digest of ``(layout, expanded well order, original stream)``.

    Snapshots and shipped packages carry this digest; a snapshot minted for a
    different stream can therefore never be re-used.
    """

    payload = {
        "format": "compressed-stream/v1",
        "layout_digest": layout_digest,
        "used_wells": list(used_wells),
        "commands": [command.to_dict() for command in commands],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Directory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandDirectory:
    """Fully provenanced boundary catalogue for one compressed stream.

    This is the *complete optional state* that stays at the dispatch end: the
    decoded values, the missing wells, the boundary memory of every command,
    and so on.  Only :meth:`shipped_view` leaves the server.
    """

    layout_digest: str
    used_wells: tuple[str, ...]
    commands: tuple[Command, ...]
    ranges: tuple[CommandRange, ...]
    boundary_memory: tuple[Fraction | None, ...]  # before each command
    memory_after: tuple[Fraction | None, ...]  # after each command
    decoded: tuple[tuple[str, Fraction], ...]
    missing: tuple[str, ...]
    stream_digest: str

    # -- construction ----------------------------------------------------------
    @classmethod
    def build(
        cls,
        layout: PublishedLayout,
        tokens: Iterable[object],
    ) -> "CommandDirectory":
        commands = parse_commands(tokens)
        used_wells = tuple(
            well.name for well in layout.wells if well.role != ROLE_UNUSED
        )
        if len(used_wells) > MAX_WELLS:
            raise StreamDecodeError(
                f"compressed input supports at most {MAX_WELLS} wells, got {len(used_wells)}"
            )
        return cls._decode(layout.digest, used_wells, commands)

    @classmethod
    def _decode(
        cls,
        layout_digest: str,
        used_wells: tuple[str, ...],
        commands: tuple[Command, ...],
    ) -> "CommandDirectory":
        n = len(used_wells)
        memory: Fraction | None = None
        pos = 0
        decoded: list[tuple[str, Fraction]] = []
        missing: list[str] = []
        ranges: list[CommandRange] = []
        boundary_memory: list[Fraction | None] = []
        memory_after: list[Fraction | None] = []

        for index, command in enumerate(commands):
            if pos >= n:
                raise StreamDecodeError(
                    f"trailing command {index + 1} ({command.text()}) after all "
                    f"{n} used wells were decoded"
                )
            boundary_memory.append(memory)
            slots = command.slots
            if pos + slots > n:
                raise StreamDecodeError(
                    f"command {index + 1} ({command.text()}) needs {slots} well(s) "
                    f"but only {n - pos} remain (out of bounds)"
                )
            wells = used_wells[pos : pos + slots]
            if command.op == OP_ABS:
                memory = command.value
                decoded.append((wells[0], memory))
            elif command.op == OP_DELTA:
                if memory is None:
                    raise StreamDecodeError(
                        f"command {index + 1} (DELTA) has no remembered value"
                    )
                memory = memory + command.value
                decoded.append((wells[0], memory))
            elif command.op == OP_REPEAT:
                if memory is None:
                    raise StreamDecodeError(
                        f"command {index + 1} (REPEAT) has no remembered value"
                    )
                for well in wells:
                    decoded.append((well, memory))
            else:  # GAP
                memory = None
                missing.extend(wells)
            memory_after.append(memory)
            ranges.append(
                CommandRange(
                    command_index=index + 1,
                    op=command.op,
                    start=pos,
                    end=pos + slots,
                    wells=tuple(wells),
                )
            )
            pos += slots

        if pos != n:
            raise StreamDecodeError(
                f"stream decoded only {pos} of {n} used wells (too few wells decoded)"
            )

        stream_digest = compute_stream_digest(layout_digest, used_wells, commands)
        return cls(
            layout_digest=layout_digest,
            used_wells=used_wells,
            commands=commands,
            ranges=tuple(ranges),
            boundary_memory=tuple(boundary_memory),
            memory_after=tuple(memory_after),
            decoded=tuple(decoded),
            missing=tuple(missing),
            stream_digest=stream_digest,
        )

    # -- views -----------------------------------------------------------------
    def to_measurements(self) -> dict[str, Fraction]:
        """Coordinates the existing readout warehouse imports unchanged."""

        return dict(self.decoded)

    def well_command(self, well: str) -> int:
        return self.well_lookup()[well][0]

    def well_lookup(self) -> dict[str, tuple[int, str]]:
        table: dict[str, tuple[int, str]] = {}
        for entry in self.ranges:
            for well in entry.wells:
                table[well] = (entry.command_index, entry.op)
        return table

    def snapshot_memory(self, command_index: int) -> Fraction | None:
        """Memory state immediately before the 1-based ``command_index``."""

        if not 1 <= command_index <= len(self.commands):
            raise CompressedStreamError(
                f"snapshot boundary {command_index} outside 1..{len(self.commands)}"
            )
        return self.boundary_memory[command_index - 1]

    def to_dict(self) -> dict:
        def rat(value: Fraction | None) -> str | None:
            return None if value is None else fmt_rational(value)

        return {
            "format": "command-directory/v1",
            "layout_digest": self.layout_digest,
            "stream_digest": self.stream_digest,
            "used_wells": list(self.used_wells),
            "commands": [command.to_dict() for command in self.commands],
            "command_text": [command.text() for command in self.commands],
            "ranges": [entry.to_dict() for entry in self.ranges],
            "boundary_memory": [rat(value) for value in self.boundary_memory],
            "memory_after": [rat(value) for value in self.memory_after],
            "decoded": {well: rat(value) for well, value in self.decoded},
            "missing": list(self.missing),
        }

    # -- shipping --------------------------------------------------------------
    def shipped_view(self, plan: "object"):
        """Build the minimal package for a lookup device (see retrieval)."""

        from .retrieval import ShippedPackage, ship

        if not hasattr(plan, "stream_digest"):
            raise CompressedStreamError("plan must be an AllocationPlan")
        return ship(self, plan)
