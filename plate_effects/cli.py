"""Small command-line entry point for design editing, publishing and fitting.

Run ``python -m plate_effects.cli demo`` to execute the bundled synthetic
scenarios and write JSON reports under ``results/``.  Run
``python -m plate_effects.cli compressed [outdir]`` to write the replayable
compressed-stream examples under ``results/compressed/`` and print the
re-derived query results.
"""

from __future__ import annotations

import json
import sys

from .compressed_scenarios import build_examples, replay_example
from .scenarios import build_all_scenarios, scenario_to_dict
from .warehouse import clear_caches


def _dump(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def demo(outdir: str = "results") -> int:
    import os

    os.makedirs(outdir, exist_ok=True)
    clear_caches()
    scenarios = build_all_scenarios()
    summary = {}
    for name, result in scenarios.items():
        summary[name] = scenario_to_dict(result)
        _dump(os.path.join(outdir, f"{name}.json"), summary[name])
    _dump(os.path.join(outdir, "summary.json"), summary)
    for name, payload in summary.items():
        print(f"{name}: {payload['headline']}")
    return 0


def compressed(outdir: str = "results/compressed") -> int:
    import os

    os.makedirs(outdir, exist_ok=True)
    clear_caches()
    examples = build_examples()
    summary = {}
    for name, payload in examples.items():
        _dump(os.path.join(outdir, f"{name}.json"), payload)
        replayed = replay_example(payload)
        summary[name] = {
            "description": payload["description"],
            "stream": payload["stream"]["text"],
            "load": payload["load"],
            "max_pages": payload["max_pages"],
            "selected_boundaries": payload["allocation"]["selected_boundaries"],
            "pages_used": payload["allocation"]["pages_used"],
            "total_weighted_cost": payload["allocation"]["total_weighted_cost"],
            "paths": [
                {
                    "well": query["well"],
                    "start_boundary": query["start_boundary"],
                    "commands": query["commands"],
                    "commands_read": query["commands_read"],
                    "weighted_cost": query["weighted_cost"],
                }
                for query in payload["allocation"]["queries"]
            ],
            "replayed": replayed["wells"],
        }
    _dump(os.path.join(outdir, "summary.json"), summary)
    for name, payload in summary.items():
        print(
            f"{name}: pages={payload['selected_boundaries']} "
            f"cost={payload['total_weighted_cost']} stream={' '.join(payload['stream'])}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "usage: python -m plate_effects.cli demo [outdir]\n"
            "       python -m plate_effects.cli compressed [outdir]"
        )
        return 0
    if argv[0] == "demo":
        return demo(argv[1] if len(argv) > 1 else "results")
    if argv[0] == "compressed":
        return compressed(argv[1] if len(argv) > 1 else "results/compressed")
    print(f"unknown command {argv[0]!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
