"""Generate one spatial transcriptomics dataset.

Example::

    python -m travlr_sim --layout layers --expression program --out travlr_sim/examples/spatial_layers
    python -m travlr_sim --layout islands --expression rust --cells 40 --out /tmp/spatial_rust
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .spatial import LAYOUTS, generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial transcriptomics synthetic dataset")
    parser.add_argument("--layout", default="layers", help="basic, enhanced, layers, islands, or a travlr scene name")
    parser.add_argument("--expression", default="program", choices=["program", "travlr", "rust"])
    parser.add_argument("--cells", type=int, default=48)
    parser.add_argument("--types", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.expression != "travlr" and args.layout not in LAYOUTS:
        parser.error(f"layout must be one of {', '.join(LAYOUTS)}")
    data = generate(
        args.out,
        layout=args.layout,
        expression=args.expression,
        n_cells=args.cells,
        n_types=args.types,
        seed=args.seed,
    )
    print(f"wrote {args.out} with {data.n_obs} cells and {data.n_vars} genes")


if __name__ == "__main__":
    main()
