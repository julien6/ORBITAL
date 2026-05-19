from __future__ import annotations

import argparse

from mma.presets import make_orbital_basic_organization


def export_org_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export an MMA organization preset.")
    parser.add_argument("--preset", default="orbital_basic", choices=["orbital_basic"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--agents", type=int, default=8)
    args = parser.parse_args(argv)
    org = make_orbital_basic_organization([f"sat_{i}" for i in range(args.agents)])
    org.to_json(args.out)


if __name__ == "__main__":
    export_org_main()
