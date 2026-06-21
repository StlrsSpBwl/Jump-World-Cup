from __future__ import annotations

import csv
import unicodedata
from dataclasses import dataclass
from pathlib import Path


VALID_CONFEDERATIONS = {"UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC"}


def default_team_registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "team_confederations.csv"


def normalize_team_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(ascii_value.casefold().replace("&", "and").split())


@dataclass(frozen=True)
class TeamConfederationRegistry:
    canonical: dict[str, str]
    lookup: dict[str, str]

    def confederation_for(self, team: str | None) -> str | None:
        if not team:
            return None
        return self.lookup.get(normalize_team_name(team))

    def canonical_metadata(self) -> dict[str, str]:
        return dict(sorted(self.canonical.items()))

    def lookup_metadata(self) -> dict[str, str]:
        return dict(sorted(self.lookup.items()))


def load_team_confederation_registry(
    path: str | Path | None = None,
    *,
    missing_ok: bool = False,
) -> TeamConfederationRegistry:
    registry_path = Path(path) if path is not None else default_team_registry_path()
    if not registry_path.exists():
        if missing_ok:
            return TeamConfederationRegistry(canonical={}, lookup={})
        raise FileNotFoundError(f"Team confederation registry not found: {registry_path}")

    canonical: dict[str, str] = {}
    lookup: dict[str, str] = {}
    with registry_path.open(newline="", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=2):
            team = (row.get("team") or "").strip()
            confederation = (row.get("confederation") or "").strip().upper()
            if not team:
                raise ValueError(f"Missing team in registry row {index}")
            if confederation not in VALID_CONFEDERATIONS:
                raise ValueError(
                    f"Invalid confederation {confederation!r} for {team!r} in row {index}"
                )
            canonical[team] = confederation
            names = [team]
            aliases = row.get("aliases") or ""
            names.extend(alias.strip() for alias in aliases.split(";") if alias.strip())
            for name in names:
                key = normalize_team_name(name)
                previous = lookup.get(key)
                if previous is not None and previous != confederation:
                    raise ValueError(
                        f"Conflicting confederations for alias {name!r}: "
                        f"{previous} vs {confederation}"
                    )
                lookup[key] = confederation
    return TeamConfederationRegistry(canonical=canonical, lookup=lookup)


def clean_confederation(value: str | None) -> str | None:
    if value is None:
        return None
    confederation = str(value).strip().upper()
    if confederation == "NAN":
        return None
    if not confederation or confederation == "UNK":
        return None
    if confederation not in VALID_CONFEDERATIONS:
        return None
    return confederation
