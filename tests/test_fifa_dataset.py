from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from worldcup_props.cli import main
from worldcup_props.fifa_dataset import ingest_fifa_world_cup_dataset
from worldcup_props.tournament import load_tournament_context


def _write_dataset(root: Path) -> None:
    root.mkdir()
    (root / "teams.csv").write_text(
        "\n".join(
            [
                "team_id,team_name,fifa_code,group_letter,confederation,fifa_ranking_pre_tournament,elo_rating,manager_name",
                "1,Ecuador,ECU,A,CONMEBOL,24,1780,Manager A",
                "2,Cabo Verde,CPV,A,CAF,68,1540,Manager B",
                "3,Algeria,ALG,A,CAF,36,1660,Manager C",
                "4,Argentina,ARG,A,CONMEBOL,1,2050,Manager D",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "matches.csv").write_text(
        "\n".join(
            [
                "match_id,date,kickoff_time_utc,stage_id,venue_id,home_team_id,away_team_id,home_score,away_score,status,home_xg,away_xg,referee_id",
                "1,2026-06-11,19:00,1,1,1,2,2,0,Completed,1.8,0.5,1",
                "2,2026-06-22,21:00,1,1,1,2,,,Scheduled,,,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "match_team_stats.csv").write_text(
        "\n".join(
            [
                "match_id,team_id,possession_pct,total_shots,shots_on_target,corners,fouls,offsides,saves,data_source,last_updated",
                "1,1,57,16,5,7,10,2,2,fifa.com,2026-06-21",
                "1,2,43,6,2,3,15,1,3,fifa.com,2026-06-21",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "match_events.csv").write_text(
        "\n".join(
            [
                "event_id,match_id,minute,event_type,team_id,player_id",
                "1,1,12,Goal,1,101",
                "2,1,63,Goal,1,102",
                "3,1,44,Yellow Card,2,201",
                "4,1,70,Red Card,2,202",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "referees.csv").write_text(
        "\n".join(
            [
                "referee_id,name,country,avg_cards_per_game",
                "1,Szymon Marciniak,Poland,4.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_fifa_world_cup_dataset_ingests_stats_cards_and_context(tmp_path):
    dataset = tmp_path / "fifa"
    _write_dataset(dataset)
    database = tmp_path / "props.sqlite"

    result = ingest_fifa_world_cup_dataset(database, dataset)

    assert result.matches_ingested == 1
    assert result.referees_ingested == 1
    assert result.tournament_context_rows_ingested == 2
    assert result.team_match_rows == 2
    assert result.rows_with_shots_on_target == 2
    assert result.rows_with_corners == 2
    assert result.rows_with_fouls == 2
    assert result.rows_with_offsides == 2
    assert result.rows_with_cards == 1
    assert result.rows_with_red_cards == 1

    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        cape_verde = connection.execute(
            """
            SELECT stats.*, matches.home_team, matches.away_team
            FROM team_match_stats AS stats
            JOIN matches AS matches ON matches.id = stats.match_id
            WHERE stats.team=?
            """,
            ("Cape Verde",),
        ).fetchone()
        assert cape_verde is not None
        assert cape_verde["away_team"] == "Cape Verde"
        assert cape_verde["fouls"] == 15
        assert cape_verde["cards"] == 2
        assert cape_verde["yellow_cards"] == 1
        assert cape_verde["red_cards"] == 1
        assert cape_verde["first_half_cards"] == 1
        assert cape_verde["first_half_red_cards"] is None

        referee = connection.execute(
            "SELECT cards_per_match FROM referees WHERE referee_name=?",
            ("Szymon Marciniak",),
        ).fetchone()
        assert referee["cards_per_match"] == 4.2

    context = load_tournament_context(database, "Ecuador", "Cape Verde")
    assert context is not None
    assert context.home is not None
    assert context.away is not None
    assert context.home.coast_if_leading
    assert not context.home.must_win
    assert context.away.must_win
    assert not context.away.coast_if_leading
    assert "group A" in (context.home.notes or "")


def test_cli_ingest_fifa_world_cup_dataset(tmp_path, capsys):
    dataset = tmp_path / "fifa"
    _write_dataset(dataset)
    config = tmp_path / "config.json"
    database = tmp_path / "props.sqlite"
    config.write_text(json.dumps({"database_path": str(database)}), encoding="utf-8")

    status = main(
        [
            "--config",
            str(config),
            "ingest-fifa-world-cup-dataset",
            str(dataset),
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["matches_ingested"] == 1
    assert payload["tournament_context_rows_ingested"] == 2
