from pathlib import Path

from worldcup_props.data import StatsHubProvider, ingest_statshub_referees
from worldcup_props.db import connect


def _stat_row(stat, half):
    values = {
        ("fouls", "ALL"): ("12.00", "9.00"),
        ("fouls", "1ST"): ("5.00", "4.00"),
        ("cornerKicks", "ALL"): ("6.00", "3.00"),
        ("cornerKicks", "1ST"): ("4.00", "1.00"),
        ("offsides", "ALL"): ("2.00", "1.00"),
        ("offsides", "1ST"): ("1.00", "0.00"),
        ("shotsOnGoal", "ALL"): ("5.00", "2.00"),
        ("shotsOnGoal", "1ST"): ("3.00", "1.00"),
        ("cards", "ALL"): ("3.00", "1.00"),
        ("cards", "1ST"): ("2.00", "0.00"),
        ("yellowCards", "ALL"): ("2.00", "1.00"),
        ("yellowCards", "1ST"): ("1.00", "0.00"),
        ("redCards", "ALL"): ("1.00", "0.00"),
        ("redCards", "1ST"): ("1.00", "0.00"),
        ("ballPossession", "ALL"): ("58.00", "42.00"),
    }
    home, away = values[(stat, half)]
    return {
        "data": [
            {
                "event_id": 12345,
                "home_team_id": 4724,
                "away_team_id": 4789,
                "time_start_timestamp": "1729045800",
                "home_value": home,
                "away_value": away,
                "home_team_name": "USA",
                "away_team_name": "Paraguay",
                "league_name": "Int. Friendly Games",
            }
        ]
    }


def test_statshub_provider_merges_full_and_first_half_stats(tmp_path):
    teams = tmp_path / "teams.csv"
    teams.write_text(
        "team,statshub_team_id,confederation\n"
        "USA,4724,CONCACAF\n"
        "Paraguay,4789,CONMEBOL\n",
        encoding="utf-8",
    )
    provider = StatsHubProvider(teams, tmp_path / "raw")

    def fake_get_json(url, params=None, headers=None):
        return _stat_row(params["statisticKey"], params["eventHalf"])

    provider.http.get_json = fake_get_json
    rows = list(provider.fetch("2020-07-01"))
    assert len(rows) == 1
    match = rows[0]
    assert match.competition_type == "friendly"
    assert match.home_stats.fouls == 12
    assert match.away_stats.corners == 3
    assert match.home_stats.offsides == 2
    assert match.home_stats.possession == 58.0
    assert match.home_stats.shots_on_target == 5
    assert match.home_stats.cards == 3
    assert match.home_stats.yellow_cards == 2
    assert match.home_stats.red_cards == 1
    assert match.home_stats.first_half_corners == 4
    assert match.home_stats.first_half_cards == 2
    assert match.home_stats.first_half_red_cards == 1
    assert match.away_stats.first_half_offsides == 0


def test_statshub_referee_ingestion(tmp_path):
    database = Path(tmp_path) / "props.sqlite"
    count = ingest_statshub_referees(
        database,
        [
            {
                "name": "Test Referee",
                "games": 50,
                "avgFoulsPerGame": 24.5,
                "avgCardsPerGame": 4.1,
            }
        ],
    )
    assert count == 1
    with connect(database) as connection:
        row = connection.execute(
            "SELECT * FROM referees WHERE referee_name='Test Referee'"
        ).fetchone()
    assert row["matches"] == 50
    assert row["fouls_per_match"] == 24.5
    assert row["cards_per_match"] == 4.1
    assert row["source"] == "statshub"


def test_statshub_player_performance_rows(tmp_path):
    teams = tmp_path / "teams.csv"
    teams.write_text(
        "team,statshub_team_id,confederation\n"
        "Switzerland,4699,UEFA\n",
        encoding="utf-8",
    )
    provider = StatsHubProvider(teams, tmp_path / "raw")

    def fake_get_json(url, params=None, headers=None):
        assert url.endswith("/api/team/4699/players/performance")
        return {
            "data": [
                {
                    "name": "G. Xhaka",
                    "position": "M",
                    "stats": {
                        "1": {
                            "minutesPlayed": 90,
                            "shots": 2,
                            "onTargetScoringAttempt": 1,
                            "goals": 0,
                            "goalAssist": 1,
                            "fouls": 2,
                            "wasFouled": 1,
                            "yellowCard": None,
                            "position": "CM",
                        },
                        "2": {
                            "minutesPlayed": 45,
                            "shots": 1,
                            "onTargetScoringAttempt": 1,
                            "goals": 1,
                            "goalAssist": 0,
                            "fouls": 1,
                            "wasFouled": 0,
                            "yellowCard": 1,
                            "position": "CM",
                        },
                    },
                }
            ]
        }

    provider.http.get_json = fake_get_json
    rows = provider.fetch_player_profile_rows(min_minutes=90)

    assert len(rows) == 1
    row = rows[0]
    assert row["player_name"] == "G. Xhaka"
    assert row["national_team"] == "Switzerland"
    assert row["competition"] == "International UEFA"
    assert row["minutes"] == 135
    assert row["shots_on_target_per90"] == 2 * 90 / 135
    assert row["source"] == "statshub_player_performance"
