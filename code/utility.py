# A bunch of convenience functions that are used often during debugging and analysis

import pandas as pd
from collections.abc import Iterable


# Get the nflId given a player name
def get_nflId(names: list, players_fname: str) -> list:
    if not isinstance(names, Iterable):
        names = [names]
    if not isinstance(names[0], str):  # if names are already nflIds, then return
        return list(names)

    df_players = pd.read_csv(players_fname)
    ids = [None] * len(names)
    for i, name in enumerate(names):
        assert name in df_players.displayName.values, f"{name} not in players dataset"
        ids[i] = df_players[df_players.displayName == name].nflId.values[0]
    return ids


# Given a list of nflIds, return a dict with keys = ids, values = names
def get_player_name(ids: list, players_fname: str) -> dict:
    if not isinstance(ids, Iterable):
        ids = [ids]

    df_players = pd.read_csv(players_fname)
    names = {}
    for id in ids:
        assert id in df_players.nflId.values, f"{id} not in players dataset"
        names[id] = df_players[df_players.nflId == id].displayName.values[0]
    return names


# Return a tuple with (displayName, club) given an nflId
def get_player_info(id: int, players_fname: str) -> str:
    df_players = pd.read_csv(players_fname)
    df_tracking = pd.read_csv(
        "data/tracking_week_1.csv"
    )  # sucks that we have to load this, but this is the only place where nflId and club are linked

    df_full = df_tracking.merge(df_players, on=["nflId", "displayName"])
    return (
        df_full[df_full.nflId == id].displayName.iloc[0],
        df_full[df_full.nflId == id].club.iloc[0],
    )
