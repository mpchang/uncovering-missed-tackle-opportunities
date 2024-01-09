"""
Functions used to clean data    
"""

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from constants import CLUB_DICT


def rotate_direction_and_orientation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rotate the direction and orientation angles so that 0° points from left to right on the field, and increasing angle goes counterclockwise
    This should be done BEFORE the call to make_plays_left_to_right, because that function with compensate for the flipped angles.

    :param df: the aggregate dataframe created using the aggregate_data() method

    :return df: the aggregate dataframe with orientation and direction angles rotated 90° clockwise
    """
    print(
        "INFO: Transforming orientation and direction angles so that 0° points from left to right, and increasing angle goes counterclockwise..."
    )
    df["o_clean"] = (-(df["o"] - 90)) % 360
    df["dir_clean"] = (-(df["dir"] - 90)) % 360
    return df


def make_plays_left_to_right(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flip tracking data so that all plays run from left to right. The new x, y, s, a, dis, o, and dir data
    will be stored in new columns with the suffix "_clean" even if the variables do not change from their original value.

    :param df: the aggregate dataframe created using the aggregate_data() method

    :return df: the aggregate dataframe with the new columns such that all plays run left to right
    """
    print("INFO: Flipping plays so that they all run from left to right...")
    df["x_clean"] = np.where(
        df["playDirection"] == "left",
        120 - df["x"],
        df[
            "x"
        ],  # 120 because the endzones (10 yds each) are included in the ["x"] values
    )
    df["y_clean"] = df["y"]
    df["s_clean"] = df["s"]
    df["a_clean"] = df["a"]
    df["dis_clean"] = df["dis"]
    df["o_clean"] = np.where(
        df["playDirection"] == "left", 180 - df["o_clean"], df["o_clean"]
    )
    df["o_clean"] = (df["o_clean"] + 360) % 360  # remove negative angles
    df["dir_clean"] = np.where(
        df["playDirection"] == "left", 180 - df["dir_clean"], df["dir_clean"]
    )
    df["dir_clean"] = (df["dir_clean"] + 360) % 360  # remove negative angles
    return df


def convert_geometry_to_int(df: pd.DataFrame):
    """
    Convert the x_clean, y_clean, dir_clean, o_clean, s_clean, a_clean columns to int to reduce dataframe size.
    We do this by multiplying the position, speed, acceleration vectors by 100, and the angle vectors by 10, and
    rounding to the nearest integer. This effectively reduces the precision of position, speed, and acceleration
    to the hundredths decimal, and the angle to the tenth decimal.

    :param df: the aggregate dataframe created using the aggregate_data() method

    :return df: the aggregate dataframe with the geometry column converted to a tuple of ints
    """
    state_cols = ["x_clean", "y_clean", "s_clean", "a_clean"]
    angle_cols = ["dir_clean", "o_clean"]

    print("INFO: Converting geometry variables from floats to int...")
    before = df.memory_usage(deep=True).sum()
    for col in state_cols:
        df.loc[:, col] = (df.loc[:, col] * 100).apply(round)
        assert (
            df.loc[:, col].abs().max() < 32767
        ), f"ERROR: The max value of column {col} is too large for int 16"
    for col in angle_cols:
        df.loc[:, col] = (df.loc[:, col] * 10).apply(round)
        assert (df.loc[:, col] >= 0).all(), "Angles should be greater than 0"
        assert (
            df.loc[:, col].max() < 32767
        ), f"ERROR: The max value of column {col} is too large for int 16"

    df = df.astype(
        {col: "int16" for col in state_cols + angle_cols}
    )  # int16 needed to cover all values
    after = df.memory_usage(deep=True).sum()
    print(f"INFO: Memory usage reduced from {before} to {after}")

    return df


def convert_teams_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the club, defensiveTeam, and possessionTeam columns to numeric values to
    reduce dataframe size.

    :param df: the aggregate dataframe created using the aggregate_data() method

    :return df: the aggregate dataframe with the club column converted to numeric values
    """

    print("INFO: Converting teams from str to numeric...")
    before = df.memory_usage(deep=True).sum()
    for col in ["club", "defensiveTeam", "possessionTeam"]:
        df.loc[:, col] = df.loc[:, col].apply(lambda x: CLUB_DICT[x]).astype("int8")
        df = df.astype({col: "int8"})
    after = df.memory_usage(deep=True).sum()
    print(f"INFO: Memory usage reduced from {before} to {after}")

    return df


def remove_plays_with_penalties(df, strict=False):
    """
    Remove rows from the dataframe where playNullifiedByPenalty == "Y" because these are not helpful for our model

    :param: df (pd.DataFrame): dataframe to filter
    :param: strict (boolean): if False, only drop plays where playNullifiedByPenalty == "Y". If True, drop plays where foulName1 is not NaN

    :return: (pd.DataFrame) filtered dataframe with plays nullified by penalties are dropped
    """
    print("INFO: Removing play with penalties...")
    before = len(df)
    if strict:
        df = df[(df.foulName1.isna()) & (df.playNullifiedByPenalty != "Y")]
    else:
        df = df[df.playNullifiedByPenalty != "Y"]
    after = len(df)
    print(f"INFO: {before - after} rows removed")
    return df


def remove_touchdowns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove touchdowns from the dataframe, since these are not relevant for tackling

    :param: df (pd.DataFrame): dataframe to filter

    :return (pd.DataFrame) filtered dataframe with all frames associated with touchdown plays removed
    """
    print("INFO: Removing plays with touchdowns")
    before = len(df)
    td_plays = []

    df_td = df[df.event == "touchdown"]
    td_plays = df_td.groupby(["gameId", "playId"]).groups.keys()

    for gameId, playId in tqdm(td_plays, ascii=True, desc="remove_touchdowns"):
        df = df[~((df.gameId == gameId) & (df.playId == playId))]

    after = len(df)
    print(f"INFO: {before - after} rows removed")

    return df


def remove_inactive_frames(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove frames before the ball is snapped and after the tackle/out-of-bounds/touchdown is made, since these are not relevant for the models.

    Args:
        df (pd.DataFrame): DataFrame of tracking plays

    Returns:
        pd.DataFrame: DataFrame with frames before ball_snap and after tackle are removed
    """

    def remove_frames(df: pd.DataFrame) -> pd.DataFrame:
        try:
            tackle_frameId = df[df.event == "tackle"].frameId.values[0]
            assert (
                len(df[df.event == "tackle"].frameId.unique()) == 1
            ), "ERROR: There cannot be multiple tackle events in one play"
        except IndexError:
            # no tackle in this play
            tackle_frameId = df.frameId.max()

        try:
            oob_frameId = df[df.event == "out_of_bounds"].frameId.values[0]
            assert (
                len(df[df.event == "out_of_bounds"].frameId.unique()) == 1
            ), "ERROR: There cannot be multiple out-of-bounds events in one play"
        except IndexError:
            # no out of bounds in this play
            oob_frameId = df.frameId.max()

        try:
            ball_snap_frameId = df[df.event == "ball_snap"].frameId.values[0] + 5
            assert (
                len(df[df.event == "ball_snap"].frameId.unique()) == 1
            ), "ERROR: There cannot be multiple ball_snap events in one play"
        except IndexError:
            # ball snapped before tracking started
            ball_snap_frameId = (
                df.frameId.min()
            )  # +5 to add half a second after the ball is snapped.

        max_frameId = min(tackle_frameId, oob_frameId)
        min_frameId = ball_snap_frameId

        return df[(df.frameId >= min_frameId) & (df.frameId <= max_frameId)]

    print("INFO: Removing inactive frames...")
    before = len(df)
    df = df.groupby(["gameId", "playId"]).apply(remove_frames).reset_index(drop=True)
    after = len(df)
    print(f"INFO: {before - after} rows removed")
    return df


def label_run_or_pass(df: pd.DataFrame) -> pd.DataFrame:
    print("INFO: Labeling plays as runs or passes")
    df["is_run"] = pd.isna(df["passResult"]) | (df["passResult"] == "R")
    return df


def downcast_ints_and_floats(df: pd.DataFrame) -> pd.DataFrame:
    print("INFO: Downcasting integers and floats...")
    before = df.memory_usage(deep=True).sum()
    int_columns = df.select_dtypes(include="integer").columns
    float_columns = df.select_dtypes(include="float").columns

    # Downcast integer columns
    for col in int_columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    # Downcast float columns
    for col in float_columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    after = df.memory_usage(deep=True).sum()
    print("INFO: Memory usage reduced from {} to {}".format(before, after))

    return df


def remove_bad_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Catch all function to remove bad data from the dataframe. "Bad data" are defined to match one of the following cases:
    1) ballcarrierId is not unique within a play
    2) ballcarrierId does not match one of the actual players tracked during the play
    """

    print("INFO: Removing bad data...")
    before = len(df)

    for key, frame in df.groupby(["gameId", "playId"]):
        if frame.ballCarrierId.nunique() != 1:
            df.drop(frame.index, inplace=True)
            print(f"INFO: >1 ballcarrier on play, dropping play: {key}")
        if frame.ballCarrierId.values[0] not in frame.nflId.unique():
            df.drop(frame.index, inplace=True)
            print(
                f"INFO: ballCarrierId does not match any player on field, dropping key: {key}"
            )
    after = len(df)
    df = df.reset_index(drop=True)
    print(f"INFO: {before-after} rows removed")
    return df


def strip_unused_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop dataframe columns from df that aren't useful in actually training our models, to make memory usage smaller.

    :param df (pd.DataFrame): dataframe to filter
    """
    # Only keep columns critical for the PlayFrame class to function
    useful_columns = [
        "gameId",
        "playId",
        "frameId",
        "club",
        "possessionTeam",
        "defensiveTeam",
        "is_run",
        "nflId",
        "o_clean",
        "a_clean",
        "s_clean",
        "x_clean",
        "y_clean",
        "dir_clean",
        "weight",
        "ballCarrierId",
        "playResult",
        "event",
        "tackle_dict",
        "age",
    ]
    columns_to_drop = [col for col in df.columns if col not in useful_columns]
    print("INFO: Removing unused columns from dataframe...")
    before = len(df.columns)
    df = df.drop(columns=columns_to_drop)
    after = len(df.columns)
    print(f"INFO: {before - after} columns removed")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes as input the aggregated dataframe of plays, tackles, players, and tracking data and performs
    the following preprocessing operations:

    1) Rotates the direction and orientation data so that the convention matches the unit circle
    2) Flips plays so that they run from left to right
    3) Adds a label to indicate whether the play is a pass or a run

    Subsequently, it cleans the data as follows:

    1) Remove plays with penalties
    2) Remove plays that resulted in touchdowns
    3) Convert teams from strings to ints to reduce memory
    4) Remove inactive frames (before the ball snap and after the tackle)
    5) Remove any bad data (not all players are tracked, multiple ballcarriers)
    6) Strip unused df columns to save memory

    :param df (pd.DataFrame): the original, aggregated dataframe
    :return df_clean (pd.DataFrame): the cleaned dataframe
    """

    # Data preprocessing so that all plays run from left-to-right and all angles match the standard unit circle convention
    df = rotate_direction_and_orientation(df)
    df = make_plays_left_to_right(df)
    df = label_run_or_pass(df)

    # Data Cleaning
    df = remove_plays_with_penalties(df, strict=True)
    # df = remove_touchdowns(df)
    df = convert_teams_to_numeric(df)
    df = remove_inactive_frames(df)
    df = remove_bad_data(df)
    df = strip_unused_data(df)

    return df


def optimize_memory_usage(df):
    """
    Optimize the memory usage by performing the following numerical operations:

    1) Converts the speed and position to ints by multiplying by 100 and then converting to int
    2) Converts angles to ints by multiplying by 10 and then converting to int
    3) Downcast all ints and floats to the most compact representation without corrupting the value
    """

    df = convert_geometry_to_int(df)
    df = downcast_ints_and_floats(df)
    # df.info(memory_usage="deep")

    return df
