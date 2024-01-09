"""
Functions used to merge and preprocess the data prior to analysis
"""

import pandas as pd
import numpy as np
import torch
from tqdm.notebook import tqdm
from datetime import datetime
from data_cleaning import downcast_ints_and_floats
from tacklesequence import TackleSequence
from spatial import (
    get_areas_from_points,
    get_positions_from_dataframe,
    get_influence_for_frame,
    get_influence_for_blockers,
)

DEBUG = True


def make_tackle_dict(frame: pd.DataFrame) -> dict:
    """
    Custom agg function used to merge tackling data with tracking data
    In the merged dataframe, the aggregate tackle data is stored as a dict

    :param tackle_tuple: a tuple of (nflId, tackles, assists)

    :return tackle_dict: a dict of {nflId: (tackles, assists)}
    """
    tackle_dict = {}

    for ix, row in frame.iterrows():
        nflId = row["nflId"]
        tackles = row["tackle"]
        assists = row["assist"]
        missed_tackles = row["pff_missedTackle"]
        if tackles + assists + missed_tackles == 0:
            continue
        else:
            if nflId in tackle_dict:  # if player already in tackle dict for this play
                tackle_dict[nflId] = (
                    tackle_dict[nflId][0] + tackles,
                    tackle_dict[nflId][1] + assists,
                    tackle_dict[nflId][2] + missed_tackles,
                )
            else:  # if player is not in the tackle dict for this play
                tackle_dict[nflId] = (tackles, assists, missed_tackles)

    return tackle_dict


def annotate_tackle_frames(
    df_tracking: pd.DataFrame, df_tackles: pd.DataFrame
) -> pd.DataFrame:
    """
    Given tracking and tackle dataframes, use the tracking data to identify and annotate which frameIds each tackle or missed tackle occurred.
    Attach this to the tackle dataframe and return it.

    :param df_tracking: the tracking dataframe derived from aggregate_data() function
    :param df_tackles: the tackle dataframe derived from tackles.csv

    :return df_tackles: the tackle dataframe with the frameId of each tackle event annotated
    :return real_tackle_ct: the number of tackling events annotated in the dataset
    """
    # Question 1: Do we include assists? I think to start with, no, since we don't really know what they mean.
    # Question 2: Do we include out of bounds tackles? To start with, no, since it is fundamentally a different type of tackle

    real_tackle_ct = 0

    before = len(df_tackles)
    valid_tackles = df_tackles[
        (df_tackles["tackle"] == 1) | (df_tackles["pff_missedTackle"] == 1)
    ].copy()
    after = len(valid_tackles)
    print(
        f"INFO: {before} tackling events -> {after} tackling events, keeping only plays with tackles or missed tackles. Note: assisted tackles are ignored."
    )

    # create a new column to track frameId of tackling event
    valid_tackles["tackleFrameId"] = -1

    # filter tracking data so that it only contains tackling events (either a tackle or OOB)
    df_tk_events = df_tracking[
        (df_tracking["event"] == "tackle") | (df_tracking["event"] == "out_of_bounds")
    ]
    useful_columns = ["gameId", "playId", "frameId", "event"]
    columns_to_drop = [col for col in df_tk_events.columns if col not in useful_columns]
    df_tk_events.drop(columns=columns_to_drop)  # shrink size before processing

    ct = 0
    n_rows = len(valid_tackles)

    for ix, row in tqdm(
        valid_tackles.iterrows(),
        total=n_rows,
        ascii=True,
        desc="annotate_tackle_frames",
    ):
        gameId = row["gameId"]
        playId = row["playId"]

        # if ct % 1000 == 0:
        #     print(f"INFO: {ct}/{len(valid_tackles)} Completed")

        # Cross-reference with the tracking data to identify frame with tackle
        df_hit = df_tk_events[
            (df_tk_events["playId"] == playId)
            & (df_tk_events["gameId"] == gameId)
            & (
                (df_tk_events["event"] == "tackle")
                | (df_tk_events["event"] == "out_of_bounds")
            )
        ]
        if len(df_hit) > 0:
            tackle_frame = df_hit.frameId.unique()
            assert (
                len(tackle_frame) == 1
            ), f"There should be exactly 1 frame with a tackle, not {len(tackle_frame)}"
            real_tackle_ct += 1
            valid_tackles.loc[ix, "tackleFrameId"] = tackle_frame[0]

        ct += 1

    return valid_tackles, real_tackle_ct


def annotate_missed_tackle_frames(
    df_tracking: pd.DataFrame, df_tackles: pd.DataFrame
) -> pd.DataFrame:
    """Annotate the frames in which missed tackles occured on a play. If the tracking data for a play cannot be found
    then the missed tackle frameId is set to -1. Currently, the criteria to detect a missed tackle is given by the minimum of the
    Euclidian distance between the tackler and the ball carrier.

    Args:
        df_tracking (pd.DataFrame): DataFrame consisting of tracking data
        df_tackles (pd.DataFrame): DataFrame consisting of tackle data

    Returns:
        df_tackles (pd.DataFrame): df_tackles dataframe with a new column annotating the frame of the missed tackle
        missed_tackles_assigned (int): Number of missed tackles annotated
    """
    # create a new column to track frameId of missed tackling events
    df_tackles["missedTackleFrameId"] = -1
    missed_tackles_assigned = 0

    # Iterate through every missed tackling event in the tackle dataset and find the frameId of the missed tackle
    n_rows = len(df_tackles)
    for ix, row in tqdm(
        df_tackles.iterrows(),
        total=n_rows,
        ascii=True,
        desc="annotate_missed_tackle_frames",
    ):
        # if ix % 1000 == 0:
        #     print(f"INFO: {ix}/{len(df_tackles)} Completed")

        if row["pff_missedTackle"] == 1:
            gameId = row["gameId"]
            playId = row["playId"]
            tacklerId = row["nflId"]

            # find the frameId of the missed tackle by using the criterion of detecting when the tackler is closest to the bc during that play
            df_bc = df_tracking[
                (df_tracking.playId == playId)
                & (df_tracking.gameId == gameId)
                & (df_tracking.nflId == df_tracking.ballCarrierId)
            ].copy()
            df_bc = restore_geometry(df_bc)

            # create a new column to track relative distance between bc and tackler
            df_bc["rd"] = -1

            df_tackler = df_tracking[
                (df_tracking.playId == playId)
                & (df_tracking.gameId == gameId)
                & (df_tracking.nflId == tacklerId)
            ].copy()
            df_tackler = restore_geometry(df_tackler)

            if len(df_bc) != 0 and len(df_tackler) != 0:
                df_bc["rd"] = np.sqrt(
                    (df_bc["x_clean"].to_numpy() - df_tackler["x_clean"].to_numpy())
                    ** 2
                    + (df_bc["y_clean"].to_numpy() - df_tackler["y_clean"].to_numpy())
                    ** 2
                )
                df_tackles.loc[ix, "missedTackleFrameId"] = df_bc.loc[
                    df_bc["rd"].idxmin()
                ].frameId
                missed_tackles_assigned += 1

    return df_tackles, missed_tackles_assigned


def build_tackle_sequence_input(
    tackle_sequence: TackleSequence,
    df_sequence: pd.DataFrame,
    num_features: int,
    tacklerId: int,
    start_frameId: int,
    offset_frames: int,
    sequence_len: int,
    dt: int,
    verbose=False,
) -> None:
    """
    Function that builds the input tensors for training TackleNet. It uses a dataframe containing the tracking data, and a separate dataframe containing the tackle and missed tackle labels.
    It stores the generated input tensors within TackleSequence objects.

    :param tackle_sequence (TackleSequence): TackleSequence object used to store the input tensor generated by this function
    :param df_sequence: The DataFrame representing the tracking data for a sequence of frames from a single play, used to create the tackle sequence input tensors
    :param num_features: The number of features created in the input tensor
    :param tacklerId: the nflId of the player who represents the tackler in the sequence
    :param offset_frames: the last frame in a sequence will be offset_frames back in time from the annotated tackling event
    :param sequence_len: the number of frames in the sequence
    :param dt: the time step between frames in the sequence (in unit of frames)
    :param verbose: if True, output verbose information
    """

    # Create the input tensor for the TackleSequence object. The features of the input tensor are:
    # 1) Absolute ballcarrier speed
    # 2) Relative x speed between ballcarrier and tackler (s_t,x - s_bc,x)
    # 3) Relative y speed between ballcarrier and tackler (s_t,x - s_bc,x)
    # 4) Euclidian Distance between ballcarrier and tackler (Euclidian)
    # 5) Angle of attack (angle between the tackler dir vector and separation vector)
    # 6) Voronoi cell are of the ballcarrier
    # 7) Influence at the ballcarrier's position
    # 8) Whether a play is a run (0) or a pass (1)

    input_tensor = torch.zeros(sequence_len, num_features, dtype=torch.float32)

    for i in range(sequence_len):
        df_frame = df_sequence[df_sequence.frameId == start_frameId + i * dt]
        df_frame = restore_geometry(df_frame)
        assert (
            df_frame.ballCarrierId.nunique() == 1
        ), f"More than one ballcarrier in sequence: {df_frame.ballCarrierId.unique()}"
        ballCarrierId = df_frame.ballCarrierId.iloc[0]

        # absolute bc speed
        s_bc = df_frame[df_frame.nflId == ballCarrierId].s_clean.values[0]
        input_tensor[i, 0] = s_bc

        # relative x, y speed between bc and tackler
        theta_bc_dir = df_frame[df_frame.nflId == ballCarrierId].dir_clean.values[0]
        s_bc_x = s_bc * np.cos(np.pi / 180 * theta_bc_dir)
        s_bc_y = s_bc * np.sin(np.pi / 180 * theta_bc_dir)

        theta_t_dir = df_frame[df_frame.nflId == tacklerId].dir_clean.values[0]
        s_t = df_frame[df_frame.nflId == tacklerId].s_clean.values[0]
        s_t_x = s_t * np.cos(np.pi / 180 * theta_t_dir)
        s_t_y = s_t * np.sin(np.pi / 180 * theta_t_dir)

        input_tensor[i, 1] = s_t_x - s_bc_x
        input_tensor[i, 2] = np.abs(s_t_y - s_bc_y)

        # Eucilian distance between bc and tackler
        pos_bc = df_frame[df_frame.nflId == ballCarrierId][
            ["x_clean", "y_clean"]
        ].values[0]
        pos_t = df_frame[df_frame.nflId == tacklerId][["x_clean", "y_clean"]].values[0]
        input_tensor[i, 3] = np.sqrt(
            (pos_bc[0] - pos_t[0]) ** 2 + (pos_bc[1] - pos_t[1]) ** 2
        )

        # Angle of attack (angle between the tackler dir vector and separation vector)
        # The separation vector is defined as the vector from the tackler to the ballcarrier's location in (offset_frames)/10 seconds, if the ballcarrier maintains their current speed and direction
        projected_pos = (
            pos_bc
            + s_bc
            * np.array(
                [
                    np.cos(np.pi / 180 * theta_bc_dir),
                    np.sin(np.pi / 180 * theta_bc_dir),
                ]
            )
            * offset_frames
            / 10
        )
        theta_sep = (
            np.arctan2(projected_pos[1] - pos_t[1], projected_pos[0] - pos_t[0])
            * 180
            / np.pi
        )
        theta_t_dir = df_frame[df_frame.nflId == tacklerId].dir_clean.values[0]
        input_tensor[i, 4] = np.cos(np.pi / 180 * (theta_sep - theta_t_dir))

        # Orientation of attack (angle between the tackler orientation vector and separation vector)
        # theta_t_o = df_frame[df_frame.nflId == tacklerId].o_clean.values[0]
        # input_tensor[i, 5] = np.cos(np.pi / 180 * (theta_sep - theta_t_o))

        # Relative mass between the ballcarrier and the tackler (tackler - ballcarrier)
        # mass_bc = df_frame[df_frame.nflId == ballCarrierId].weight.values[0]
        # mass_t = df_frame[df_frame.nflId == tacklerId].weight.values[0]
        # input_tensor[i, 5] = mass_t - mass_bc

        # Voronoi cell area of the ballcarrier
        point_dict = get_positions_from_dataframe(df_frame)
        area_dict = get_areas_from_points(point_dict, ballCarrierId, restriction=5)
        input_tensor[i, 5] = area_dict[ballCarrierId]

        # Influence at the ballcarrier position
        influence_dict, _ = get_influence_for_frame(df_frame)
        input_tensor[i, 6] = influence_dict[ballCarrierId]

        # Influence of the blockers at tackler position
        blocker_influence_dict, _ = get_influence_for_blockers(df_frame)
        input_tensor[i, 7] = blocker_influence_dict[tacklerId]

        # Run or Pass
        input_tensor[i, 8] = int(df_frame.is_run.values[0])

        # Age of the tackler
        # input_tensor[i, 8] = float(df_frame[df_frame.nflId == tacklerId].age.values[0])

        tackle_sequence.input_tensor = input_tensor


def build_inference_tackle_sequences(
    df_tracking: pd.DataFrame,
    gameId: int,
    playId: int,
    tacklerId: list,
    offset_frames: int,
    sequence_len: int,
    dt: int = 1,
    verbose=False,
) -> list:
    """
    Use this function to build TackleSequences used specifically for inference.
    Given a single play and player nflId (representing the tackler), construct a list of TackleSequences that will be used for inference.

    :param df_tracking: the aggregate dataframe created using the aggregate_data() method
    :param gameId: the gameId of the play
    :param playId: the playId of the play
    :param tacklerId: the nflId ID of the player who is assumed to be tackler
    :param offset_frames: the number of frames to look ahead to project tackling probability. This MUST match what offset_frames was used when training the model.
    :param sequence_len: the number of consecutive frames to used when generating the input tensor
    :param dt: the time step between frames in each sequence (in units of frames)
    :param verbose: if True, output verbose information

    :return list of TackleSequence objects
    """
    ct = 0  # counter object for number of sequences created
    num_features = 9  # number of features TackleSequence inputs
    tackle_sequences = []  # list to store TackleSequence objects

    # get the tracking data for the specific play
    df_play = df_tracking[
        (df_tracking.playId == playId) & (df_tracking.gameId == gameId)
    ].copy()

    if tacklerId not in df_play.nflId.values:
        raise ValueError("tacklerId is not in the tracking data")

    frameIds = df_play.frameId.unique()
    for frameId in frameIds[::dt]:
        df_sequence = df_play[df_play.frameId == frameId]
        ts = TackleSequence(df_sequence=df_sequence, target=None, tacklerId=tacklerId)
        build_tackle_sequence_input(
            ts,
            df_sequence,
            num_features,
            tacklerId,
            frameId,
            offset_frames,
            sequence_len,
            dt,
            verbose,
        )
        tackle_sequences.append(ts)
        ct += 1
    assert (
        ct == len(frameIds) // dt
    ), "Number of TackleSequences created does not match the number of frames in the play (divided by dt)"

    return tackle_sequences


def build_training_tackle_sequences(
    df_tracking: pd.DataFrame,
    df_tackles: pd.DataFrame,
    offset_frames: int = 5,
    n_sequences: int = None,
    sequence_len: int = 1,
    dt: int = 1,
    verbose=False,
) -> list:
    """
    Function that builds the input tensors for training TackleNet. It uses a dataframe containing the tracking data, and a separate dataframe containing the tackle and missed tackle labels.
    It stores the generated input tensors within TackleSequence objects and returns those.

    :param df_tracking: the aggregate dataframe created using the aggregate_data() method
    :param df_tackles: the tackle dataframe derived from tackles.csv and annotated with made and missed tackle frameIds
    :param offset_frames: the last frame in a sequence will be offset_frames back in time from the annotated tackling event
    :param n_sequences: the number of sequences to return (useful if testing with a smaller batch)
    :param sequence_len: the number of frames in each sequence
    :param dt: the time step between frames in each sequence (in units of frames)
    :param verbose: if True, output verbose information

    :return list of TackleSequence objects, unnormalized tensor stack of inputs, normalized tensor stack of inputs, and a tuple of (means, stds) of each feature (used this for inference)
    """

    ct = 0  # counter object for number of sequences created
    skipped_ct = 0
    num_features = 9  # number of features TackleSequence inputs
    tackle_sequences = []  # list to store TackleSequence objects

    n_rows = len(df_tackles)
    for ix, row in tqdm(
        df_tackles.iterrows(),
        total=n_rows,
        ascii=True,
        desc="build_training_tackle_sequences",
    ):
        if verbose:
            print(f"INFO: {ct}/{len(df_tackles)} Completed")

        # get the tracking data for the specific play
        gameId = row["gameId"]
        playId = row["playId"]
        tacklerId = row["nflId"]
        df_play = df_tracking[
            (df_tracking.playId == playId) & (df_tracking.gameId == gameId)
        ].copy()

        if tacklerId not in df_play.nflId.values:
            # tacklerId is not in the tracking data. Skip it.
            skipped_ct += 1
            continue

        # get the frameId of the made or missed tackle
        tackleFrameId = row["tackleFrameId"]
        missedTackleFrameId = row["missedTackleFrameId"]

        if row["tackle"] == 1:
            eventFrameId = tackleFrameId
        elif row["pff_missedTackle"] == 1:
            eventFrameId = missedTackleFrameId
        else:
            # this play had neither a tackle nor a missed tackle, so skip it
            skipped_ct += 1
            continue

        if eventFrameId == -1:
            # this play did not get annotated with tackleFrameId or missedTackleFrameId. It may be because the tracking data did not include this play or there was a penalty.
            skipped_ct += 1
            continue

        # get start_frame and end_frame IDs
        start_frameId = eventFrameId - offset_frames - (sequence_len - 1) * dt
        end_frameId = eventFrameId - offset_frames

        if start_frameId <= 0:
            # this play is not long enough to satisfy the requirement of sequence length and offset_frames. Skip it.
            skipped_ct += 1
            continue

        # get the tracking data for the sequence
        df_sequence = df_play[
            (df_play.frameId >= start_frameId) & (df_play.frameId <= end_frameId)
        ].copy()

        if df_sequence.empty:
            # this play is too short to meet the offset_frame requirements
            skipped_ct += 1
            continue

        # create the TackleSequence object
        if row["tackle"] == 1:
            target = 1
        elif row["pff_missedTackle"] == 1:
            target = 0
        else:
            raise ValueError(
                "row['tackle'] and row['pff_missedTackle'] should not both be zero"
            )

        ts = TackleSequence(df_sequence, target, tacklerId)
        build_tackle_sequence_input(
            ts,
            df_sequence,
            num_features,
            tacklerId,
            start_frameId,
            offset_frames,
            sequence_len,
            dt,
            verbose,
        )

        # add the TackleSequence object to the list
        tackle_sequences.append(ts)

        # increment the counter
        ct += 1

        # check if we have created enough sequences
        if n_sequences is not None and ct >= n_sequences:
            break

    print(
        f"INFO: {skipped_ct} tackling events skipped due to missing tracking data, missing annotation data, or being too short"
    )
    print(f"INFO: {len(tackle_sequences)} tackle sequences created")
    return tackle_sequences


def standardize_tackle_sequences(
    tackle_sequences: list, means: torch.tensor = None, stds: torch.tensor = None
) -> tuple:
    """
    Use this function to standardize a tackle sequence. If means and stds are provided, then they will be used to perform the standardization.
    If they are not, then the means/stds are calculated from the input tackle sequence. Means and stds will be outputed, so they can be used to standardize other tackle sequences.

    :param tackle_sequences: list of TackleSequence objects
    :param means: means of each feature
    :param stds: standard deviations of each feature

    :return unnormalized tensor stack, normalized tensor stack, tuple of (means, stds)
    """
    num_features = tackle_sequences[0].input_tensor.shape[1]
    unnorm_tensor_stack = torch.stack(
        [ts.input_tensor for ts in tackle_sequences]
    ).view(-1, num_features)
    if means is None:
        means = unnorm_tensor_stack.mean(dim=0)
    if stds is None:
        stds = unnorm_tensor_stack.std(dim=0)

    idx_std = [
        0,
        1,
        2,
        3,
        5,
        6,
        7,
        8,
    ]  # standardize all except the angle of attack, which have specific meaning unnormalized
    for ts in tackle_sequences:
        ts.input_tensor[:, idx_std] = (
            ts.input_tensor[:, idx_std] - means[idx_std]
        ) / stds[idx_std]
    norm_tensor_stack = torch.stack([ts.input_tensor for ts in tackle_sequences]).view(
        -1, num_features
    )

    return unnorm_tensor_stack, norm_tensor_stack, (means, stds)


def aggregate_data(
    plays_fname: str, tackle_fname: str, players_fname: str, tracking_fname_list: list
) -> pd.DataFrame:
    """
    Create the aggregate dataframe by merging together the plays data, tracking data, and tackles data
    In the aggregate dataframe, the tackles will be represented by a tackle_dict column. Each entry is a
    dict which consists of the nflId of the tackler as the key, and a tuple of (tackles, assists) as the value

    :param plays_fname: the filename of the plays data
    :param tackle_fname: the filename of the tackles data
    :param players_fname: the filename of the players data
    :param tracking_fname_list: a list of filenames of all tracking data

    :return df_agg3: the aggregate dataframe
    """
    print(
        "INFO: Aggregating data from play data, tracking data, tackles data, and players data into a master dataframe..."
    )
    # import files
    df_plays = pd.read_csv(plays_fname)
    df_tracking = pd.concat(
        [pd.read_csv(tracking_fname) for tracking_fname in tracking_fname_list]
    )
    df_players = pd.read_csv(players_fname)
    # Create column for age
    # Birthdays have NAs but we'll ignore those for now.
    df_players["birthDate"] = pd.to_datetime(
        df_players["birthDate"], format="%Y-%m-%d", errors="coerce"
    )
    df_players["age"] = (
        datetime.now().year
        - df_players["birthDate"].dt.year
        + (datetime.now().month - df_players["birthDate"].dt.month) / 12
    )
    df_players["age"] = df_players["age"].astype("float32")
    df_players["age"] = df_players["age"].fillna(df_players["age"].mean())

    # aggregate plays, tracking, players tables
    df_agg1 = pd.merge(df_plays, df_tracking, on=["gameId", "playId"], how="inner")
    df_agg2 = pd.merge(
        df_agg1, df_players, on=["nflId"], how="inner"
    )  # how = inner will drop any nflId and displayNames that are not common in both dataframes. For example the "football" rows are dropped.

    return df_agg2


def get_play_outcomes(
    agg_data,
) -> pd.DataFrame:  # Input needs to have play directions flipped already
    print("INFO: Creating dataframe of outcomes to train on...")
    df = agg_data[["gameId", "playId", "frameId", "ballCarrierId", "nflId", "x_clean"]]
    df = df[df.nflId == df.ballCarrierId]
    max_frame = df.loc[df.groupby(["gameId", "playId"])["frameId"].idxmax()]
    max_frame = max_frame.rename(columns={"x_clean": "ballCarrier_final_x_clean"})

    outcomes = df.merge(max_frame, on=["gameId", "playId"], how="left")
    outcomes["yards_gained_ball_carrier"] = (
        outcomes.ballCarrier_final_x_clean - outcomes.x_clean
    )

    outcomes.reset_index(drop=True, inplace=True)
    outcomes = outcomes[["gameId", "playId", "frameId_x", "yards_gained_ball_carrier"]]
    outcomes = outcomes.rename(columns={"frameId_x": "frameId"})

    downcast_ints_and_floats(outcomes)

    return outcomes


def restore_geometry(df):
    """
    Given a dataframe with columns ["x_clean", "y_clean", "s_clean", "a_clean", "dir_clean", "o_clean"], restore the original
    geometry by dividing x_clean, y_clean, s_clean, and a_clean by 100, and dividing dir_clean and o_clean by 10. The output
    datatype will be float16.
    """
    df.loc[:, "x_clean"] = df["x_clean"] / 100
    df.loc[:, "y_clean"] = df["y_clean"] / 100
    df.loc[:, "s_clean"] = df["s_clean"] / 100
    df.loc[:, "a_clean"] = df["a_clean"] / 100
    df.loc[:, "dir_clean"] = df["dir_clean"] / 10
    df.loc[:, "o_clean"] = df["o_clean"] / 10

    return df
