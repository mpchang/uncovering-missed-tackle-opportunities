import os
from matplotlib.animation import PillowWriter
from data_preprocessing import restore_geometry
from data_preprocessing import build_inference_tackle_sequences
from utility import get_nflId, get_player_name
from plotting.animate_play_with_metric import (
    make_play_with_metric_animation,
    combine_gifs,
    make_tackle_probability_animation,
)


def run_inference(
    model,
    df_tracking,
    gameId,
    playId,
    players_fname,
    plays_fname,
    defenseIds=None,
    display=False,
    output_dir=None,
    highlight_players=None,
    presentation_mode=False,
):
    """
    Run inference on a single play, given gameId and playId.

    param model: Pre-trained XGBoost Model
    param df_tracking: DataFrame of all the tracking data. Must include tracking data for gameId and playId. Geometry should not yet be restored on this df.
    param gameId: gameId to run inference on
    param playId: playId to run inference on
    param players_fname: file location of the players.csv dataset
    param plays_fname: file location of the plays.csv dataset
    param defense_ids: a list of defender nflIds or displayNames to predict tackle probability. The list must be all one or the other.
    param display: If True, creates animations of the play
    param output_dir: File location to save the generated animations, if display == True
    param highlight_players: Particular players to highlight in the animation
    param presentation_mode: blow up linewidths, fontsizes, and marker sizes

    return probs_dict: a dict with key = nflId and value = tackle probability over duration of play
    """

    # Isolate tracking data for this play
    df_sequence = df_tracking[
        (df_tracking.gameId == gameId) & (df_tracking.playId == playId)
    ]
    assert not df_sequence.empty, "ERROR: Play not found."

    # Get defense and offense Ids, if not provided
    restore_geometry(df_sequence)
    if defenseIds is None:
        defenseIds = df_sequence[
            df_sequence.club == df_sequence.defensiveTeam
        ].nflId.unique()
    offenseIds = df_sequence[
        df_sequence.club == df_sequence.possessionTeam
    ].nflId.unique()
    ballCarrierId = df_sequence.ballCarrierId.unique()[0]

    # convert defenseIds from displayNames to nflIds, if necessary
    defenseIds = get_nflId(defenseIds, players_fname)

    # create tackle sequences for each player and run inference on them, store resulting tackle probabilities in probs_dict
    probs_dict = {}
    outcome_dict = {}

    for id in defenseIds:
        ts_list = build_inference_tackle_sequences(
            df_tracking,
            gameId,
            playId,
            id,
            offset_frames=10,
            sequence_len=1,
            dt=1,
            verbose=True,
        )

        for ts in ts_list:
            x = ts.input_tensor
            x = x.view(x.shape[0], -1).cpu().numpy()
            probs = model.predict_proba(x)
            outcome_dict[(id, ts.start_frameId)] = probs[
                0, 1
            ]  # probability of a made tackle in the next offset_frames * 0.1 seconds

        frames = [ts.start_frameId for ts in ts_list]
        probs_dict[id] = [outcome_dict[(id, ts.start_frameId)] for ts in ts_list]

    if display:
        # Extract displayNames to label players
        ids = list(defenseIds) + list(offenseIds)
        names = get_player_name(ids, players_fname)

        # Convert highlight_names to highlightIds
        if highlight_players is not None:
            highlightIds = get_nflId(highlight_players, players_fname)
        else:
            highlightIds = None

        # extract frames where missed opportunity occured
        missed_opp_dict = find_missed_opportunity_frame(probs_dict)
        if highlightIds is not None:
            rm_ids = [id for id in missed_opp_dict.keys() if id not in highlightIds]
            for id in rm_ids:
                missed_opp_dict.pop(id, None)

        # create animation of tackle probability plot
        plot_animation = make_tackle_probability_animation(
            frames,
            probs_dict,
            defenseIds,
            names,
            highlightIds=highlightIds,
            missed_tackle_dict=missed_opp_dict,
            presentation_mode=presentation_mode,
        )
        writer1 = PillowWriter(fps=5, bitrate=1800)
        plot_animation.save(
            os.path.join(output_dir, f"tackle_prob_{gameId}_{playId}.gif"),
            writer=writer1,
            dpi=75,
        )

        # create animation of play with metric
        play_animation = make_play_with_metric_animation(
            gameId,
            playId,
            frames,
            defenseIds,
            offenseIds,
            ballCarrierId,
            probs_dict,
            df_sequence,
            names,
            highlightIds=highlightIds,
            missed_tackle_dict=missed_opp_dict,
            made_tackle_dict=None,
            plays_fname=plays_fname,
            presentation_mode=presentation_mode,
        )
        writer2 = PillowWriter(fps=5, bitrate=1800)
        play_animation.save(
            os.path.join(output_dir, f"play_with_metric_{gameId}_{playId}.gif"),
            writer=writer2,
            dpi=75,
        )

        # Combine animations
        combine_gifs(
            os.path.join(output_dir, f"tackle_prob_{gameId}_{playId}.gif"),
            os.path.join(output_dir, f"play_with_metric_{gameId}_{playId}.gif"),
            os.path.join(output_dir, f"play_with_tackle_prob_{gameId}_{playId}.gif"),
        )

    return probs_dict


def find_missed_opportunity_frame(probs_dict: dict) -> dict:
    """
    This function takes a probs_dict that is the output of the run_inference() function.
    It runs a dict that identifies the frame of the last missed opportunity (if any) for each player.
    If the player did not have any missed opportunities, then the missed_opp_frame will return -1

    param probs_dict: dictionary where keys = nflId, values = array of tackle probs for this play

    return: dict where keys = nflId, values = frame where the last missed opportunity occured for this player
    """

    missed_opp_dict = {}

    for id, probs in probs_dict.items():
        # walk through probs list frame by frame to identify missed oops
        state = 0
        missed_opp_frame = -1
        for frame, p in enumerate(probs):
            if p > 0.75:
                if state == 0:
                    state = 11
                elif state in [11, 12, 13]:  # increment tackle opp counter
                    state += 1
                elif state == 14:  # assign tackle opp
                    state += 1  # enter tackle opp steady state
                elif state in [
                    21,
                    22,
                    23,
                    24,
                ]:  # return back to tackle opp steady state, disable missed tackle opp counter
                    state = 15
            else:
                if state in [0, 11, 12, 13, 14]:  # reset tackle opp counter
                    state = 0
                elif state == 15:  # start missed tackle opp counter
                    state = 21
                elif state in [21, 22, 23]:  # increment missed tackle opp counter
                    state += 1
                elif state == 24:  # assign missed tackle opp, return to state 0
                    state = 0
                    missed_opp_frame = frame
        if missed_opp_frame != -1:
            missed_opp_dict[id] = missed_opp_frame

    return missed_opp_dict
