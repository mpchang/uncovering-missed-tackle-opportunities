import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import patches
import matplotlib as mpl
import numpy as np
import imageio
import os
import sys
import pandas as pd
import xgboost as xgb
import pickle
from functools import partial

root_dir = os.path.join(os.getcwd())
sys.path.append(root_dir)

from data_preprocessing import build_inference_tackle_sequences, restore_geometry


# Create an animation of players on the field through the duration of a play. A trail of markers will persist for each player.
# The color of the markers behind the defensive players will correspond to a provided metric.
# Ballcarrier will be black
# Offensive players that are no the ballcarrier will be white.


# Create an animation of players on the field
def create_football_field(ax, linenumbers=True, xlim=None, ylim=None):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    rect = patches.Rectangle(
        (0, 0),
        100,
        53.3,
        linewidth=0.1,
        edgecolor="r",
        facecolor="white",
        zorder=0,
        alpha=0.5,
    )
    ax.add_patch(rect)

    # Yard Lines
    linecolors = "black"
    ax.plot([10, 10], [0, 53.3], color=linecolors)
    ax.plot([20, 20], [0, 53.3], color=linecolors)
    ax.plot([30, 30], [0, 53.3], color=linecolors)
    ax.plot([40, 40], [0, 53.3], color=linecolors)
    ax.plot([50, 50], [0, 53.3], color=linecolors)
    ax.plot([60, 60], [0, 53.3], color=linecolors)
    ax.plot([70, 70], [0, 53.3], color=linecolors)
    ax.plot([80, 80], [0, 53.3], color=linecolors)
    ax.plot([90, 90], [0, 53.3], color=linecolors)

    # Yard Numbers
    if linenumbers:
        numbers = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
        label = ["", 10, 20, 30, 40, 50, 40, 30, 20, 10, "", ""]
        for loc in [3, 50]:
            for i, number in enumerate(numbers):
                ax.plot([number, number], [0, 53.3], color=linecolors)
                ax.text(
                    number,
                    loc,
                    str(label[i]),
                    fontdict={
                        "family": "sans-serif",
                        "size": 20,
                        "weight": "normal",
                        "ha": "center",
                        "va": "center",
                        "color": linecolors,
                    },
                )

    # Draw hashmarks in the appropriate locations on the football field
    hmark_length = 0.67  # 2 ft
    hm1_start = (53.3 - 18.5 / 3) / 2 - hmark_length / 2
    hm2_start = (53.3 - 18.5 / 3) / 2 + 18.5 / 3 - hmark_length / 2
    for yardline in np.arange(10, 110):
        ax.plot(
            [yardline, yardline],
            [hm1_start, hm1_start + hmark_length],
            color=linecolors,
        )
        ax.plot(
            [yardline, yardline],
            [hm2_start, hm2_start + hmark_length],
            color=linecolors,
        )

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return ax


# Function that plots a player's trail in the frame
# The color of the data point will correspond to the metric provided
def plot_metric_player(x, y, metric, ax, labels=None, marker_size=100):
    if isinstance(x, list):
        assert (
            len(x) == len(y) == len(metric)
        ), "Length of x, y, and metric must be the same"
    else:
        assert (
            isinstance(x, (float, np.float32))
            and isinstance(y, (float, np.float32))
            and isinstance(metric, (float, np.float32))
        ), "x, y, and metric must be floats if one of them is a float"

    # Plot players
    c = ax.scatter(x, y, c=metric, cmap="coolwarm", s=marker_size, vmin=0, vmax=1)
    if labels is not None:
        ax.text(x, y + 1, labels, fontsize=12, color="black", ha="center", va="center")
    return c


# Function that plot's a player's location. It does not leave a trail, nor does it color it based on any metric.
def plot_nonmetric_player(
    x, y, ax, edgecolors="black", facecolors="black", marker_size=100
):
    # Plot players
    c = ax.scatter(
        x,
        y,
        s=marker_size,
        edgecolors=edgecolors,
        facecolors=facecolors,
        marker="s",
    )
    return c


# Function to animate the play using the metric
# Ballcarrier will be black
# Offensive players will be white
# Defensive players with metric will be colored using the metric (e.g. tackle probability)
def update_play_animation(
    frame,
    max_frames,
    x_def,
    y_def,
    defenseIds,
    x_off,
    y_off,
    offenseIds,
    ballCarrierId,
    labels,
    metric,
    play_desc,
    gameId,
    playId,
    ax,
    highlightIds=None,
    missed_tackle_dict=None,
    made_tackle_dict=None,
):
    """Function to animate the player movement and metric"""

    # remove all lines and collections:
    for artist in ax.collections + ax.texts + ax.patches:
        artist.remove()

    if frame >= max_frames:  # add 5 frames to pause at the end
        frame = max_frames - 1

    for id in defenseIds:  # plot defensive players with metric
        # create trail of past positions
        plot_metric_player(
            x_def[id][:frame], y_def[id][:frame], metric[id][:frame], ax, marker_size=50
        )

        # create current position
        lab = labels[id]
        if (
            highlightIds is None
        ):  # if a highlightIds list was provided, only label these players. If not, label everyone
            plot_metric_player(
                x=x_def[id][frame],
                y=y_def[id][frame],
                metric=metric[id][frame],
                labels=lab,
                ax=ax,
                marker_size=150,
            )
        else:
            if id not in highlightIds:
                plot_metric_player(
                    x=x_def[id][frame],
                    y=y_def[id][frame],
                    metric=metric[id][frame],
                    labels=None,
                    ax=ax,
                    marker_size=150,
                )

    for id in offenseIds:  # plot offensive players without metric
        plot_nonmetric_player(
            x_off[id][frame],
            y_off[id][frame],
            ax,
            marker_size=100,
            edgecolors="black",
            facecolors="white",
        )

    # plot ballcarrier
    plot_nonmetric_player(
        x_off[ballCarrierId][frame],
        y_off[ballCarrierId][frame],
        ax,
        marker_size=150,
        edgecolors="black",
        facecolors="black",
    )

    # add time elapsed counter
    ax.text(
        ax.get_xlim()[0],
        -2,
        f"Time Elapsed {frame * 0.1:.1f} seconds",
        fontsize=20,
        color="black",
        ha="left",
    )

    # add play description
    if 10 <= len(play_desc.split(" ")) < 20:
        play_desc = (
            " ".join(play_desc.split(" ")[:10])
            + "\n"
            + " ".join(play_desc.split(" ")[10:])
        )
    elif len(play_desc.split(" ")) >= 20:
        play_desc = (
            " ".join(play_desc.split(" ")[:10])
            + "\n"
            + " ".join(play_desc.split(" ")[10:20])
            + "\n"
            + " ".join(play_desc.split(" ")[20:])
        )

    ax.text(
        ax.get_xlim()[0],
        ax.get_ylim()[1] + 1,
        f"GameId = {gameId}, PlayId = {playId} \n {play_desc}",
        ha="left",
        fontsize=20,
    )

    # if missed_tackle_dict provided, annotate it with a red box
    if missed_tackle_dict is not None:
        for id, m_frame in missed_tackle_dict.items():
            if m_frame <= frame:
                rectangle = patches.Rectangle(
                    (x_def[id][m_frame] - 1.5, y_def[id][m_frame] - 1.5),
                    3,
                    3,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="red",
                    fill=False,
                    alpha=1,
                )
                ax.add_patch(rectangle)

    if made_tackle_dict is not None:
        assert (
            len(made_tackle_dict) == 1
        ), "ERROR: Only one made tackle allowed per play"
        for id, m_frame in made_tackle_dict.items():
            if m_frame <= frame:
                rectangle = patches.Rectangle(
                    (x_def[id][m_frame] - 1.5, y_def[id][m_frame] - 1.5),
                    3,
                    3,
                    linewidth=2,
                    edgecolor="green",
                    facecolor="green",
                    fill=False,
                    alpha=1,
                )
                ax.add_patch(rectangle)


def make_play_with_metric_animation(
    gameId,
    playId,
    defenseIds,
    offenseIds,
    df_sequence,
    names,
    highlightIds=[],
    missed_tackle_dict=None,
    made_tackle_dict=None,
):
    # create dictionary of player positions throughout the play
    for id in defenseIds:
        x_def = {
            id: list(df_sequence[df_sequence.nflId == id].x_clean) for id in defenseIds
        }
        y_def = {
            id: list(df_sequence[df_sequence.nflId == id].y_clean) for id in defenseIds
        }

    for id in offenseIds:
        x_off = {
            id: list(df_sequence[df_sequence.nflId == id].x_clean) for id in offenseIds
        }
        y_off = {
            id: list(df_sequence[df_sequence.nflId == id].y_clean) for id in offenseIds
        }

    # Find limits of the field of view for the animation
    bc_id = df_sequence.ballCarrierId.unique()[0]
    x_min = max(x_off[bc_id]) - 35
    x_max = max(x_off[bc_id]) + 15
    # y_min = max(y_off[bc_id]) - 10
    # y_max = max(y_off[bc_id]) + 10
    y_min = 0
    y_max = 53.3

    # Extract Play Description
    df_plays = pd.read_csv(os.path.join(root_dir, "data/plays.csv"))
    desc = df_plays[
        (df_plays.gameId == gameId) & (df_plays.playId == playId)
    ].playDescription.values[0]

    # Create Figure and Animation
    fig, ax = plt.subplots(1, figsize=((x_max - x_min) * 10 / 53.3, 10))
    create_football_field(ax, xlim=[x_min, x_max], ylim=[y_min, y_max])
    cmap = mpl.cm.coolwarm
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    plt.box(on=False)
    ax.set_axis_off()

    # Add Colorbar
    cb = plt.colorbar(
        mpl.cm.ScalarMappable(norm, cmap),
        ax=ax,
        fraction=0.046,
        orientation="vertical",
    )
    cb.set_label(label="Tackle Probability", size=20)
    cb.ax.tick_params(axis="y", labelsize=20)
    animation = FuncAnimation(
        fig,
        partial(
            update_play_animation,
            max_frames=len(frames),
            x_def=x_def,
            y_def=y_def,
            x_off=x_off,
            y_off=y_off,
            ballCarrierId=ballCarrierId,
            defenseIds=defenseIds,
            offenseIds=offenseIds,
            labels=names,
            metric=probs_dict,
            gameId=gameId,
            playId=playId,
            play_desc=desc,
            highlightIds=highlightIds,
            missed_tackle_dict=missed_tackle_dict,
            made_tackle_dict=made_tackle_dict,
            ax=ax,
        ),
        frames=len(frames) + 5,  # add 5 frames to pause at the end
        interval=200,
        repeat=True,
    )
    return animation


# Create animation of tackle probability over time plot
# Create function to call while animating sequence
def update_tackle_probability(
    frame,
    max_frames,
    probs_dict,
    ids,
    names,
    ax,
    highlightIds=None,
    missed_tackle_dict=None,
):
    if frame >= max_frames:  # if frames exceeds max_frames, pause on the last frame
        frame = max_frames - 1
    if highlightIds is None:
        highlightIds = []

    t = np.linspace(0, frame * 0.1, frame + 1)
    ax.cla()
    for id in ids:
        if id in highlightIds:
            ax.plot(t, probs_dict[id][: frame + 1], label=names[id], linewidth=3)
            ax.scatter(t[-1], probs_dict[id][frame], s=50)
        else:
            ax.plot(
                t,
                probs_dict[id][: frame + 1],
                linewidth=3,
                color="gray",
                alpha=0.4,
            )
            ax.scatter(t[-1], probs_dict[id][frame], s=50, color="gray", alpha=0.5)
    ax.set_xlabel("Time (s)", fontsize=24)
    ax.set_ylabel("Tackle Probability in the next 1 second", fontsize=24)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, max_frames * 0.1])
    ax.hlines(0.75, 0, max_frames * 0.1, linestyles="dashed", colors="gray")
    ax.text(
        max_frames * 0.1,
        0.76,
        "Tackle Opportunity Threshold",
        fontsize=20,
        color="gray",
        ha="right",
    )
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="both", which="major", labelsize=24)
    ax.tick_params(axis="both", which="minor", labelsize=24)
    ax.legend(loc="upper left", fancybox=True, fontsize=20)

    ax.set_title("Tackle Probability over Time", y=1.1, fontsize=20)
    ax.set_frame_on(False)

    if missed_tackle_dict is not None:
        first = True
        added = False
        for id, m_frame in missed_tackle_dict.items():
            if m_frame <= frame and first:
                ax.text(
                    m_frame * 0.1,
                    probs_dict[id][m_frame] + 0.03,
                    "Missed Tackle Opportunity",
                    fontdict={
                        "family": "sans-serif",
                        "size": 20,
                        "weight": "normal",
                        "ha": "center",
                        "va": "center",
                        "color": "red",
                    },
                )
                first = False

            if m_frame <= frame:
                ax.scatter(
                    m_frame * 0.1,
                    probs_dict[id][m_frame],
                    color="red",
                    marker="x",
                    s=300,
                    label="Missed Tackle Opp.",
                )
                if not added:
                    ax.legend(loc="upper left", fancybox=True, fontsize=20)
                    added = True


def make_tackle_probability_animation(
    frames,
    probs_dict,
    defenseIds,
    names,
    display=False,
    highlightIds=[],
    missed_tackle_dict=None,
):
    fig, ax = plt.subplots(1, figsize=(15, 10))
    animation = FuncAnimation(
        fig,
        partial(
            update_tackle_probability,
            max_frames=len(frames),
            probs_dict=probs_dict,
            ids=defenseIds,
            names=names,
            highlightIds=highlightIds,
            missed_tackle_dict=missed_tackle_dict,
            ax=ax,
        ),
        frames=len(frames) + 5,  # add 5 frames to pause at the end
        interval=200,
        repeat=True,
    )
    if display:
        plt.show()
    return animation


def combine_gifs(gif1_path, gif2_path, output_path):
    # Read GIFs
    gif1 = imageio.get_reader(gif1_path)
    gif2 = imageio.get_reader(gif2_path)

    # Get frame duration (assuming both GIFs have the same frame duration)
    duration = gif1.get_meta_data()["duration"]

    # Create a writer for the combined GIF
    combined_gif_writer = imageio.get_writer(output_path, duration=duration, loop=0)

    combined_frame = None
    # Combine frames side by side and write
    for frame1, frame2 in zip(gif1, gif2):
        combined_frame = imageio.core.util.Image(
            np.concatenate([frame2, frame1], axis=1)
        )
        combined_gif_writer.append_data(combined_frame)

    # pause on the last frame for 5 frames
    for i in range(5):
        combined_gif_writer.append_data(combined_frame)

    # Close
    combined_gif_writer.close()
    gif1.close()
    gif2.close()


if __name__ == "__main__":
    # USER VARIABLES - CHANGE THESE

    gameId = 2022110700
    playId = 2902
    model_fname = os.path.join(
        root_dir, "tacklenet/model_states/TackleNet_XGB_12282023-1909.model"
    )
    dataset_fname = os.path.join(
        root_dir, "tacklenet/inputs/test_tracking_data_12262023-1754.pkl"
    )

    output_dir = os.path.join(root_dir, "plotting/generated_plots/")

    # These are the players to follow. If none, then it will follow the whole defense
    # defenseIds = [52482, 44851, 48027, 52665, 54514, 43409, 48537, 53505, 37097]   # play 2022110700, 2902
    # defenseIds = [53532, 43294, 41239]  # 2022110609, 271
    defenseIds = None
    offenseIds = None
    # highlightIds = None
    # highlightIds = [53532, 43294, 41239]
    highlightIds = [54514, 52482, 44851, 52665, 48027]
    # dictionary of missed tackles to annotate
    # missed_tackle_dict = None
    missed_tackle_dict = {54514: 32, 52482: 48, 52665: 49, 48027: 40}
    # made_tackle_dict = {43294: 60}
    made_tackle_dict = {44851: 70}
    # made_tackle_dict = None

    # END USER VARIABLES - DO NOT CHANGE ANYTHING BELOW THIS LINE
    # ===========================================================

    # Load TackleTree XGBoost Model from File
    clf = xgb.XGBClassifier()
    clf.load_model(model_fname)

    # Load Dataset to run inference on
    df_tracking = pickle.load(open(dataset_fname, "rb"))

    # Build Tackle Sequences for Inference
    df_sequence = df_tracking[
        (df_tracking.gameId == gameId) & (df_tracking.playId == playId)
    ]
    assert not df_sequence.empty, "ERROR: Play not found."
    restore_geometry(df_sequence)
    if defenseIds is None:
        defenseIds = df_sequence[
            df_sequence.club == df_sequence.defensiveTeam
        ].nflId.unique()
    if offenseIds is None:
        offenseIds = df_sequence[
            df_sequence.club == df_sequence.possessionTeam
        ].nflId.unique()
    ballCarrierId = df_sequence.ballCarrierId.unique()[0]

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
            probs = clf.predict_proba(x)
            outcome_dict[(id, ts.start_frameId)] = probs[
                0, 1
            ]  # probability of a made tackle in the next offset_frames * 0.1 seconds

        frames = [ts.start_frameId for ts in ts_list]
        probs_dict[id] = [outcome_dict[(id, ts.start_frameId)] for ts in ts_list]

    # Extract displayNames to label players
    df_players = pd.read_csv(os.path.join(root_dir, "data/players.csv"))
    names = {}
    for id in defenseIds:
        names[id] = df_players[df_players.nflId == id].displayName.values[0]
    for id in offenseIds:
        names[id] = df_players[df_players.nflId == id].displayName.values[0]

    # create animation of tackle probability plot
    plot_animation = make_tackle_probability_animation(
        frames,
        probs_dict,
        defenseIds,
        names,
        highlightIds=highlightIds,
        missed_tackle_dict=missed_tackle_dict,
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
        defenseIds,
        offenseIds,
        df_sequence,
        names,
        highlightIds=highlightIds,
        missed_tackle_dict=missed_tackle_dict,
        made_tackle_dict=made_tackle_dict,
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
