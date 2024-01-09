"""
A script that animates tracking data, given gameId and playId. 
Players can be identified by mousing over the individuals dots. 
The play description is also displayed at the bottom of the plot, 
together with play information at the top of the plot. 

Data should be stored in a dir named data, in the same dir as this script. 

Original Source: https://www.kaggle.com/code/huntingdata11/animated-and-interactive-nfl-plays-in-plotly/notebook
"""

import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np

pio.renderers.default = (
    "browser"  # modify this to plot on something else besides browser
)

# Modify the variables below to plot your desired play
tracking_file = "data/tracking_week_9.csv"
plays_file = "data/plays.csv"
game_id = 2022110610
play_id = 1066

# Test cases:
# 2022090800, 1385 = going left, on attacking half
# 2022090800, 80 = going left, on defending half
# 2022090800, 775 = going right, on defending half
# 2022091100, 914 = going right, on attacking half

# team colors to distinguish between players on plots
colors = {
    "ARI": "#97233F",
    "ATL": "#A71930",
    "BAL": "#241773",
    "BUF": "#00338D",
    "CAR": "#0085CA",
    "CHI": "#C83803",
    "CIN": "#FB4F14",
    "CLE": "#311D00",
    "DAL": "#003594",
    "DEN": "#FB4F14",
    "DET": "#0076B6",
    "GB": "#203731",
    "HOU": "#03202F",
    "IND": "#002C5F",
    "JAX": "#9F792C",
    "KC": "#E31837",
    "LA": "#FFA300",
    "LAC": "#0080C6",
    "LV": "#000000",
    "MIA": "#008E97",
    "MIN": "#4F2683",
    "NE": "#002244",
    "NO": "#D3BC8D",
    "NYG": "#0B2265",
    "NYJ": "#125740",
    "PHI": "#004C54",
    "PIT": "#FFB612",
    "SEA": "#69BE28",
    "SF": "#AA0000",
    "TB": "#D50A0A",
    "TEN": "#4B92DB",
    "WAS": "#5A1414",
    "football": "#CBB67C",
    "tackle": "#FFC0CB",
}

# Handle Data I/O
df_tracking = pd.read_csv(tracking_file)
df_plays = pd.read_csv(plays_file)

df_full_tracking = df_tracking.merge(df_plays, on=["gameId", "playId"])

df_focused = df_full_tracking[
    (df_full_tracking["playId"] == play_id) & (df_full_tracking["gameId"] == game_id)
]

# Get General Play Information
absolute_yd_line = df_focused.absoluteYardlineNumber.values[0]
play_going_right = (
    df_focused.playDirection.values[0] == "right"
)  # 0 if left, 1 if right

line_of_scrimmage = absolute_yd_line

# place LOS depending on play direction and absolute_yd_line. 110 because absolute_yd_line includes endzone width

first_down_marker = (
    (line_of_scrimmage + df_focused.yardsToGo.values[0])
    if play_going_right
    else (line_of_scrimmage - df_focused.yardsToGo.values[0])
)  # Calculate 1st down marker

down = df_focused.down.values[0]
quarter = df_focused.quarter.values[0]
gameClock = df_focused.gameClock.values[0]
playDescription = df_focused.playDescription.values[0]
ballcarrierId = df_focused.ballCarrierId.values[0]
tackle_frame_id = -1

# Handle case where we have a really long Play Description and want to split it into two lines
if len(playDescription.split(" ")) > 15 and len(playDescription) > 115:
    playDescription = (
        " ".join(playDescription.split(" ")[0:16])
        + "<br>"
        + " ".join(playDescription.split(" ")[16:])
    )

print(
    f"Line of Scrimmage: {line_of_scrimmage}, First Down Marker: {first_down_marker}, Down: {down}, Quarter: {quarter}, Game Clock: {gameClock}, Play Description: {playDescription}"
)

# initialize plotly play and pause buttons for animation
updatemenus_dict = [
    {
        "buttons": [
            {
                "args": [
                    None,
                    {
                        "frame": {"duration": 100, "redraw": False},
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                    },
                ],
                "label": "Play",
                "method": "animate",
            },
            {
                "args": [
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
                "label": "Pause",
                "method": "animate",
            },
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top",
    }
]

# initialize plotly slider to show frame position in animation
sliders_dict = {
    "active": 0,
    "yanchor": "top",
    "xanchor": "left",
    "currentvalue": {
        "font": {"size": 20},
        "prefix": "Frame:",
        "visible": True,
        "xanchor": "right",
    },
    "transition": {"duration": 300, "easing": "cubic-in-out"},
    "pad": {"b": 10, "t": 50},
    "len": 0.9,
    "x": 0.1,
    "y": 0,
    "steps": [],
}

# Frame Info
sorted_frame_list = df_focused.frameId.unique()
sorted_frame_list.sort()

frames = []
for frameId in sorted_frame_list:
    data = []
    # Add Yardline Numbers to Field
    data.append(
        go.Scatter(
            x=np.arange(20, 110, 10),
            y=[5] * len(np.arange(20, 110, 10)),
            mode="text",
            text=list(
                map(str, list(np.arange(20, 61, 10) - 10) + list(np.arange(40, 9, -10)))
            ),
            textfont_size=30,
            textfont_family="Courier New, monospace",
            textfont_color="#ffffff",
            showlegend=False,
            hoverinfo="none",
        )
    )
    data.append(
        go.Scatter(
            x=np.arange(20, 110, 10),
            y=[53.5 - 5] * len(np.arange(20, 110, 10)),
            mode="text",
            text=list(
                map(str, list(np.arange(20, 61, 10) - 10) + list(np.arange(40, 9, -10)))
            ),
            textfont_size=30,
            textfont_family="Courier New, monospace",
            textfont_color="#ffffff",
            showlegend=False,
            hoverinfo="none",
        )
    )
    # Add line of scrimage
    data.append(
        go.Scatter(
            x=[line_of_scrimmage, line_of_scrimmage],
            y=[0, 53.5],
            line_dash="dash",
            line_color="blue",
            showlegend=False,
            hoverinfo="none",
        )
    )
    # Add First down line
    data.append(
        go.Scatter(
            x=[first_down_marker, first_down_marker],
            y=[0, 53.5],
            line_dash="dash",
            line_color="yellow",
            showlegend=False,
            hoverinfo="none",
        )
    )
    # Plot Players
    for club in df_focused.club.unique():
        plot_df = df_focused[
            (df_focused.club == club) & (df_focused.frameId == frameId)
        ].copy()
        if club != "football":
            hover_text_array = []
            for nflId in plot_df.nflId:
                selected_player_df = plot_df[plot_df.nflId == nflId]
                hover_text_array.append(
                    f"nflId:{selected_player_df['nflId'].values[0]}<br>displayName:{selected_player_df['displayName'].values[0]}"
                )
            data.append(
                go.Scatter(
                    x=plot_df["x"],
                    y=plot_df["y"],
                    mode="markers",
                    marker_color=colors[club],
                    marker_size=10,
                    name=club,
                    hovertext=hover_text_array,
                    hoverinfo="text",
                )
            )
            if (
                plot_df.event.values[0] == "tackle"
                and club == plot_df.possessionTeam.values[0]
            ):
                tackle_frame_id = frameId
                ballcarrier_df = df_focused[
                    (df_focused.nflId == ballcarrierId)
                    & (df_focused.frameId == frameId)
                ].copy()
                data.append(
                    go.Scatter(
                        x=ballcarrier_df["x"],
                        y=ballcarrier_df["y"],
                        mode="markers",
                        marker_color=colors["tackle"],
                        marker_size=25,
                        name="tackle",
                        hovertext=["Tackle"],
                        hoverinfo="text",
                    )
                )
        else:
            data.append(
                go.Scatter(
                    x=plot_df["x"],
                    y=plot_df["y"],
                    mode="markers",
                    marker_color=colors[club],
                    marker_size=10,
                    name=club,
                    hoverinfo="none",
                )
            )

    # add frame to slider
    slider_step = {
        "args": [
            [frameId],
            {
                "frame": {"duration": 100, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": 0},
            },
        ],
        "label": str(frameId),
        "method": "animate",
    }
    sliders_dict["steps"].append(slider_step)
    frames.append(go.Frame(data=data, name=str(frameId)))

scale = 10
layout = go.Layout(
    autosize=False,
    width=120 * scale,
    height=60 * scale,
    xaxis=dict(
        range=[0, 120],
        autorange=False,
        tickmode="array",
        tickvals=np.arange(10, 111, 5).tolist(),
        showticklabels=False,
    ),
    yaxis=dict(range=[0, 53.3], autorange=False, showgrid=False, showticklabels=False),
    plot_bgcolor="#00B140",
    # Create title and add play description at the bottom of the chart for better visual appeal
    title=f"GameId: {game_id}, PlayId: {play_id}<br>{gameClock} {quarter}Q, Tackled at Frame {tackle_frame_id}"
    + "<br>" * 19
    + f"{playDescription}",
    updatemenus=updatemenus_dict,
    sliders=[sliders_dict],
)

fig = go.Figure(data=frames[0]["data"], layout=layout, frames=frames[1:])

# Create First Down Markers
for y_val in [0, 53]:
    fig.add_annotation(
        x=first_down_marker,
        y=y_val,
        text=str(down),
        showarrow=False,
        font=dict(family="Courier New, monospace", size=16, color="black"),
        align="center",
        bordercolor="black",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=1,
    )

fig.show()
