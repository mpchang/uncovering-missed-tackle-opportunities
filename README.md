Code for the [NFL Big Data Bowl 2024](https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/data) submission "[Uncovering Tackle Opportunities and Missed Opportunities](https://www.kaggle.com/code/matthewpchang/uncovering-missed-tackle-opportunities)".

Created and maintained by [@mpchang](https://github.com/mpchang), [@katdai](https://github.com/katdai), [@bolongcheng](https://bolongcheng.com/), [@danielrjiang](https://danielrjiang.github.io/)

# Intro

Traditional metrics such as made and missed tackles offer only a surface-level understanding of a defender's tackling skill. Before attempting a tackle, a defender engages in a long series of steps: accurately predicting the ball carrier's path, strategically positioning himself, and forcing the ball carrier into a vulnerable position for a tackle or push out of bounds. The outcome, whether a made or missed tackle, marks the end of a complex process that unfolds throughout the play.

We present a new set of metrics to analyze how well defenders perform in the tackling process:

- **Tackle Probability**: Probability that defender X tackles the ball carrier within T seconds (blue line in Figure 1). T is a tunable parameter, which is set to 1 second in this work.
- **Tackle Opportunity**: When defender X's tackle probability on a play exceeds 75% for $>0.5$ second interval. This represents a window of time over which the defender has a real opportunity to make a tackle.
- **Missed Tackle Opportunity**: When defender X's tackle probability on a play exceeds 75% for $>0.5$ second interval and subsequently falls below 75% for $>0.5$ second interval . This occurs when neither defender X nor any of their teammates makes a tackle during the tackle opportunity. This is a new class of defensive mistake not captured by current tackling metrics.
- **Converted Tackle Opportunity**: When defender X is assigned a tackle in the tackle data.

## Set Up

Create a virtual environment. It's easiest to use Miniconda to manage Python and dependencies.

You can follow the instruction [here](https://docs.conda.io/projects/miniconda/en/latest/index.html#quick-command-line-install) to download and install Miniconda on your machine.

This project uses Python 3.10. Create an environment with

```bash
conda create -n tackle-probability python=3.10
```

Activate the virtual environment with

```bash
conda activate tackle-probability
```

Install all python dependencies

```
pip install -r requirements.txt
```

## Running Instructions

To train the model, run inference on plays, and extract metrics by player, follow these instructions:

1. **Add raw data files to the code/data/ directory**. Tracking data, tackle data, play data, and player data are all required. Download the raw data from Kaggle: https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/data
2. **Execute load_data.ipynb.** This notebook will load and process the data into TackleSequence objects that can be consumed by the model.
3. **Execute train_model.ipynb.** This notebook will split the data into train/test, train the model, and evaluate it on the test set.
4. **Execute run_inference.ipynb** This notebook will use the pre-trained model to run inference on an entire play, and generate nice visualizations including tackle probability charts and birds-eye view heatmaps for select defenders on each play.
5. **Execute build_metrics.ipynb.** This notebook will use the pre-trained model to extract tackle opportunities, missed tackle opportunities, and tackle conversions from all tracking data, and create visualizations. **Note, this code takes over 5 hours to run on an M1 Macbook Pro.**
6. (Optional). **Execute spatial_analysis.ipynb.** This notebook is used to produce visualizations of the forward voronoi area, team influence, and blocker influence spatial features.
