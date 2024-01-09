import torch
import pandas as pd
from torch.utils.data import Dataset


class TackleSequence:
    """
    Wrapper class that goes around a pytorch tensor that represents the input features to TackleNet.

    """

    def __init__(self, df_sequence: pd.DataFrame, target: int, tacklerId: int):
        """
        :param frames (pd.DataFrame): A dataframe containing all the tracking data for a sequence of frames from a single play
        :param target (int): 1 for made tackle, 0 for missed tackle
        :param tacklerId (int): the nflId of the player who made or missed the tackle
        """
        self.playId = df_sequence.playId.iloc[0]
        self.gameId = df_sequence.gameId.iloc[0]
        self.start_frameId = df_sequence.frameId.min()
        self.end_frameId = df_sequence.frameId.max()
        self.ballCarrierId = df_sequence.ballCarrierId.iloc[0]
        self.tacklerId = tacklerId
        self.target = target

        assert (
            self.tacklerId in df_sequence.nflId.values
        ), "tacklerId must be in the frames"
        assert (
            self.ballCarrierId in df_sequence.nflId.values
        ), "ballCarrierId must be in the frames"
        assert (
            self.end_frameId >= self.start_frameId
        ), "end_frameId must be greater than or equal to start_frameId"

        self.input_tensor = None

    def __repr__(self) -> str:
        return f"TackleSequence(gameId={self.gameId}, playId={self.playId}, start_frameId={self.start_frameId}, end_frameId={self.end_frameId}, ballCarrierId={self.ballCarrierId}, tacklerId={self.tacklerId}, target={self.target})"

    def __len__(self) -> int:
        return self.end_frameId - self.start_frameId + 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.input_tensor[idx]


class TackleSequenceDataset(Dataset):
    """
    A dataset class for Tackle Sequences
    """

    def __init__(self, tacklesequence_list):
        self.ts_list = tacklesequence_list

    def __len__(self):
        return len(self.ts_list)

    def __getitem__(self, idx):
        input_tensor = self.ts_list[idx].input_tensor
        target = self.ts_list[idx].target

        return input_tensor, target, idx
