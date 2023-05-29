from agents.ada_agent.utils import *
import os
import pickle

from geniusweb.issuevalue import Bid
from geniusweb.profile.utilityspace import LinearAdditiveUtilitySpace
from geniusweb.progress import ProgressTime


class LearningModel:
    """
        Learning Model
    """
    profile: LinearAdditiveUtilitySpace
    progress: ProgressTime
    received_bids: list                 # Received bids
    my_bids: list                       # Generated bids by Bidding Strategy
    data: dict                          # Data will be saved.

    def __init__(self, profile: LinearAdditiveUtilitySpace, progress: ProgressTime, **kwargs):
        """
            Constructor
        :param profile: Linear additive profile
        :param progress: Negotiation session progress time
        :param kwargs: Additional parameters if needed
        """
        self.profile = profile
        self.progress = progress
        self.received_bids = []
        self.my_bids = []
        self.data = {}

    def receive_bid(self, curr_bid, prev_bid, reward, learning_rate, discount_factor):
        """
            This method is called when a bid is received from the opponent.
        :param curr_bid: Received bid
        :param prev_bid: Previous bid
        :param reward: Reward
        :param learning_rate: Learning rate
        :param discount_factor: Discount factor
        :return: Nothing
        """
        if curr_bid is not None:
            self.received_bids.append(curr_bid)

    def save_bid(self, bid: Bid):
        """
            Save
        :param bid: Offered bid
        :return: Nothing
        """
        self.my_bids.append(bid)

    def reach_agreement(self, accepted_bid: Bid, opponent_accepted: bool):
        """
            This method is called when an agreement is reached.
        :param accepted_bid: Accepted bid
        :param opponent_accepted: If the opponent accepted our bid, or we accepted the opponent's bid.
        :return: Nothing
        """
        time = get_time(self.progress)

        self.accepted_bid = accepted_bid
        self.opponent_accepted = opponent_accepted
        self.acceptance_time = time

    def save_data(self, storage_dir: str, opponent_agent: str):
        """
            This method is called at the end of negotiation session to save learning data
        :param storage_dir: Storage directory
        :param opponent_agent: The name of opponent agent
        :return: Nothing
        """
        # If there is no information
        if opponent_agent is None or storage_dir is None:
            return

        # Save the data as Pickle format.
        with open(f"{storage_dir}/{opponent_agent}_data.pkl", "wb") as f:
            pickle.dump(self.data, f)

    def load_data(self, storage_dir: str, opponent_agent: str) -> dict:
        """
            This method is called at the beginning of the negotiation session to get learned data
        :param storage_dir: Storage directory
        :param opponent_agent: The name of the opponent agent
        :return: Learned data as dictionary
        """
        self.data = {}

        # If there is no information
        if storage_dir is None or opponent_agent is None:
            return self.data

        # Load corresponding data
        if os.path.exists(f"{storage_dir}/{opponent_agent}_data.pkl"):
            with open(f"{storage_dir}/{opponent_agent}_data.pkl", "rb") as f:
                self.data = pickle.load(f)

        return self.data






