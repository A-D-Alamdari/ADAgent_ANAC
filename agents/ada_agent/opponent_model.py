from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.DiscreteValueSet import DiscreteValueSet
from geniusweb.issuevalue.Domain import Domain
from geniusweb.issuevalue.Value import Value
from agents.ada_agent.utils import *
from geniusweb.profile.utilityspace import LinearAdditiveUtilitySpace
from geniusweb.progress import ProgressTime

import numpy as np


class OpponentModel:
    """
    Opponent Model using Reinforcement Learning (Q-learning)
    """
    profile: LinearAdditiveUtilitySpace
    progress: ProgressTime
    offers: list  # Received bids
    domain: Domain  # Agent's domain
    issues: dict  # Issues
    alpha: float  # Learning rate
    gamma: float  # Discount factor
    q_table: dict  # Q-table

    def __init__(self, domain: Domain, profile: LinearAdditiveUtilitySpace, progress: ProgressTime,
                 alpha: float = 0.1, gamma: float = 0.9):
        """
        Constructor
        :param domain: Negotiation domain
        :param profile: Linear additive profile
        :param progress: Negotiation session progress time
        :param alpha: Learning rate (default: 0.1)
        :param gamma: Discount factor (default: 0.9)
        """
        self.domain = domain
        self.profile = profile
        self.progress = progress
        self.alpha = alpha
        self.gamma = gamma
        self.offers = []  # List to store received bids

        self.issues = {issue: Issue(values) for issue, values in domain.getIssuesValues().items()}
        self.normalize()

        # Initialize Q-table
        self.q_table = {}

    def update(self, prev_bid: Bid, curr_bid: Bid, reward: float, learning_rate: float, discount_factor: float):
        """
        This method is called when a new bid is received. It updates the opponent model and performs Q-learning update.
        :param prev_bid: Previous bid
        :param curr_bid: Current bid
        :param reward: Reward associated with the current bid
        :param learning_rate: Learning rate for Q-learning update
        :param discount_factor: Discount factor for Q-learning update
        :return: Nothing
        """
        if prev_bid is None or curr_bid is None:
            return

        self.offers.append(curr_bid)

        self.update_issue_weights()

        self.q_learning_update(prev_bid, curr_bid, reward, learning_rate, discount_factor)

    def update_issue_weights(self):
        """
        This method updates the issue weights by considering the last two consecutive bids.
        :return: Nothing
        """
        t = get_time(self.progress)

        if len(self.offers) < 2:
            return

        last_two_bids = self.offers[-2:]

        for issue_name, issue_obj in self.issues.items():
            if last_two_bids[0].getValue(issue_name) == last_two_bids[1].getValue(issue_name):
                issue_obj.weight += self.alpha * (1.0 - t * 3)

    def q_learning_update(self, prev_bid: Bid, curr_bid: Bid, reward: float, learning_rate: float,
                          discount_factor: float):
        prev_state = self.get_state_representation(prev_bid)
        prev_action = prev_bid.getValue(list(self.domain.getIssues())[0]) if prev_bid else None
        curr_state = self.get_state_representation(curr_bid)
        curr_action = curr_bid.getValue(list(self.domain.getIssues())[0]) if curr_bid else None

        # Update Q-table
        prev_q_value = self.q_table.get((prev_state, prev_action), 0.0)

        issue_name = list(self.domain.getIssues())[0]
        values = self.domain.getValues(issue_name)
        max_q_value = max(self.q_table.get((curr_state, a), 0.0) for a in values)

        new_q_value = (1 - learning_rate) * prev_q_value + learning_rate * (reward + discount_factor * max_q_value)
        self.q_table[(prev_state, prev_action)] = new_q_value

    # def get_state_representation(self, bid: Bid) -> tuple:
    #     """
    #     Convert a bid into a state representation.
    #     :param bid: Bid object
    #     :return: State representation (tuple)
    #     """
    #     # TODO: Implement the state representation based on your negotiation scenario
    #     # Return a tuple that represents the current state given the bid
    def get_state_representation(self, bid: Bid) -> tuple:
        """
        Convert a bid into a state representation.
        :param bid: Bid object
        :return: State representation (tuple)
        """
        state = []

        price_value = bid.getValue('price')
        state.append(price_value)

        quantity_value = bid.getValue('quantity')
        state.append(quantity_value)

        return tuple(state)

    def get_utility(self, bid: Bid) -> float:
        """
        Calculate the estimated utility.
        :param bid: The bid to be calculated.
        :return: Estimated utility
        """
        if bid is None:
            return 0.0

        total = 0.0

        for issue_name, issue_obj in self.issues.items():
            total += issue_obj.get_utility(bid.getValue(issue_name))

        return total

    def normalize(self):
        """
        Normalize the value and issue weights.
        - Value weight must be in range [0.0, 1.0]
        - The sum of issue weights must be 1.0
        :return: Nothing
        """
        total_issue_weight = 0.0
        for issue_name, issue_obj in self.issues.items():
            total_issue_weight += issue_obj.weight
            max_val = max(issue_obj.value_weights.values())

            for value_name in issue_obj.value_weights:
                issue_obj.value_weights[value_name] /= max_val

        for issue_obj in self.issues.values():
            issue_obj.weight /= total_issue_weight


class Issue:
    """
        This class can be used to estimate issue weight and value weights.
    """
    weight: float = 1.0  # Issue Weight
    value_weights: dict  # Value Weights

    def __init__(self, values: DiscreteValueSet, **kwargs):
        """
            Constructor
        :param values: The set of discrete value set
        :param kwargs: Additional parameters if needed
        """
        # Initial value weights are zero
        self.value_weights = {value: 0.0 + 1e-10 for value in values}

    def update(self, value: Value, **kwargs):
        """
            This method will be called when a bid received.
        :param value: Received bid
        :param kwargs: Additional parameters if needed
        :return: None
        """
        if value is None:
            return

        self.value_weights[value] += 1.

    def get_utility(self, value: Value) -> float:
        """
            Calculate estimated utility of the issue with value
        :param value: The value of Issue
        :return: Estimated Utility
        """
        if value is None:
            return 0.0

        return self.weight * self.value_weights[value]



