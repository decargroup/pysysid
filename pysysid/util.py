"""Utilities shared between regressors."""

from typing import List, Tuple

import numpy as np
import pandas


def hankel(X: np.ndarray, first_row: int, n_brow: int,
           n_col: int) -> np.ndarray:
    """Form a block Hankel matrix from data.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    first_row : np.ndarray
        First row index to use.
    n_brow : int
        Number of (block) rows. Full number of rows is `n_brow * X.shape[0]`.
    n_col : int
        Number of columns in Hankel matrix.

    Returns
    -------
    np.ndarray
        Hankel matrix.

    Raises
    ------
    ValueError
        If the data matrix does not have enough timesteps.
    """
    if X.shape[1] < (first_row + n_brow + n_col - 1):
        raise ValueError(
            '`X` must have at least `first_row + n_brow + n_col` timesteps.')
    rows = []
    for row in range(first_row, first_row + n_brow):
        rows.append(X[:, row:(row + n_col)])
    H = np.vstack(rows)
    return H


def extract_initial_conditions(
    X: np.ndarray,
    min_samples: int = 1,
    n_inputs: int = 0,
    episode_feature: bool = False,
) -> np.ndarray:
    """Extract initial conditions from each episode.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    min_samples : int
        Number of samples in initial condition.
    n_inputs : int
        Number of input features at the end of ``X``.
    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    np.ndarray
        Initial conditions from each episode.
    """
    episodes = split_episodes(X, episode_feature=episode_feature)
    # Strip each episode
    initial_conditions = []
    for (i, X_i) in episodes:
        if n_inputs == 0:
            initial_condition = X_i[:min_samples, :]
        else:
            initial_condition = X_i[:min_samples, :-n_inputs]
        initial_conditions.append((i, initial_condition))
    # Concatenate the initial conditions
    X0 = combine_episodes(initial_conditions, episode_feature=episode_feature)
    return X0


def extract_input(
    X: np.ndarray,
    n_inputs: int = 0,
    episode_feature: bool = False,
) -> np.ndarray:
    """Extract input from a data matrix.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    n_inputs : int
        Number of input features at the end of ``X``.
    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    np.ndarray
        Input extracted from data matrix.
    """
    episodes = split_episodes(X, episode_feature=episode_feature)
    # Strip each episode
    inputs = []
    for (i, X_i) in episodes:
        if n_inputs == 0:
            input_ = np.zeros((X_i.shape[0], 0))
        else:
            n_states = X_i.shape[1] - n_inputs
            input_ = X_i[:, n_states:]
        inputs.append((i, input_))
    # Concatenate the inputs
    u = combine_episodes(inputs, episode_feature=episode_feature)
    return u


def strip_initial_conditions(
    X: np.ndarray,
    min_samples: int = 1,
    episode_feature: bool = False,
) -> np.ndarray:
    """Strip initial conditions from each episode.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    min_samples : int
        Number of samples in initial condition.
    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    np.ndarray
        Data matrix with initial conditions removed.
    """
    episodes = split_episodes(X, episode_feature=episode_feature)
    # Strip each episode
    stripped_episodes = []
    for (i, X_i) in episodes:
        stripped_episode = X_i[min_samples:, :]
        stripped_episodes.append((i, stripped_episode))
    # Concatenate the stripped episodes
    Xs = combine_episodes(stripped_episodes, episode_feature=episode_feature)
    return Xs


def split_episodes(
    X: np.ndarray,
    episode_feature: bool = False,
) -> List[Tuple[float, np.ndarray]]:
    """Split a data matrix into episodes.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    List[Tuple[float, np.ndarray]]
        List of episode tuples. The first element of each tuple contains the
        episode index. The second element contains the episode data.
    """
    # Extract episode feature
    if episode_feature:
        X_ep = X[:, 0]
        X = X[:, 1:]
    else:
        X_ep = np.zeros((X.shape[0], ))
    # Split X into list of episodes. Each episode is a tuple containing
    # its index and its associated data matrix.
    episodes = []
    # ``pandas.unique`` is faster than ``np.unique`` and preserves order.
    for i in pandas.unique(X_ep):
        episodes.append((i, X[X_ep == i, :]))
    # Return list of episodes
    return episodes


def combine_episodes(episodes: List[Tuple[float, np.ndarray]],
                     episode_feature: bool = False) -> np.ndarray:
    """Combine episodes into a data matrix.

    Parameters
    ----------
    episodes : List[Tuple[float, np.ndarray]]
        List of episode tuples. The first element of each tuple contains the
        episode index. The second element contains the episode data.
    episode_feature : bool
        True if first feature of output should indicate which episode a
        timestep is from.

    Returns
    -------
    np.ndarray
        Combined data matrix.
    """
    combined_episodes = []
    for (i, X) in episodes:
        if episode_feature:
            combined_episodes.append(
                np.hstack((i * np.ones((X.shape[0], 1)), X)))
        else:
            combined_episodes.append(X)
    # Concatenate the combined episodes
    Xc = np.vstack(combined_episodes)
    return Xc
