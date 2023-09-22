"""Utilities shared between regressors."""

from typing import List, Tuple, Optional

import numpy as np
import pandas


def block_hankel(
    X: np.ndarray,
    n_row: Optional[int],
    n_col: int,
    first_feature: int = 0,
    episode_feature: bool = False,
) -> np.ndarray:
    """Form a block Hankel matrix.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Data matrix, where each row is a feature. Must have at least
        ``first_feature + n_row + n_col - 1`` samples in each episode.
    n_row : int
        Number of rows in Hankel matrix for each episode. If ``None``, is set
        to ``n_row = n_samples - first_feature - n_col + 1``.
    n_col : int
        Number of block columns in Hankel matrix for each episode. Full number
        of columns per episode is ``n_col * X.shape[1]``.
    first_feature : int
        First feature of each to use.
    episode_feature : bool
        True if first feature indicates which episode a timestep is from.

    Returns
    -------
    np.ndarray
        Hankel matrix.

    Raises
    ------
    ValueError
        If the any episode in the data matrix does not have enough timesteps.

    Examples
    -------
    The following examples are written out with strings instead of floats to
    make them easier to understand.

    Hankel matrix with no offset:

    >>> X = np.array([
    ...     ['x0'],
    ...     ['x1'],
    ...     ['x2'],
    ...     ['x3'],
    ...     ['x4'],
    ...     ['x5'],
    ...     ['x6'],
    ... ], dtype=object)
    >>> block_hankel(X, n_row=3, n_col=4)
    array([['x0', 'x1', 'x2', 'x3'],
           ['x1', 'x2', 'x3', 'x4'],
           ['x2', 'x3', 'x4', 'x5']], dtype=object)

    Hankel matrix with an offset:

    >>> X = np.array([
    ...     ['x0'],
    ...     ['x1'],
    ...     ['x2'],
    ...     ['x3'],
    ...     ['x4'],
    ...     ['x5'],
    ...     ['x6'],
    ... ], dtype=object)
    >>> block_hankel(X, n_row=3, n_col=4, first_feature=1)
    array([['x1', 'x2', 'x3', 'x4'],
           ['x2', 'x3', 'x4', 'x5'],
           ['x3', 'x4', 'x5', 'x6']], dtype=object)

    Multi-episode Hankel matrix:

    >>> X = np.array([
    ...     [0, 'x0'],
    ...     [0, 'x1'],
    ...     [0, 'x2'],
    ...     [0, 'x3'],
    ...     [0, 'x4'],
    ...     [0, 'x5'],
    ...     [0, 'x6'],
    ...     [1, 'y0'],
    ...     [1, 'y1'],
    ...     [1, 'y2'],
    ...     [1, 'y3'],
    ...     [1, 'y4'],
    ...     [1, 'y5'],
    ...     [1, 'y6'],
    ... ], dtype=object)
    >>> block_hankel(X, n_row=3, n_col=4, first_feature=1,
    ...              episode_feature=True)
    array([['x1', 'x2', 'x3', 'x4'],
           ['x2', 'x3', 'x4', 'x5'],
           ['x3', 'x4', 'x5', 'x6'],
           ['y1', 'y2', 'y3', 'y4'],
           ['y2', 'y3', 'y4', 'y5'],
           ['y3', 'y4', 'y5', 'y6']], dtype=object)
    """
    eps = split_episodes(X, episode_feature=episode_feature)
    hankels = []
    for ep, X_ep in eps:
        # If number of rows is unspecified, set it to use all the data
        if n_row is None:
            n_row_ep = X_ep.shape[0] - first_feature - n_col + 1
        else:
            n_row_ep = n_row
        # Check that there are enough samples
        min_samples = first_feature + n_row_ep + n_col - 1
        if X_ep.shape[0] < min_samples:
            raise ValueError(
                f'Episode {ep} has {X_ep.shape[0]} samples, must have at '
                f'least {min_samples} samples.'
            )
        # Build matrix
        cols = []
        for col in range(first_feature, n_col + first_feature):
            cols.append(X_ep[col:(col + n_row_ep), :])
        H_ep = np.hstack(cols)
        hankels.append(H_ep)
    H = np.vstack(hankels)
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


def extract_output(
    X: np.ndarray,
    n_inputs: int = 0,
    episode_feature: bool = False,
) -> np.ndarray:
    """Extract output from a data matrix.

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
        Output extracted from data matrix.
    """
    if episode_feature:
        n_states = X.shape[1] - n_inputs - 1
        if n_states == 0:
            Y = X[:, [0]]
        else:
            Y = np.hstack((
                X[:, [0]],
                X[:, 1:(n_states + 1)],
            ))
    else:
        n_states = X.shape[1] - n_inputs
        if n_states == 0:
            Y = np.zeros((X.shape[0], 0))
        else:
            Y = X[:, :n_states]
    return Y


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
    if episode_feature:
        if n_inputs == 0:
            U = X[:, [0]]
        else:
            U = np.hstack((
                X[:, [0]],
                X[:, -n_inputs:],
            ))
    else:
        if n_inputs == 0:
            U = np.zeros((X.shape[0], 0))
        else:
            U = X[:, -n_inputs:]
    return U


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
