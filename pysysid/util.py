"""Utilities shared between regressors."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.interpolate
import scipy.signal

from . import dynamic_models


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
        to ``n_row = n_samples - first_feature - n_col + 1``. If negative, is
        set to the maximum minus ``n_row``.
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
        if n_row is None:
            # If number of rows is unspecified, set it to use all the data
            n_row_ep = X_ep.shape[0] - first_feature - n_col + 1
        elif n_row < 0:
            # If number of rows is negative, use max minus that many rows
            n_row_ep = X_ep.shape[0] - first_feature - n_col + 1 - (-n_row)
        else:
            n_row_ep = n_row
        # Check that there are enough samples
        min_samples = first_feature + n_row_ep + n_col - 1
        if X_ep.shape[0] < min_samples:
            raise ValueError(
                f'Episode {ep} has {X_ep.shape[0]} samples, must have at '
                f'least {min_samples} samples.')
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
        # Split X into list of episodes. Each episode is a tuple containing
        # its index and its associated data matrix.
        episodes = []
        for i in unique_episodes(X_ep):
            episodes.append((i, X[X_ep == i, :]))
    else:
        episodes = [(0, X)]
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


def random_state(low, high, rng=None):
    """Generate a random initial state.

    Generates uniform random data between specified bounds.

    Very simply wrapper. Really only exists to keep a common interface with
    `random_input`, which is much more complex.

    Parameters
    ----------
    low : float or (n, 1) np.ndarray
        Lower bound for uniform random distribution.
    high : float or (n, 1) np.ndarray
        Upper bound for uniform random distribution.
    rng : Optional[Generator]
        Random number generator, `numpy.random.default_rng(seed)`.

    Returns
    -------
    np.ndarray:
        Random initial state.

    Examples
    --------
    Simulate a mass-spring-damper with random initial condition

    >>> msd = dynamic_models.MassSpringDamper(
    ...     mass=0.5,
    ...     stiffness=0.7,
    ...     damping=0.6,
    ... )
    >>> x_max = np.array([1, 1])
    >>> x0 = pysysid.random_state(-x_max, x_max)
    >>> t, x = msd.simulate((0, 1), 1e-3, x0, lambda t: 0)
    """
    if rng is None:
        rng = np.random.default_rng()
    x_rand = rng.uniform(low, high, low.shape)
    return x_rand


def random_input(
    t_range,
    t_step,
    low,
    high,
    cutoff,
    order=2,
    rng=None,
    output='function',
):
    """Generate a smooth random input.

    Generates uniform random data between specified bounds, lowpass filters the
    data, then optionally linearly interpolates to return a function of time.

    Uses a Butterworth filter of specified order.

    Parameters
    ----------
    t_range : (2,) tuple
        Start and end times in a tuple (s).
    t_step : float
        Time step at which to generate random data (s).
    low : float or (n, 1) np.ndarray
        Lower bound for uniform random distribution.
    high : float or (n, 1) np.ndarray
        Upper bound for uniform random distribution.
    cutoff : float
        Cutoff frequency for Butterworth lowpass filter (Hz).
    order : int
        Order of Butterworth lowpass filter.
    rng : Generator
        Random number generator, `numpy.random.default_rng(seed)`.
    output : str
        Output format to use. Value 'array' causes the function to return an
        array of smoothed data. Value 'function' causes the function to return
        a function generated by linearly interpolating that same array.

    Returns
    -------
    function or np.ndarray :
        If `output` is 'function', returns a function representing
        linearly-interpolated lowpass-filtered uniformly-random data. If
        `output` is 'array', returns an array containing lowpass-filtered
        uniformly-random data. Units are same as `low` and `high`.

    Examples
    --------
    Simulate a mass-spring-damper with random input

    >>> msd = dynamic_models.MassSpringDamper(
    ...     mass=0.5,
    ...     stiffness=0.7,
    ...     damping=0.6,
    ... )
    >>> t_range = (0, 1)
    >>> t_step = 1e-3
    >>> x0 = np.array([0, 0])
    >>> u_max = np.array([1])
    >>> u = pysysid.random_input(t_range, t_step, -u_max, u_max, cutoff=0.01)
    >>> t, x = msd.simulate(t_range, t_step, x0, u)
    """
    t = np.arange(*t_range, t_step)
    size = np.shape(low) + (t.shape[-1], )  # Concatenate tuples
    if rng is None:
        rng = np.random.default_rng()
    u_rough = rng.uniform(np.reshape(low, size[:-1] + (1, )),
                          np.reshape(high, size[:-1] + (1, )), size)
    sos = scipy.signal.butter(order, cutoff, output='sos', fs=1 / t_step)
    u_smooth = scipy.signal.sosfilt(sos, u_rough)
    if output == 'array':
        return u_smooth
    elif output == 'function':
        f_smooth = scipy.interpolate.interp1d(
            t,
            u_smooth,
            fill_value='extrapolate',
        )
        return f_smooth
    else:
        raise ValueError(f'{output} is not a valid output form.')


def unique_episodes(X_ep: np.ndarray) -> np.ndarray:
    """Find all the unique episodes in an episode feature array.

    Parameters
    ----------
    X_ep : np.ndarray
        Episode feature (as would be passed to :func:`KoopmanPipeine.fit()`).

    Returns
    -------
    np.ndarray
        List of unique episode indices.

    Raises
    ------
    ValueError
        If episode feature contains negative or fractional numbers.
    """
    if np.any((X_ep % 1) != 0) or np.any(X_ep < 0):
        raise ValueError(
            'Episode feature must contain only positive whole numbers.')
    return np.flatnonzero(np.bincount(X_ep.astype(int)))


def example_data_msd() -> Dict[str, Any]:
    """Get example mass-spring-damper data.

    Returns
    -------
    Dict[str, Any]
        Sample mass-spring damper data.
    """
    # Create mass-spring-damper object
    msd = dynamic_models.MassSpringDamper(
        mass=0.5,
        stiffness=0.7,
        damping=0.6,
    )
    # Set timestep
    n_ep = 10
    n_train = 9
    t_range = (0, 10)
    t_step = 0.01
    t_sim = np.arange(*t_range, t_step)
    # Simulate episodes
    rng = np.random.default_rng(seed=2432)
    X_msd_lst = []
    for ep in range(n_ep):
        # Simulate ODE
        u = random_input(
            t_range,
            t_step,
            low=-1,
            high=1,
            cutoff=(0.01 / t_step),
            rng=rng,
        )
        t, x = msd.simulate(
            t_range=t_range,
            t_step=t_step,
            x0=np.zeros((2, )),
            u=u,
            rtol=1e-8,
            atol=1e-8,
        )
        # Format the data
        X_msd_lst.append(
            np.hstack((
                ep * np.ones((t.shape[0], 1)),
                x,
                np.reshape(u(t), (-1, 1)),
            )))
    # Stack data and return
    X_msd = np.vstack(X_msd_lst)
    X_train = X_msd[X_msd[:, 0] < n_train, :]
    X_valid = X_msd[X_msd[:, 0] >= n_train, :]
    n_inputs = 1
    episode_feature = True
    x0_valid = extract_initial_conditions(
        X_valid,
        n_inputs=n_inputs,
        episode_feature=episode_feature,
    )
    u_valid = extract_input(
        X_valid,
        n_inputs=n_inputs,
        episode_feature=episode_feature,
    )
    return {
        'X_train': X_train,
        'X_valid': X_valid,
        'x0_valid': x0_valid,
        'u_valid': u_valid,
        'n_inputs': n_inputs,
        'episode_feature': episode_feature,
        't': t_sim,
        'dynamic_model': msd,
    }
