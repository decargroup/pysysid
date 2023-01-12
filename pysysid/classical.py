"""Classical system identification methods."""

from typing import Any, Dict

import numpy as np
import sklearn.base


class Arx(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """ARX model."""

    # Array check parameters for :func:`fit` when ``X`` and ``y` are given
    _check_X_y_params: Dict[str, Any] = {
        'multi_output': True,
        'y_numeric': True,
    }

    # Array check parameters for :func:`predict` and :func:`fit` when only
    # ``X`` is given
    _check_array_params: Dict[str, Any] = {
        'dtype': 'numeric',
    }

    def __init__(self):
        """Instantiate :class:`Arx`."""
        pass

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        n_inputs: int = 0,
        episode_feature: bool = False,
    ) -> 'Arx':
        """Fit the model.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        y : np.ndarray
            Ignored.
        n_inputs : int
            Number of input features at the end of ``X``.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.

        Returns
        -------
        Arx
            Instance of itself.

        Raises
        -----
        ValueError
            If constructor or fit parameters are incorrect.
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform a single-step prediction for each state in each episode.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        np.ndarray
            Predicted data matrix.
        """
        # TODO WIll single-step prediction work here?
        # TODO How will initial conditions work?
        pass

    # Extra estimator tags
    # https://scikit-learn.org/stable/developers/develop.html#estimator-tags
    def _more_tags(self):
        return {
            'multioutput': True,
            'multioutput_only': True,
        }
