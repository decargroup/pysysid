"""Classical system identification methods."""

from typing import Any, Dict, Optional

import numpy as np
import scipy.linalg
import sklearn.base
# import sklearn.linear_model
import sklearn.utils.validation

from . import util


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

    def __init__(
        self,
        n_lags_input: int = 0,
        n_lags_output: int = 1,
    ) -> None:
        """Instantiate :class:`Arx`.

        Parameters
        ----------
        n_lags_input : int
            TODO
        n_lags_output : int
            TODO
        """
        self.n_lags_input = n_lags_input
        self.n_lags_output = n_lags_output

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
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
        # Check data
        X = sklearn.utils.validation.check_array(X)
        # Check inputs
        if n_inputs < 0:
            raise ValueError('`n_inputs` must be greater than or equal to 0.')
        # Check parameters
        if self.n_lags_input < 0:
            raise ValueError('`n_lags_input` must be positive.')
        if self.n_lags_output < 0:
            raise ValueError('`n_lags_output` must be positive.')
        if self.n_lags_input > self.n_lags_output + 1:
            raise ValueError(
                '`n_lags_input` must be less than `n_lags_output + 1`.')
        # Save fit information
        self.n_features_in_ = X.shape[1]
        n_episode_features = 1 if episode_feature else 0
        self.n_outputs_in_ = X.shape[1] - n_inputs - n_episode_features
        self.n_inputs_in_ = n_inputs
        self.episode_feature_ = episode_feature
        self.min_samples_ = max(self.n_lags_output + 1, self.n_lags_input)
        # Split data
        Y = util.extract_output(
            X,
            n_inputs=self.n_inputs_in_,
            episode_feature=self.episode_feature_,
        )
        U = util.extract_input(
            X,
            n_inputs=self.n_inputs_in_,
            episode_feature=self.episode_feature_,
        )
        # Formulate least squares problem
        H_Y_future = util.block_hankel(
            Y,
            n_row=None,
            n_col=1,
            first_feature=self.n_lags_output,
            episode_feature=self.episode_feature_,
        )
        H_Y_past = util.block_hankel(
            Y,
            n_row=-1,
            n_col=self.n_lags_output,
            first_feature=0,
            episode_feature=self.episode_feature_,
        )
        # TODO Handle n_lags_input=0 later
        H_U_past = util.block_hankel(
            U,
            n_row=None,
            n_col=self.n_lags_input,
            first_feature=(self.n_lags_output - self.n_lags_input + 1),
            episode_feature=self.episode_feature_,
        )
        H_past = np.hstack((-H_Y_past, H_U_past))
        coefs = scipy.linalg.lstsq(H_past, H_Y_future)[0].T
        # est = sklearn.linear_model.Ridge(alpha=0.01, fit_intercept=False)
        # est.fit(H_past, H_Y_future)
        # coefs = est.coef_
        self.coef_Y_ = np.split(
            coefs[:, :(self.n_outputs_in_ * self.n_lags_output)],
            self.n_lags_output,
            axis=-1,
        )
        self.coef_U_ = np.split(
            coefs[:, (self.n_outputs_in_ * self.n_lags_output):],
            self.n_lags_input,
            axis=-1,
        )
        return self

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

        Raises
        ------
        ValueError
            If an episode does not have enough samples.
        """
        # Check if fitted
        sklearn.utils.validation.check_is_fitted(self)
        # Validate array
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Split episodes
        episodes = util.split_episodes(
            X,
            episode_feature=self.episode_feature_,
        )
        # Predict for each episode
        predictions = []
        for (i, X_i) in episodes:
            if X_i.shape[0] < self.min_samples_:
                raise ValueError(f'Episode {i} has {X_i.shape[0]} samples but '
                                 f'`min_samples_`={self.min_samples_} samples '
                                 'are required.')
            Y_i = util.extract_output(
                X_i,
                n_inputs=self.n_inputs_in_,
                episode_feature=False,
            )
            U_i = util.extract_input(
                X_i,
                n_inputs=self.n_inputs_in_,
                episode_feature=False,
            )
            n_ic = self.min_samples_ - 1
            n_pred = Y_i.shape[0] - n_ic
            Yp_i = np.zeros_like(Y_i)
            Yp_i[:n_ic, :] = Y_i[:n_ic, :]
            for (j, coef_Y_j) in enumerate(self.coef_Y_):
                Yp_i[n_ic:, :] -= Y_i[j:(j + n_pred), :] @ coef_Y_j.T
            for (j, coef_U_j) in enumerate(self.coef_U_):
                Yp_i[n_ic:, :] += U_i[j:(j + n_pred), :] @ coef_U_j.T
            predictions.append((i, Yp_i))
        # Combine and return
        Y_pred = util.combine_episodes(
            predictions,
            episode_feature=self.episode_feature_,
        )
        return Y_pred

    def predict_trajectory(self, X: np.ndarray) -> np.ndarray:
        """Predict multi-step trajectory given input for each episode."""
        # Check if fitted
        sklearn.utils.validation.check_is_fitted(self)
        # Validate array
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Split episodes
        episodes = util.split_episodes(
            X,
            episode_feature=self.episode_feature_,
        )
        # Predict for each episode
        predictions = []
        for (i, X_i) in episodes:
            if X_i.shape[0] < self.min_samples_:
                raise ValueError(f'Episode {i} has {X_i.shape[0]} samples but '
                                 f'`min_samples_`={self.min_samples_} samples '
                                 'are required.')
            Y_i = util.extract_output(
                X_i,
                n_inputs=self.n_inputs_in_,
                episode_feature=False,
            )
            U_i = util.extract_input(
                X_i,
                n_inputs=self.n_inputs_in_,
                episode_feature=False,
            )
            n_ic = self.min_samples_ - 1
            n_pred = Y_i.shape[0] - n_ic
            Yp_i = np.zeros_like(Y_i)
            Yp_i[:n_ic, :] = Y_i[:n_ic, :]
            for k in range(n_ic, Yp_i.shape[0]):
                for (j, coef_Y_j) in enumerate(self.coef_Y_):
                    lag = self.n_lags_output - j
                    Yp_i[[k], :] -= Yp_i[[k - lag], :] @ coef_Y_j.T
                for (j, coef_U_j) in enumerate(self.coef_U_):
                    lag = self.n_lags_input - j - 1
                    Yp_i[[k], :] += U_i[[k - lag], :] @ coef_U_j.T
            predictions.append((i, Yp_i))
        # Combine and return
        Y_pred = util.combine_episodes(
            predictions,
            episode_feature=self.episode_feature_,
        )
        return Y_pred

    # Extra estimator tags
    # https://scikit-learn.org/stable/developers/develop.html#estimator-tags
    def _more_tags(self):
        return {
            'multioutput': True,
            'multioutput_only': True,
        }
