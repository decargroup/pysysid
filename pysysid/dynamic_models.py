"""Example dynamic models."""

import abc
from typing import Callable, Tuple

import numpy as np
from scipy import constants, integrate


class ContinuousDynamicModel(metaclass=abc.ABCMeta):
    """Continuous-time dynamic model."""

    @abc.abstractmethod
    def f(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Implement differential equation.

        Parameters
        ----------
        t : float
            Time (s).
        x : np.ndarray
            State.
        u : np.ndarray
            Input.

        Returns
        -------
        np.ndarray
            Time derivative of state.
        """
        raise NotImplementedError()

    def g(self, t: float, x: np.ndarray) -> np.ndarray:
        """Implement output equation.

        Parameters
        ----------
        t : float
            Time (s).
        x : np.ndarray
            State.

        Returns
        -------
        np.ndarray
            Measurement of state.
        """
        return x

    def simulate(
        self,
        t_range: Tuple[float, float],
        t_step: float,
        x0: np.ndarray,
        u: Callable[[float], np.ndarray],
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate the model using numerical integration.

        Parameters
        ----------
        t_range : Tuple[float, float]
            Start and stop times in a tuple.
        t_step : float
            Timestep of output data.
        x0 : np.ndarray
            Initial condition, shape (n, ).
        u : Callable[[float], np.ndarray]
            Input function of time.
        **kwargs : dict
            Keyword arguments for :func:`integrate.solve_ivp`.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Time and state at every timestep. Each timestep is one row.
        """
        sol = integrate.solve_ivp(
            lambda t, x: self.f(t, x, u(t)),
            t_range,
            x0,
            t_eval=np.arange(*t_range, t_step),
            **kwargs,
        )
        return (sol.t, sol.y.T)


class DiscreteDynamicModel(metaclass=abc.ABCMeta):
    """Discrete-time dynamic model."""

    @abc.abstractmethod
    def f(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Implement next-state equation.

        Parameters
        ----------
        t : float
            Time (s).
        x : np.ndarray
            State.
        u : np.ndarray
            Input.

        Returns
        -------
        np.ndarray
            Next state.
        """
        raise NotImplementedError()

    def g(self, t, x):
        """Implement output equation.

        Parameters
        ----------
        t : float
            Time (s).
        x : np.ndarray
            State.

        Returns
        -------
        np.ndarray
            Measurement of state.
        """
        return x

    def simulate(
        self,
        t_range: Tuple[float, float],
        t_step: float,
        x0: np.ndarray,
        u: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate the model.

        Parameters
        ----------
        t_range : Tuple[float, float]
            Start and stop times in a tuple.
        t_step : float
            Timestep of output data.
        x0 : np.ndarray
            Initial condition, shape (n, ).
        u : np.ndarray
            Input array.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Time and state at every timestep. Each timestep is one row.
        """
        t = np.arange(*t_range, t_step)
        x = np.empty((t.shape[0], x0.shape[0]))
        x[0, :] = x0
        for k in range(1, t.shape[0]):
            x[k, :] = self.f(t[k - 1], x[k - 1, :], u[k - 1])
        return (t, x)


class MassSpringDamper(ContinuousDynamicModel):
    """Mass-spring-damper model.

    State is ``[position, velocity]``.

    Examples
    --------
    Simulate a mass-spring-damper

    >>> msd = pysysid.dynamic_models.MassSpringDamper(0.5, 0.7, 0.6)
    >>> x0 = np.array([1, 0])
    >>> t, x = msd.simulate((0, 1), 1e-3, x0, lambda t: 0)
    """

    def __init__(self, mass: float, stiffness: float, damping: float) -> None:
        """Instantiate :class:`MassSpringDamper`.

        Parameters
        ----------
        mass : float
            Mass (kg).
        stiffness : float
            Stiffness (N/m).
        damping : float
            Viscous damping (N.s/m).
        """
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping

    @property
    def A(self):
        """Compute ``A`` matrix."""
        A = np.array([
            [0, 1],
            [-self.stiffness / self.mass, -self.damping / self.mass],
        ])
        return A

    @property
    def B(self):
        """Compute ``B`` matrix."""
        B = np.array([
            [0],
            [1 / self.mass],
        ])
        return B

    def f(self, t: float, x: np.ndarray, u: np.ndarray):
        # noqa: D102
        x_dot = (self.A @ np.reshape(x, (-1, 1))
                 + self.B @ np.reshape(u, (-1, 1)))
        return np.ravel(x_dot)
