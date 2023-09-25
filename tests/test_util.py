"""Test :mod:`utils`."""

import numpy as np
import pytest

import pysysid


@pytest.mark.parametrize(
    'X, n_row, n_col, first_feature, episode_feature, H_exp',
    [
        (
            np.array([
               [1, 2],
               [2, 4],
               [3, 6],
               [4, 8],
               [5, 10],
            ]),
            2,
            3,
            0,
            False,
            np.array([
                [1, 2, 2, 4, 3, 6],
                [2, 4, 3, 6, 4, 8],
            ]),
        ),
        (
            np.array([
               [1, 2],
               [2, 4],
               [3, 6],
               [4, 8],
               [5, 10],
            ]),
            None,
            3,
            0,
            False,
            np.array([
                [1, 2, 2, 4, 3, 6],
                [2, 4, 3, 6, 4, 8],
                [3, 6, 4, 8, 5, 10],
            ]),
        ),
        (
            np.array([
               [1, 2],
               [2, 4],
               [3, 6],
               [4, 8],
               [5, 10],
            ]),
            3,
            3,
            0,
            False,
            np.array([
                [1, 2, 2, 4, 3, 6],
                [2, 4, 3, 6, 4, 8],
                [3, 6, 4, 8, 5, 10],
            ]),
        ),
        (
            np.array([
               [1, 2],
               [2, 4],
               [3, 6],
               [4, 8],
               [5, 10],
            ]),
            2,
            2,
            1,
            False,
            np.array([
                [2, 4, 3, 6],
                [3, 6, 4, 8],
            ]),
        ),
        (
            np.array([
               [1, 2],
               [2, 4],
               [3, 6],
               [4, 8],
               [5, 10],
            ]),
            None,
            2,
            1,
            False,
            np.array([
                [2, 4, 3, 6],
                [3, 6, 4, 8],
                [4, 8, 5, 10],
            ]),
        ),
        (
            np.array([
               [1, 2],
               [2, 4],
               [3, 6],
               [4, 8],
               [5, 10],
            ]),
            2,
            3,
            1,
            False,
            np.array([
                [2, 4, 3, 6, 4, 8],
                [3, 6, 4, 8, 5, 10],
            ]),
        ),
        (
            np.array([
               [1, 2],
               [2, 4],
               [3, 6],
               [4, 8],
               [5, 10],
            ]),
            3,
            4,
            0,
            False,
            'raise',
        ),
        (
            np.array([
               [1, 2],
               [2, 4],
               [3, 6],
               [4, 8],
               [5, 10],
            ]),
            4,
            3,
            0,
            False,
            'raise',
        ),
        (
            np.array([
               [1, 2],
               [2, 4],
               [3, 6],
               [4, 8],
               [5, 10],
            ]),
            3,
            3,
            1,
            False,
            'raise'
        ),
        (
            np.array([
               [0, 1, 2],
               [0, 2, 4],
               [0, 3, 6],
               [0, 4, 8],
               [0, 5, 10],
               [1, -1, -2],
               [1, -2, -4],
               [1, -3, -6],
               [1, -4, -8],
               [1, -5, -10],
            ]),
            2,
            3,
            0,
            True,
            np.array([
                [1, 2, 2, 4, 3, 6],
                [2, 4, 3, 6, 4, 8],
                [-1, -2, -2, -4, -3, -6],
                [-2, -4, -3, -6, -4, -8],
            ]),
        ),
        (
            np.array([
               [0, 1, 2],
               [0, 2, 4],
               [0, 3, 6],
               [0, 4, 8],
               [0, 5, 10],
               [1, -1, -2],
               [1, -2, -4],
               [1, -3, -6],
               [1, -4, -8],
               [1, -5, -10],
            ]),
            None,
            3,
            0,
            True,
            np.array([
                [1, 2, 2, 4, 3, 6],
                [2, 4, 3, 6, 4, 8],
                [3, 6, 4, 8, 5, 10],
                [-1, -2, -2, -4, -3, -6],
                [-2, -4, -3, -6, -4, -8],
                [-3, -6, -4, -8, -5, -10],
            ]),
        ),
        (
            np.array([
               [0, 1, 2],
               [0, 2, 4],
               [0, 3, 6],
               [0, 4, 8],
               [0, 5, 10],
               [1, -1, -2],
               [1, -2, -4],
               [1, -3, -6],
            ]),
            2,
            3,
            0,
            True,
            'raise'
        ),
    ],
)
class TestHankel:
    """Test :func:`pysysid.block_hankel`."""

    def test_block_hankel(
        self,
        X,
        n_row,
        n_col,
        first_feature,
        episode_feature,
        H_exp,
    ):
        """Test :func:`pysysid.block_hankel` against known answers."""
        if isinstance(H_exp, np.ndarray):
            H = pysysid.block_hankel(
                X,
                n_row,
                n_col,
                first_feature=first_feature,
                episode_feature=episode_feature,
            )
            np.testing.assert_allclose(H, H_exp)
        else:
            # If ``H_exp='raise'``, make sure it raises a :class:`ValueError`.
            with pytest.raises(ValueError):
                pysysid.block_hankel(
                    X,
                    n_row,
                    n_col,
                    first_feature=first_feature,
                    episode_feature=episode_feature,
                )


class TestEpisodeExtraction:
    """Test episode initial condition and input extraction.

    Specifically, tests :func:`extract_initial_conditions`,
    :func:`extract_output`, :func:`extract_input`, and
    :func:`strip_initial_conditions`.
    """

    @pytest.mark.parametrize(
        'X, ic_exp, min_samples, n_inputs, episode_feature',
        [
            (
                np.array([
                    [1, 2, 3, 4],
                    [4, 5, 6, 7],
                ]).T,
                np.array([
                    [1],
                    [4],
                ]).T,
                1,
                0,
                False,
            ),
            (
                np.array([
                    [1, 2, 3, 4],
                    [4, 5, 6, 7],
                ]).T,
                np.array([
                    [1, 2],
                    [4, 5],
                ]).T,
                2,
                0,
                False,
            ),
            (
                np.array([
                    [1, 2, 3, 4],
                    [4, 5, 6, 7],
                    [5, 5, 5, 5],
                ]).T,
                np.array([
                    [1],
                    [4],
                ]).T,
                1,
                1,
                False,
            ),
            (
                np.array([
                    [0, 0, 1, 1],
                    [1, 2, 3, 4],
                    [4, 5, 6, 7],
                ]).T,
                np.array([
                    [0, 1],
                    [1, 3],
                    [4, 6],
                ]).T,
                1,
                0,
                True,
            ),
            (
                np.array([
                    [0, 0, 1, 1],
                    [1, 2, 3, 4],
                    [4, 5, 6, 7],
                    [9, 9, 9, 9],
                ]).T,
                np.array([
                    [0, 1],
                    [1, 3],
                    [4, 6],
                ]).T,
                1,
                1,
                True,
            ),
            (
                np.array([
                    [0, 0, 0, 1, 1, 1],
                    [1, 2, 2, 3, 4, 5],
                    [4, 5, 5, 6, 7, 6],
                    [9, 9, 9, 9, 9, 6],
                ]).T,
                np.array([
                    [0, 0, 1, 1],
                    [1, 2, 3, 4],
                    [4, 5, 6, 7],
                ]).T,
                2,
                1,
                True,
            ),
        ],
    )
    def test_extract_initial_conditions(
        self,
        X,
        ic_exp,
        min_samples,
        n_inputs,
        episode_feature,
    ):
        """Test :func:`extract_initial_conditions`."""
        ic = pysysid.extract_initial_conditions(
            X,
            min_samples,
            n_inputs,
            episode_feature,
        )
        np.testing.assert_allclose(ic, ic_exp)

    @pytest.mark.parametrize(
        'X, y_exp, n_inputs, episode_feature',
        [
            (
                np.array([
                    [1, 2, 3, 4],
                    [6, 7, 8, 9],
                ]).T,
                np.array([
                    [1, 2, 3, 4],
                ]).T,
                1,
                False,
            ),
            (
                np.array([
                    [1, 2, 3, 4],
                    [6, 7, 8, 9],
                ]).T,
                np.array([]).reshape((0, 4)).T,
                2,
                False,
            ),
            (
                np.array([
                    [0, 0, 1, 1],
                    [1, 2, 3, 4],
                    [6, 7, 8, 9],
                ]).T,
                np.array([
                    [0, 0, 1, 1],
                    [1, 2, 3, 4],
                ]).T,
                1,
                True,
            ),
            (
                np.array([
                    [0, 0, 1, 1],
                    [1, 2, 3, 4],
                    [6, 7, 8, 9],
                ]).T,
                np.array([
                    [0, 0, 1, 1],
                ]).T,
                2,
                True,
            ),
        ],
    )
    def test_extract_output(self, X, y_exp, n_inputs, episode_feature):
        """Test :func:`extract_output`."""
        y = pysysid.extract_output(X, n_inputs, episode_feature)
        np.testing.assert_allclose(y, y_exp)

    @pytest.mark.parametrize(
        'X, u_exp, n_inputs, episode_feature',
        [
            (
                np.array([
                    [1, 2, 3, 4],
                    [6, 7, 8, 9],
                ]).T,
                np.array([
                    [6, 7, 8, 9],
                ]).T,
                1,
                False,
            ),
            (
                np.array([
                    [1, 2, 3, 4],
                    [6, 7, 8, 9],
                ]).T,
                np.array([]).reshape((0, 4)).T,
                0,
                False,
            ),
            (
                np.array([
                    [0, 0, 1, 1],
                    [1, 2, 3, 4],
                    [6, 7, 8, 9],
                ]).T,
                np.array([
                    [0, 0, 1, 1],
                    [6, 7, 8, 9],
                ]).T,
                1,
                True,
            ),
            (
                np.array([
                    [0, 0, 1, 1],
                    [1, 2, 3, 4],
                    [6, 7, 8, 9],
                ]).T,
                np.array([
                    [0, 0, 1, 1],
                ]).T,
                0,
                True,
            ),
        ],
    )
    def test_extract_input(self, X, u_exp, n_inputs, episode_feature):
        """Test :func:`extract_input`."""
        u = pysysid.extract_input(X, n_inputs, episode_feature)
        np.testing.assert_allclose(u, u_exp)

    def test_strip_initial_conditons(self):
        """Test :func:`strip_initial_conditions`."""
        X1 = np.array([
            [0, 0, 1, 1, 1, 2, 2, 2],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [2, 3, 4, 5, 6, 7, 8, 9],
        ]).T
        X2 = np.array([
            [0, 0, 1, 1, 1, 2, 2, 2],
            [-1, 2, -1, 4, 5, -1, 7, 8],
            [-1, 3, -1, 5, 6, -1, 8, 9],
        ]).T
        X1s = pysysid.strip_initial_conditions(
            X1,
            min_samples=1,
            episode_feature=True,
        )
        X2s = pysysid.strip_initial_conditions(
            X2,
            min_samples=1,
            episode_feature=True,
        )
        np.testing.assert_allclose(X1s, X2s)


@pytest.mark.parametrize(
    'X, episodes, episode_feature',
    [
        # Multiple episodes
        (
            np.array([
                [0, 0, 0, 1, 1, 1],
                [1, 2, 3, 4, 5, 6],
                [6, 5, 4, 3, 2, 1],
            ]).T,
            [
                (
                    0,
                    np.array([
                        [1, 2, 3],
                        [6, 5, 4],
                    ]).T,
                ),
                (
                    1,
                    np.array([
                        [4, 5, 6],
                        [3, 2, 1],
                    ]).T,
                ),
            ],
            True,
        ),
        # One episode
        (
            np.array([
                [0, 0, 0, 0, 0, 0],
                [1, 2, 3, 4, 5, 6],
                [6, 5, 4, 3, 2, 1],
            ]).T,
            [
                (
                    0,
                    np.array([
                        [1, 2, 3, 4, 5, 6],
                        [6, 5, 4, 3, 2, 1],
                    ]).T,
                ),
            ],
            True,
        ),
        # No episode feature
        (
            np.array([
                [1, 2, 3, 4, 5, 6],
                [6, 5, 4, 3, 2, 1],
            ]).T,
            [
                (
                    0,
                    np.array([
                        [1, 2, 3, 4, 5, 6],
                        [6, 5, 4, 3, 2, 1],
                    ]).T,
                ),
            ],
            False,
        ),
        # Out-of-order episodes
        (
            np.array([
                [2, 2, 2, 0, 0, 1],
                [1, 2, 3, 4, 5, 6],
                [6, 5, 4, 3, 2, 1],
            ]).T,
            [
                (
                    2,
                    np.array([
                        [1, 2, 3],
                        [6, 5, 4],
                    ]).T,
                ),
                (
                    0,
                    np.array([
                        [4, 5],
                        [3, 2],
                    ]).T,
                ),
                (
                    1,
                    np.array([
                        [6],
                        [1],
                    ]).T,
                ),
            ],
            True,
        ),
        # Negative episode
        (
            np.array([
                [0, 0, 0, -1, -1, -1],
                [1, 2, 3, 4, 5, 6],
                [6, 5, 4, 3, 2, 1],
            ]).T,
            'raise',
            True,
        ),
        # Fractional episode
        (
            np.array([
                [0, 0, 0, 1.1, 1.1, 1.1],
                [1, 2, 3, 4, 5, 6],
                [6, 5, 4, 3, 2, 1],
            ]).T,
            'raise',
            True,
        ),
    ],
)
class TestSplitCombineEpisodes:
    """Test :func:`split_episodes` and :func:`combine_episodes`."""

    def test_split_episodes(self, X, episodes, episode_feature):
        """Test :func:`split_episodes`."""
        if episodes == 'raise':
            with pytest.raises(ValueError):
                pysysid.split_episodes(X, episode_feature=episode_feature)
        else:
            episodes_actual = pysysid.split_episodes(
                X, episode_feature=episode_feature)
            episodes_expected = list(sorted(episodes))
            zipped_episodes = zip(episodes_actual, episodes_expected)
            if episode_feature:
                X_ep = X[:, 0]
                X = X[:, 1:]
                for (i_act, X_i_act), (i_exp, X_i_exp) in zipped_episodes:
                    assert i_act == i_exp
                    np.testing.assert_allclose(X_i_act, X_i_exp)
            else:
                assert len(episodes) == 1
                X_exp = episodes[0][1]
                np.testing.assert_allclose(X, X_exp)

    def test_combine_episodes(self, X, episodes, episode_feature):
        """Test :func:`combine_episodes`."""
        if episodes == 'raise':
            with pytest.raises(ValueError):
                pysysid.combine_episodes(
                    episodes,
                    episode_feature=episode_feature,
                )
        else:
            X_actual = pysysid.combine_episodes(
                episodes,
                episode_feature=episode_feature,
            )
            np.testing.assert_allclose(X_actual, X)


@pytest.mark.parametrize('fn', [
    pysysid.example_data_msd,
])
class TestExampleData:
    """Test example dynamic model data.

    Attributes
    ----------
    tol : float
        Tolerance for regression test.
    """

    tol = 1e-6

    def test_example_data(self, ndarrays_regression, fn):
        """Test example dynamic model data."""
        data = fn()
        ndarrays_regression.check(
            {
                'X_train': data['X_train'],
                'X_valid': data['X_valid'],
                'x0_valid': data['x0_valid'],
                'u_valid': data['u_valid'],
                'n_inputs': data['n_inputs'],
                'episode_feature': data['episode_feature'],
                't': data['t'],
            },
            default_tolerance=dict(atol=self.tol, rtol=0),
        )
