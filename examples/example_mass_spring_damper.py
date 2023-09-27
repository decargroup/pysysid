"""Test Hankel."""

import numpy as np
from matplotlib import pyplot as plt

import pysysid


def main():
    """Test Hankel."""
    data = pysysid.example_data_msd()

    # Different ARX models
    # Strictly proper
    arx = pysysid.Arx(n_lags_input=1, n_lags_output=1)
    # Biproper
    # arx = pysysid.Arx(n_lags_input=3, n_lags_output=2)
    # Strictly proper
    # arx = pysysid.Arx(n_lags_input=2, n_lags_output=3)

    # Fit model
    arx.fit(
        data['X_train'],
        n_inputs=data['n_inputs'],
        episode_feature=data['episode_feature'],
    )

    # Run one-step-ahead prediction and running prediction
    Yp_singlestep = arx.predict(data['X_valid'])
    Yp_running = arx.predict_trajectory(data['X_valid'])

    # Take first episode of validation set for plotting
    Y_valid_i = pysysid.split_episodes(
        data['X_valid'],
        episode_feature=data['episode_feature'],
    )[0][1]
    Yp_singlestep_i = pysysid.split_episodes(
        Yp_singlestep,
        episode_feature=data['episode_feature'],
    )[0][1]
    Yp_running_i = pysysid.split_episodes(
        Yp_running,
        episode_feature=data['episode_feature'],
    )[0][1]

    # Plot predicted trajectories
    fig, ax = plt.subplots(3, 1, constrained_layout=True, sharex=True)
    ax[0].plot(Y_valid_i[:, 0], )
    ax[1].plot(Y_valid_i[:, 1], )
    ax[2].plot(Y_valid_i[:, 2])
    ax[0].plot(Yp_singlestep_i[:, 0], '--', label='Single step prediction')
    ax[1].plot(Yp_singlestep_i[:, 1], '--')
    ax[0].plot(Yp_running_i[:, 0], '--', label='Running prediction')
    ax[1].plot(Yp_running_i[:, 1], '--')
    ax[0].legend(loc='upper right')
    ax[0].set_ylabel(r'$\theta[k]$')
    ax[1].set_ylabel(r'$\dot{\theta}[k]$')
    ax[2].set_ylabel(r'$u[k]$')
    ax[2].set_xlabel(r'$k$')
    for a in np.ravel(ax):
        a.grid(linestyle='--')

    # Plot errors
    fig, ax = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    ax[0].plot(
        Y_valid_i[:, 0] - Yp_singlestep_i[:, 0],
        color='C1',
        label='Single step prediction',
    )
    ax[1].plot(Y_valid_i[:, 1] - Yp_singlestep_i[:, 1], color='C1')
    ax[0].plot(
        Y_valid_i[:, 0] - Yp_running_i[:, 0],
        color='C2',
        label='Running prediction',
    )
    ax[1].plot(Y_valid_i[:, 1] - Yp_running_i[:, 1], color='C2')
    ax[0].legend(loc='upper right')
    ax[0].set_ylabel(r'$\Delta\theta[k]$')
    ax[1].set_ylabel(r'$\Delta\dot{\theta}[k]$')
    ax[1].set_xlabel(r'$k$')
    for a in np.ravel(ax):
        a.grid(linestyle='--')

    plt.show()


if __name__ == '__main__':
    main()
