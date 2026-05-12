import matplotlib.pyplot as plt
import numpy as np
import torch

from models.base import FinancialModel
from plotting.utils import plot_kde
from policies.learnable import RNNPolicy
from utils.consts import MertonConsts, SystemConsts


def _clip_action(action: float, action_bound: float | None) -> float:
    if action_bound is None:
        return action
    return float(np.clip(action, -action_bound, action_bound))


def _crra_utility(x: np.ndarray, gamma: float) -> np.ndarray:
    x_safe = np.maximum(x, 1e-12)
    if np.isclose(gamma, 1.0):
        return np.log(x_safe)
    return (x_safe ** (1 - gamma)) / (1 - gamma)


def _compute_xlim(arrays: list[np.ndarray]) -> tuple[float, float] | None:
    combined = np.concatenate([np.asarray(arr).ravel() for arr in arrays])
    combined = combined[np.isfinite(combined)]
    if combined.size < 2:
        return None

    lo, hi = np.percentile(combined, [1, 85])
    if np.isclose(lo, hi):
        return (-0.5, float(hi) + 1.0)

    pad = 0.05 * (hi - lo)
    return (-0.5, float(hi) + pad)


@torch.no_grad()
def compare_rnn_to_plugin_policy(
    rnn_policy: RNNPolicy,
    financial_model: FinancialModel,
    m: int = 2000,
    state_type: str = "pomdp",
    device: torch.device | None = None,
    plugin_min_obs: int = 5,
    action_bound: float | None = None,
    savepath: str | None = None,
):
    """
    Compares the performance of a trained RNN policy against a naive plug-in policy
    that re-estimates mu and sigma at each step.

    Args:
        rnn_policy (RNNPolicy): The trained RNN policy.
        financial_model (FinancialModel): The financial model for simulation.
        m (int): Number of trajectories to simulate for evaluation.
        state_type (str): The type of state to use for the RNN policy.
        device (torch.device | None): The device to run the simulation on.
        plugin_min_obs (int): Minimum observations before plug-in estimation starts.
        action_bound (float | None): Symmetric action clip for all policies. If None,
            uses the RNN action scale (when available).
        savepath (str | None): Path to save the comparison plot. If None, plot is not saved.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rnn_policy.to(device)
    rnn_policy.eval()

    if not isinstance(financial_model.params, MertonConsts):
        raise TypeError("This comparison is only implemented for Merton models.")
    const_params = financial_model.params

    if action_bound is None:
        action_bound = getattr(rnn_policy, "action_scale", None)

    terminal_wealths_rnn = []
    terminal_wealths_plugin = []
    terminal_wealths_expert = []

    base_seed = SystemConsts.terminal_wealth_base_seed_policy

    for i in range(m):
        seed = base_seed + i
        rng = np.random.default_rng(seed=seed)

        # --- Setup trajectory-specific parameters ---
        if financial_model.traj_dep:
            mu_t = const_params.mu + const_params.distr_var * rng.standard_normal()
            sigma_t = rng.lognormal(
                mean=np.log(const_params.sigma), sigma=const_params.distr_var
            )
        else:
            mu_t, sigma_t = const_params.mu, const_params.sigma

        # --- Initialize wealth and history for all policies ---
        X_rnn = torch.tensor(const_params.init_wealth, dtype=torch.float32)
        X_plugin = torch.tensor(const_params.init_wealth, dtype=torch.float32)
        X_expert = torch.tensor(const_params.init_wealth, dtype=torch.float32)
        N = int(const_params.T / const_params.delta_t)
        returns_history = []
        R_prev = 0.0
        hidden = rnn_policy.get_initial_hidden(batch_size=1, device=device)

        for t in range(N):
            # --- 1) Decide actions using information up to t-1 only ---
            if state_type == "pomdp":
                state = torch.tensor([[R_prev]], dtype=torch.float32).to(device)
            else:
                # Add other state representations if needed
                raise NotImplementedError(f"State type '{state_type}' not supported.")

            if rnn_policy.probabilistic:
                pi_rnn, _, hidden = rnn_policy(state, hidden)
            else:
                pi_rnn, hidden = rnn_policy(state, hidden)

            pi_rnn = _clip_action(float(pi_rnn.cpu().item()), action_bound)

            # --- 2. Naive Plug-in Policy ---
            min_obs = max(plugin_min_obs, 2)
            if len(returns_history) < min_obs:
                pi_plugin = 0.0
            else:
                np_returns = np.array(returns_history)
                mu_hat = np.mean(np_returns) / const_params.delta_t
                sigma_hat = np.std(np_returns, ddof=1) / np.sqrt(const_params.delta_t)
                if sigma_hat < 0.01:  # Avoid division by zero
                    pi_plugin = 0.0
                else:
                    pi_plugin = (mu_hat - const_params.r) / (
                        const_params.gamma * sigma_hat**2
                    )
            pi_plugin = _clip_action(float(pi_plugin), action_bound)

            pi_expert = (mu_t - const_params.r) / (const_params.gamma * sigma_t**2)

            # --- 4) Realize market return at t and update all policies ---
            epsilon = rng.standard_normal()
            R = (
                mu_t * const_params.delta_t
                + sigma_t * const_params.delta_t**0.5 * epsilon
            )
            market_return_increment = R - const_params.r * const_params.delta_t

            X_rnn *= (
                1
                + const_params.r * const_params.delta_t
                + pi_rnn * market_return_increment
            )
            X_plugin *= (
                1
                + const_params.r * const_params.delta_t
                + pi_plugin * market_return_increment
            )
            X_expert *= (
                1
                + const_params.r * const_params.delta_t
                + pi_expert * market_return_increment
            )

            returns_history.append(R)
            R_prev = R

        terminal_wealths_rnn.append(X_rnn.item())
        terminal_wealths_plugin.append(X_plugin.item())
        terminal_wealths_expert.append(X_expert.item())

    rnn_arr = np.array(terminal_wealths_rnn)
    plugin_arr = np.array(terminal_wealths_plugin)
    expert_arr = np.array(terminal_wealths_expert)
    xlim = _compute_xlim([rnn_arr, plugin_arr, expert_arr])

    # --- Plotting and Statistics ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plot_kde(
        ax,
        data=rnn_arr,
        label=f"RNN (Median: {np.median(rnn_arr):.2f})",
        color="blue",
        xlim=xlim,
    )
    plot_kde(
        ax,
        data=plugin_arr,
        label=f"Naive Plug-in (Median: {np.median(plugin_arr):.2f})",
        color="red",
        xlim=xlim,
    )
    plot_kde(
        ax,
        data=expert_arr,
        label=f"Expert (Median: {np.median(expert_arr):.2f})",
        color="black",
        xlim=xlim,
    )

    ax.set_title("Distribution of Terminal Wealth (m={})".format(m))
    ax.set_xlabel("Terminal Wealth")
    ax.set_ylabel("Density")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    if savepath is not None:
        plt.savefig(savepath, dpi=100, bbox_inches="tight")

    plt.show()

    # --- Compute CRRA utility ---
    utility_rnn = _crra_utility(rnn_arr, const_params.gamma)
    utility_plugin = _crra_utility(plugin_arr, const_params.gamma)
    utility_expert = _crra_utility(expert_arr, const_params.gamma)

    print("\n--- Terminal Wealth Statistics ---")
    print(
        f"RNN Policy:           Mean={np.mean(rnn_arr):.3f}, "
        f"Median={np.median(rnn_arr):.3f}, "
        f"Std={np.std(rnn_arr):.3f}, "
        f"P5/P95={np.percentile(rnn_arr, 5):.3f}/{np.percentile(rnn_arr, 95):.3f}"
    )
    print(
        f"Naive Plug-in Policy: Mean={np.mean(plugin_arr):.3f}, "
        f"Median={np.median(plugin_arr):.3f}, "
        f"Std={np.std(plugin_arr):.3f}, "
        f"P5/P95={np.percentile(plugin_arr, 5):.3f}/{np.percentile(plugin_arr, 95):.3f}"
    )
    print(
        f"Expert Policy:        Mean={np.mean(expert_arr):.3f}, "
        f"Median={np.median(expert_arr):.3f}, "
        f"Std={np.std(expert_arr):.3f}, "
        f"P5/P95={np.percentile(expert_arr, 5):.3f}/{np.percentile(expert_arr, 95):.3f}"
    )
    print("\n--- Mean CRRA Utility ---")
    print(f"RNN:        {np.mean(utility_rnn):.6f}")
    print(f"Naive:      {np.mean(utility_plugin):.6f}")
    print(f"Expert:     {np.mean(utility_expert):.6f}")
