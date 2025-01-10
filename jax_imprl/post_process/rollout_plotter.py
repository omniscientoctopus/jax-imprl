import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class RolloutPlotter:
    def __init__(self, env, agent):

        self.env = env
        self.agent = agent
        self.oracle = None
        k = env.k
        n = env.n_components
        self.env_name = f"{k}-out-of-{n}"
        self.time_horizon = env.time_horizon
        self.num_components = env.n_components
        self.num_damage_states = env.n_damage_states
        self.num_component_actions = env.n_comp_actions

    def get_sample_rollout(self, key):
        # data to be collected
        data = {
            "time": np.arange(0, self.time_horizon + 1),
            "true_states": np.empty(
                (self.time_horizon + 1, self.num_components), dtype=int
            ),
            "beliefs": np.empty(
                (self.time_horizon + 1, self.num_components, self.num_damage_states)
            ),
            "actions": np.ones((self.time_horizon, self.num_components), dtype=int)
            * -1,
            "system_risk": np.empty(self.time_horizon + 1),
            "rewards": np.empty(self.time_horizon),
            "cost_mobilisation": np.empty(self.time_horizon),
            "cost_penalties": np.empty(self.time_horizon),
            "cost_inspections": np.empty(self.time_horizon),
            "cost_replacements": np.empty(self.time_horizon),
            "failure_timepoints": np.zeros(self.time_horizon + 1),
            "episode_cost": 0,
        }

        terminated, truncated = False, False
        episode_reward = 0

        _, state = self.env.reset(key)

        time = state.timestep
        system_belief = state.belief
        data["beliefs"][time, :, :] = system_belief

        while not truncated and not terminated:

            # damage states
            data["true_states"][time, :] = state.damage_state

            # systems failure risk
            pf = system_belief[-1, :]
            data["system_risk"][time] = self.env.pf_sys(pf, self.env.k)

            # select action
            action = jnp.array([0, 1, 2, 0])

            # step the environment
            key, step_key = jax.random.split(key)
            _, state, reward, terminated, truncated, info = self.env.step(
                step_key, state, action
            )

            data["actions"][time, :] = action
            system_belief = state.belief
            data["beliefs"][time + 1, :, :] = system_belief

            _discount = self.env.discount_factor**time
            data["rewards"][time] = reward * _discount
            data["cost_mobilisation"][time] = info["reward_mobilisation"] * _discount
            data["cost_penalties"][time] = info["reward_penalties"] * _discount
            data["cost_inspections"][time] = info["reward_inspections"] * _discount
            data["cost_replacements"][time] = info["reward_replacements"] * _discount

            # note system failure timepoints
            if info["reward_penalties"] < 0:
                data["failure_timepoints"][time] = 1

            # update episode reward
            episode_reward += reward * _discount

            # update time
            time = state.timestep

        np.testing.assert_allclose(info["returns"], episode_reward, rtol=1e-3)

        data["episode_cost"] = -episode_reward
        data["true_states"][time, :] = state.damage_state
        data["system_risk"][time] = self.env.pf_sys(system_belief[-1, :], self.env.k)

        return data

    def _setup_plot(self):

        mosaic = """
            111333
            222444
            ..BB..
            """
        fig = plt.figure(figsize=(15, 6.5))
        ax_dict = fig.subplot_mosaic(mosaic)

        self.time_horizon_ticks = np.arange(0, self.time_horizon + 1, 2)
        self.action_markersize = 8

        plt.rcParams.update(
            {
                "axes.titlesize": "medium",
                "axes.labelsize": "large",
            }
        )

        self.action_idxs = np.arange(self.num_component_actions)
        self.action_labels = ["do-nothing", "repair", "inspect"]
        self.action_colors = ["gray", "darkviolet", "orange"]
        self.action_markers = [".", "s", ">"]

        for c in range(self.num_components):
            ax = ax_dict[f"{c+1}"]

            ## Plot agent actions
            ax2 = ax.twinx()
            ax2.set_yticks(self.action_idxs)

            ax.set_ylim([-0.05, 1.05])
            ax.set_xlim([-0.5, self.time_horizon + 0.5])
            ax.set_yticks([0, 0.5, 1])
            ax.set_xlabel("time", fontsize=15)
            ax.set_title(f"Component {c+1}", weight="bold", fontsize=16, pad=15)
            ax.grid()

            ax2.tick_params(right=False, labelright=False)
            ax2.spines[["top", "right", "left"]].set_visible(False)

        # create legend handles
        legend_handles = []
        for a in self.action_idxs:
            legend_handles += [
                Line2D(
                    [],
                    [],
                    marker=self.action_markers[a],
                    markersize=self.action_markersize,
                    label=self.action_labels[a],
                    color=self.action_colors[a],
                    linestyle="None",
                )
            ]

        labels = ["mobilisation", "repair", "inspect", "failure"]
        colors = ["lightpink", "darkviolet", "orange", "lightcoral"]

        barplot = ax_dict["B"].barh(labels, [0] * len(labels), color=colors, height=0.4)
        ax_dict["B"].set_xlim([0, 100])
        ax_dict["B"].set_xticks([0, 25, 50, 75, 100])
        ax_dict["B"].set_xticklabels(["0%", "25%", "50%", "75%", "100%"])

        return fig, ax_dict, legend_handles, barplot

    def _plot_agent_and_oracle_actions(self, c, ax):
        _y_action = 1
        for a in self.action_idxs:
            # Agent actions
            _x = np.where(self.data["actions"][:, c] == a)
            ax.plot(
                _x,
                _y_action,
                self.action_markers[a],
                markersize=self.action_markersize,
                label=self.action_labels[a],
                color=self.action_colors[a],
            )

            # Oracle actions
            if self.oracle is not None:
                # shade oracle actions region
                ax.fill_between(
                    np.arange(-1, self.time_horizon + 2),
                    self.num_damage_states - 0.48,
                    self.num_damage_states + 0.48,
                    color="green",
                    alpha=0.05,
                )
                _x = np.where(self.data["oracle_actions"][:, c] == a)
                ax.plot(
                    _x,
                    self.num_damage_states,
                    self.action_markers[a],
                    markersize=self.action_markersize,
                    label=self.action_labels[a],
                    color=self.action_colors[a],
                )
        ax.set_ylabel("damage-state", fontsize=8, color="tab:blue", loc="bottom")
        if self.oracle is not None:
            ax.spines.top.set_position(("data", self.num_damage_states - 0.48))
        ax.grid(False)

    def _plot_failure_lines(self, ax):
        if self.data["failure_timepoints"].sum() > 0:
            ax.vlines(
                np.where(self.data["failure_timepoints"]),
                -1,
                self.num_damage_states - 0.48,
                label="system-failure",
                color="red",
                alpha=0.5,
            )

    def _plot_beliefs_and_true_state(self, c, ax):
        colorbar = ax.pcolormesh(
            self.data["time"],
            np.arange(self.num_damage_states),
            self.data["beliefs"][:, c, :].T,
            shading="nearest",
            cmap="binary",
            alpha=0.2,
            vmin=0,
            vmax=1,
            edgecolors="face",
        )

        # true state
        (h_true_state,) = ax.plot(
            self.data["time"],
            self.data["true_states"][:, c],
            "-",
            label="damage state",
            color="tab:blue",
            markersize=2,
            alpha=0.8,
        )

        ax.set_yticks(np.arange(self.num_damage_states))
        ax.set_ylim([-0.5, self.num_damage_states - 0.48])
        ax.set_xticks(self.time_horizon_ticks)
        ax.set_xticklabels(self.time_horizon_ticks, fontsize=14)
        return colorbar, h_true_state

    def _plot_system_risk(self, ax2):
        scaled_risk = self.data["system_risk"] * 0.5
        (h_system_risk,) = ax2.plot(
            self.data["time"],
            scaled_risk,
            "-",
            label="system risk",
            color="tab:orange",
            markersize=2,
            alpha=0.8,
        )

        ax2.set_yticks([0, 0.25, 0.5])
        ax2.set_yticklabels([0, 0.5, 1], color="tab:orange")  # Rescale to [0, 1]
        ax2.set_ylim([-0.05, 0.75])
        ax2.set_ylabel("system risk", fontsize=10, color="tab:orange", loc="bottom")
        return h_system_risk

    def _make_barplot(self, ax, barplot):
        total_costs = np.array(
            [
                -self.data["cost_mobilisation"].sum() + 1e-15,  # numerical stability
                -self.data["cost_replacements"].sum(),
                -self.data["cost_inspections"].sum() + 1e-15,  # numerical stability
                -self.data["cost_penalties"].sum(),
            ]
        )
        _percentages = total_costs * 100 / total_costs.sum()

        assert np.isclose(
            total_costs.sum(), self.data["episode_cost"]
        ), "Total costs do not sum up to episode cost"

        # update bar plot
        for b, bar in enumerate(barplot):
            bar.set_width(_percentages[b])

        # add text to bar plot
        for b, bar in enumerate(barplot):
            ax.text(
                108,
                bar.get_y() + bar.get_height() / 2,
                f"{total_costs[b]:.1f}",
                va="center",
                ha="center",
                fontsize=12,
                color="black",
            )

        ax.set_title(f"Episode cost: {self.data['episode_cost']:.3f}", fontsize=14)

    def plot(self, key=None, data=None, save_fig_kwargs=None):

        self.data = self.get_sample_rollout(key) if data is None else data

        # get base plot from Plotter
        fig, ax_dict, legend_handles, barplot = self._setup_plot()

        for c in range(self.num_components):
            ax = ax_dict[f"{c+1}"]

            self._plot_failure_lines(ax)
            self._plot_agent_and_oracle_actions(c, ax)
            colorbar, h_true_state = self._plot_beliefs_and_true_state(c, ax)

            ax2 = ax.twinx()
            h_system_risk = self._plot_system_risk(ax2)

        self._make_barplot(ax_dict["B"], barplot)

        # title
        fig.suptitle(f"{self.env_name} system", fontsize=18, weight="bold")
        config_text = (
            "(parallel configuration)"
            if self.env_name == "1-out-of-4"
            else "(series configuration)" if self.env_name == "4-out-of-4" else ""
        )
        if config_text:
            fig.text(0.5, 0.92, config_text, fontsize=14, ha="center", va="center")

        # update legend_handles
        legend_handles += [h_true_state, h_system_risk]
        if self.data["failure_timepoints"].sum() > 0:
            legend_handles += [
                Line2D([], [], color="red", label="system-failure", alpha=1)
            ]
        fig.legend(handles=legend_handles, loc=(0.83, 0.05), fontsize=12)

        fig.tight_layout()

        # colorbar for belief next to ax_dict["B"]
        cbar_ax = fig.add_axes([0.72, 0.05, 0.02, 0.2])
        fig.colorbar(colorbar, cax=cbar_ax, label="belief")

        # behavior policy: agent name
        fig.text(
            0.05, 0.2, "Behavior policy: constant action", fontsize=14, weight="bold"
        )
        if self.oracle:
            fig.text(
                0.05,
                0.15,
                "Oracle policy: SARSOP",
                fontsize=14,
                color="green",
                weight="bold",
            )
            fig.text(
                0.05,
                0.12,
                "(oracle actions on top with green background)",
                fontsize=10,
                color="green",
            )

        plt.show()

        if save_fig_kwargs is not None:
            fig.savefig(**save_fig_kwargs)
