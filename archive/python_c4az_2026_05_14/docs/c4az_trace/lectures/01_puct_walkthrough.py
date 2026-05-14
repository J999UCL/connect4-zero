from __future__ import annotations

from lectrace import note, table, text

from c4az.trace_demo import DemoEvaluator, build_demo_position, render_board_layers
from c4az.mcts import MCTSConfig, PUCTMCTS


def main() -> None:
    text("# PUCT Walkthrough")
    text(
        "A tiny deterministic run through the clean-room `c4az` MCTS stack. "
        "The right panel shows the live variables at each step."
    )

    position = build_demo_position()  # @inspect position.ply position.heights
    layers = render_board_layers(position)  # @inspect layers
    text("The board is canonical: `X` is always the side to move, `O` is the opponent.")
    text("\n\n".join(layers), verbatim=True)

    evaluator = DemoEvaluator()  # @inspect evaluator.calls evaluator.positions_evaluated
    config = MCTSConfig(
        simulations_per_move=12,
        c_puct=1.5,
        temperature=1.0,
        seed=11,
        trace=True,
    )  # @inspect config
    search = PUCTMCTS(evaluator, config)  # @inspect search.config

    note(
        "This evaluator is deterministic and tiny. It supplies policy logits and a value; "
        "there are no random rollouts in this path."
    )

    result = search.search(position, add_root_noise=False, temperature=1.0)  # @inspect result.root.visits result.root_value
    visits = result.visit_counts  # @inspect visits
    policy = result.policy  # @inspect policy
    q_values = result.q_values  # @inspect q_values

    top_rows = top_policy_rows(visits, policy, q_values)  # @inspect top_rows
    table(top_rows, headers=["action", "visits", "policy", "q"], caption="Top root actions after 12 PUCT simulations")

    event_counts = count_events(result.trace)  # @inspect event_counts
    trace_preview = result.trace[:12]  # @inspect trace_preview
    text("The internal trace records selection, evaluation/terminal handling, and backup events.")
    table(
        [[name, count] for name, count in sorted(event_counts.items())],
        headers=["event", "count"],
        caption="PUCT trace event counts",
    )

    text(
        "The self-play target is the root visit distribution. Training later optimizes "
        "`cross_entropy(policy, logits) + mse(final_game_value, value)`."
    )


def top_policy_rows(visits, policy, q_values):
    rows = []
    order = sorted(range(len(visits)), key=lambda action: (-int(visits[action]), action))
    for action in order[:6]:
        if int(visits[action]) == 0:
            continue
        rows.append(
            [
                int(action),
                int(visits[action]),
                round(float(policy[action]), 4),
                round(float(q_values[action]), 4),
            ]
        )
    return rows


def count_events(trace):
    counts = {}
    for event in trace:
        name = event["event"]
        counts[name] = counts.get(name, 0) + 1
    return counts


if __name__ == "__main__":
    main()
