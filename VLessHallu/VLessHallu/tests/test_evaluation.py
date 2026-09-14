from __future__ import annotations

from vlesshallu.evaluation import choose_sweep_winner, final_success


def test_sweep_selection_enforces_quality_constraints() -> None:
    baseline = {"F1": 50.0, "average_length": 100.0}
    candidates = [
        {
            "run_hash": "eligible",
            "metrics": {
                "F1": 49.0,
                "average_length": 90.0,
                "CHAIRs": 10.0,
                "CHAIRi": 20.0,
            },
        },
        {
            "run_hash": "short",
            "metrics": {
                "F1": 60.0,
                "average_length": 89.9,
                "CHAIRs": 1.0,
                "CHAIRi": 1.0,
            },
        },
    ]
    winner = choose_sweep_winner(baseline, candidates)
    assert winner["run_hash"] == "eligible"


def test_final_success_matches_protocol() -> None:
    outcome = final_success(
        {"CHAIRs": 10.0, "CHAIRi": 20.0, "F1": 70.0},
        {"CHAIRs": 9.0, "CHAIRi": 19.0, "F1": 69.0},
        {"CHAIR": 8.0, "Cover": 80.0, "Hal": 7.0, "Cog": 6.0},
        {"CHAIR": 7.0, "Cover": 78.0, "Hal": 6.0, "Cog": 6.5},
    )
    assert outcome["success"] is True
    assert outcome["conditions"]["two_hallucination_metrics_improve"] is True
