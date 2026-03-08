"""
Recommendation Engine — StyleSense
Rules-based: suggests NST method based on user constraints.
Owner: Manas Kashyap
"""


SCENARIOS = {
    "real-time": {
        "recommended_method": "fast",
        "reason": "Fast NST produces results in <150ms, ideal for real-time filters.",
        "expected_runtime_ms": 120
    },
    "quality-first": {
        "recommended_method": "optimization",
        "reason": "Optimization-based NST gives highest quality, suitable for offline artwork.",
        "expected_runtime_ms": None  # configurable iterations
    },
    "batch": {
        "recommended_method": "fast",
        "reason": "Fast NST scales well for batch processing with consistent speed.",
        "expected_runtime_ms": 130
    }
}


def recommend(scenario: str, hardware: str = "gpu", time_constraint_ms: int = None) -> dict:
    """
    Returns recommended NST method based on user scenario.

    Args:
        scenario: "real-time" | "quality-first" | "batch"
        hardware: "gpu" | "cpu"
        time_constraint_ms: optional time limit in milliseconds

    Returns:
        dict with recommended_method, reason, expected_runtime_ms
    """
    scenario = scenario.lower().strip()

    if scenario not in SCENARIOS:
        return {
            "recommended_method": "fast",
            "reason": "Unknown scenario — defaulting to Fast NST as safe choice.",
            "expected_runtime_ms": 130
        }

    result = SCENARIOS[scenario].copy()

    # Override: if CPU and real-time constraint, warn user
    if hardware == "cpu" and scenario == "real-time":
        result["reason"] += " Note: CPU may exceed 150ms; GPU recommended."

    # Override: if time_constraint is very tight, force fast
    if time_constraint_ms is not None and time_constraint_ms < 200:
        result["recommended_method"] = "fast"
        result["reason"] = f"Time constraint {time_constraint_ms}ms is strict — Fast NST recommended."

    return result


if __name__ == "__main__":
    test_cases = [
        {"scenario": "real-time", "hardware": "gpu", "time_constraint_ms": 100},
        {"scenario": "quality-first", "hardware": "gpu"},
        {"scenario": "batch", "hardware": "cpu"},
    ]
    for tc in test_cases:
        result = recommend(**tc)
        print(f"Scenario: {tc['scenario']:15s} → Method: {result['recommended_method']:12s} | {result['reason']}")
