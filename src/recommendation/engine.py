"""
Recommendation Engine — StyleSense (Updated Day 3)
Owner: Manas Kashyap
"""

SCENARIOS = {
    "real-time": {
        "recommended_method": "fast",
        "reason": "Fast NST produces results in <150ms — ideal for real-time filters.",
        "expected_runtime_ms": 120
    },
    "quality-first": {
        "recommended_method": "optimization",
        "reason": "Optimization NST gives highest quality — ideal for offline artwork.",
        "expected_runtime_ms": None
    },
    "batch": {
        "recommended_method": "fast",
        "reason": "Fast NST scales efficiently for batch processing.",
        "expected_runtime_ms": 130
    }
}


def recommend(scenario: str, hardware: str = "gpu",
              time_constraint_ms: int = None) -> dict:
    scenario = scenario.lower().strip()
    result = SCENARIOS.get(scenario, {
        "recommended_method": "fast",
        "reason": "Unknown scenario — defaulting to Fast NST.",
        "expected_runtime_ms": 130
    }).copy()

    if hardware == "cpu":
        result["reason"] += " Note: CPU will be slower; GPU strongly recommended."
    if time_constraint_ms and time_constraint_ms < 200:
        result["recommended_method"] = "fast"
        result["reason"] = (f"Tight time constraint ({time_constraint_ms}ms) "
                            f"— Fast NST recommended.")
    return result


if __name__ == "__main__":
    for tc in [
        {"scenario": "real-time",     "hardware": "gpu", "time_constraint_ms": 100},
        {"scenario": "quality-first", "hardware": "gpu"},
        {"scenario": "batch",         "hardware": "cpu"},
    ]:
        r = recommend(**tc)
        print(f"{tc['scenario']:15s} → {r['recommended_method']:12s} | {r['reason']}")
