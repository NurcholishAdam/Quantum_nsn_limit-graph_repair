# NSN Integration Quick Start Guide

Get started with NSN integration in 5 minutes!

## Installation

No additional dependencies required. The NSN integration uses existing quantum_integration packages.

## Quick Examples

### 1. Backend-Aware Rank Selection (30 seconds)

```python
from quantum_integration.nsn_integration import BackendAwareRankSelector, BackendType

selector = BackendAwareRankSelector()
recommendation = selector.get_rank_recommendation(
    backend_type=BackendType.IBM_WASHINGTON,
    compute_budget=1e8,
    min_reliability=0.85
)

print(f"Recommended Rank: {recommendation['recommended_rank']}")
print(f"Rationale: {recommendation['rationale']}")
```

### 2. Multilingual Evaluation (1 minute)

```python
from quantum_integration.nsn_integration import MultilingualNSNEvaluator

evaluator = MultilingualNSNEvaluator()
result = evaluator.evaluate_language_edit('indonesian', rank=64)

print(f"Accuracy: {result.edit_accuracy:.3f}")
print(f"Uncertainty: {result.uncertainty:.3f}")
```

### 3. Contributor Challenge (2 minutes)

```python
from quantum_integration.nsn_integration import NSNLeaderboard

leaderboard = NSNLeaderboard()
challenge = leaderboard.create_challenge(
    challenge_id="my_challenge",
    title="My First Challenge",
    description="Test multilingual editing",
    languages=['english', 'chinese']
)

# Submit edit
rank_results = {
    32: {'accuracy': 0.88, 'uncertainty': 0.12, 'flops': 1e7, 'efficiency': 0.009}
}

submission = leaderboard.submit_edit(
    challenge_id="my_challenge",
    contributor_id="me",
    language="english",
    edit_description="My edit",
    rank_results=rank_results
)

rankings = leaderboard.get_leaderboard("my_challenge")
print(f"Position: {rankings[0]['position']}")
```

## Run Complete Demo

```bash
python quantum_integration/nsn_integration/demo_complete_nsn_integration.py
```

## Run Tests

```bash
python quantum_integration/nsn_integration/test_nsn_integration.py
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore visualization with NSNDashboard
- Integrate with LIMIT-Graph benchmarking
- Submit to contributor challenges

## Support

Check the README.md or open an issue for help!
