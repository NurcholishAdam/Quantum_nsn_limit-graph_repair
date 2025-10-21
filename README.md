# NSN Integration with LIMIT-Graph and REPAIR

Comprehensive integration of **Nested Subspace Networks (NSNs)** with LIMIT-Graph and REPAIR to enhance quantum benchmarking and multilingual edit reliability.

## Overview

This integration implements three key stages:

1. **Backend-Aware Rank Selection**: Dynamically adjust model rank based on quantum backend constraints
2. **Multilingual Edit Reliability**: Evaluate how rank affects correction accuracy across languages
3. **Contributor Challenges**: Design leaderboard tasks with rank-aware evaluation and compute-performance frontiers

## Architecture

```
nsn_integration/
├── __init__.py                          # Package initialization
├── backend_aware_rank_selector.py       # Stage 1: Backend-aware rank selection
├── multilingual_nsn_evaluator.py        # Stage 2: Multilingual evaluation
├── nsn_leaderboard.py                   # Stage 3: Contributor challenges
├── nsn_dashboard.py                     # Visualization dashboard
├── limit_graph_nsn_integration.py       # LIMIT-Graph integration
├── demo_complete_nsn_integration.py     # Complete demo
└── README.md                            # This file
```

## Stage 1: Backend-Aware Rank Selection

### Features

- **Dynamic Rank Adjustment**: Automatically select optimal NSN rank based on quantum backend characteristics
- **Backend Support**:
  - IBM Manila (5 qubits, noisy) → Low-rank inference (r=8)
  - IBM Washington (127 qubits, high-fidelity) → High-rank inference (r=128-256)
  - Russian Simulators (stable) → Maximum-rank inference (r=256)
- **FLOPs vs Reliability Visualization**: Plot compute-performance curves for each backend

### Usage

```python
from quantum_integration.nsn_integration import BackendAwareRankSelector, BackendType

# Create selector
selector = BackendAwareRankSelector()

# Get rank recommendation
recommendation = selector.get_rank_recommendation(
    backend_type=BackendType.IBM_WASHINGTON,
    compute_budget=1e8,
    min_reliability=0.85
)

print(f"Recommended Rank: {recommendation['recommended_rank']}")
print(f"Expected Reliability: {recommendation['expected_reliability']:.3f}")
print(f"Rationale: {recommendation['rationale']}")

# Compute FLOPs vs reliability curve
curve = selector.compute_flops_vs_reliability(BackendType.IBM_WASHINGTON)
```

## Stage 2: Multilingual Edit Reliability

### Features

- **Cross-Language Evaluation**: Assess edit accuracy across 15+ languages
- **Resource-Aware Training**: Uncertainty-weighted training for low/medium/high-resource languages
- **Subspace Containment Analysis**: Visualize how low-resource language edits nest within high-resource language subspaces
- **Optimal Rank Selection**: Find best rank per language given accuracy and compute constraints

### Language Support

- **High-Resource**: English, Chinese, Spanish, French, German
- **Medium-Resource**: Russian, Arabic, Japanese, Korean, Portuguese
- **Low-Resource**: Indonesian, Vietnamese, Thai, Swahili, Yoruba

### Usage

```python
from quantum_integration.nsn_integration import MultilingualNSNEvaluator

# Create evaluator
evaluator = MultilingualNSNEvaluator()

# Evaluate single language
result = evaluator.evaluate_language_edit(
    language='indonesian',
    rank=64
)

print(f"Accuracy: {result.edit_accuracy:.3f}")
print(f"Uncertainty: {result.uncertainty:.3f}")

# Comprehensive analysis
languages = ['english', 'chinese', 'indonesian', 'swahili']
analysis = evaluator.analyze_rank_language_matrix(languages)

# Get uncertainty weights for balanced training
weights = evaluator.compute_uncertainty_weights(languages)

# Analyze subspace containment
containment = evaluator.evaluate_subspace_containment(
    source_lang='indonesian',
    target_lang='english',
    rank=64
)

print(f"Containment Score: {containment.containment_score:.3f}")
```

## Stage 3: Contributor Challenges

### Features

- **Leaderboard System**: Track contributor submissions across multiple ranks
- **Pareto Frontier**: Visualize compute-performance trade-offs
- **Rank-Specific Feedback**: Provide detailed feedback on expressiveness, efficiency, and uncertainty
- **Challenge Management**: Create and manage multilingual editing challenges

### Usage

```python
from quantum_integration.nsn_integration import NSNLeaderboard

# Create leaderboard
leaderboard = NSNLeaderboard()

# Create challenge
challenge = leaderboard.create_challenge(
    challenge_id="multilingual_edit_2025",
    title="Multilingual Model Editing Challenge",
    description="Optimize edit accuracy across languages and ranks",
    languages=['english', 'chinese', 'indonesian'],
    ranks=[8, 16, 32, 64, 128, 256]
)

# Submit edit
rank_results = {
    8: {'accuracy': 0.75, 'uncertainty': 0.20, 'flops': 6.4e5, 'efficiency': 0.012},
    32: {'accuracy': 0.88, 'uncertainty': 0.12, 'flops': 1.02e7, 'efficiency': 0.009},
    128: {'accuracy': 0.95, 'uncertainty': 0.05, 'flops': 1.64e8, 'efficiency': 0.006}
}

submission = leaderboard.submit_edit(
    challenge_id="multilingual_edit_2025",
    contributor_id="contributor_001",
    language="english",
    edit_description="Optimized factual correction",
    rank_results=rank_results
)

# Get leaderboard
rankings = leaderboard.get_leaderboard("multilingual_edit_2025")

# Compute Pareto frontier
frontier = leaderboard.compute_pareto_frontier("multilingual_edit_2025")

# Generate feedback
feedback = leaderboard.generate_feedback(submission.submission_id)
```

## Dashboard Visualizations

### Available Plots

1. **FLOPs vs Reliability**: Backend performance curves
2. **Multilingual Heatmap**: Accuracy matrix across languages and ranks
3. **Subspace Containment**: Nested subspace analysis
4. **Pareto Frontier**: Compute-performance trade-offs
5. **Leaderboard Rankings**: Top contributor visualization
6. **Uncertainty Analysis**: Uncertainty reduction across ranks
7. **Comprehensive Dashboard**: Multi-panel overview

### Usage

```python
from quantum_integration.nsn_integration import NSNDashboard

# Create dashboard
dashboard = NSNDashboard()

# Plot FLOPs vs Reliability
dashboard.plot_flops_vs_reliability(
    backend_curves=backend_curves,
    save_path='flops_vs_reliability.png'
)

# Plot multilingual heatmap
dashboard.plot_multilingual_heatmap(
    accuracy_matrix=accuracy_matrix,
    save_path='multilingual_heatmap.png'
)

# Plot Pareto frontier
dashboard.plot_pareto_frontier(
    frontier_data=frontier_data,
    save_path='pareto_frontier.png'
)

# Create comprehensive dashboard
dashboard.create_comprehensive_dashboard(
    backend_curves=backend_curves,
    accuracy_matrix=accuracy_matrix,
    containment_data=containment_data,
    frontier_data=frontier_data,
    leaderboard=rankings,
    save_path='comprehensive_dashboard.png'
)
```

## LIMIT-Graph Integration

### Benchmarking Harness

The NSN integration is embedded into the LIMIT-Graph benchmarking harness for seamless evaluation:

```python
from quantum_integration.nsn_integration.limit_graph_nsn_integration import (
    LIMITGraphNSNBenchmark,
    BenchmarkConfig
)

# Create configuration
config = BenchmarkConfig(
    backend_type=BackendType.IBM_WASHINGTON,
    languages=['english', 'chinese', 'indonesian'],
    target_reliability=0.85,
    compute_budget=1e8
)

# Create benchmark
benchmark = LIMITGraphNSNBenchmark(config)

# Run benchmark
test_cases = [
    {'language': 'english', 'text': 'The capital of France is Paris'},
    {'language': 'chinese', 'text': '北京是中国的首都'},
    {'language': 'indonesian', 'text': 'Jakarta adalah ibu kota Indonesia'}
]

results = benchmark.run_benchmark(test_cases)

# Visualize results
benchmark.visualize_benchmark_results(results, save_path='benchmark_results.png')

# Compare backends
comparison = benchmark.compare_backends(test_cases)
```

## Running the Complete Demo

```bash
# Run complete NSN integration demo
python quantum_integration/nsn_integration/demo_complete_nsn_integration.py

# Run LIMIT-Graph integration demo
python quantum_integration/nsn_integration/limit_graph_nsn_integration.py
```

### Demo Output

The demo will:
1. Test backend-aware rank selection for IBM Manila, IBM Washington, and Russian Simulator
2. Evaluate multilingual edit reliability across 9 languages
3. Create contributor challenges and generate leaderboard
4. Generate comprehensive visualizations
5. Export results to JSON

### Generated Files

- `nsn_flops_vs_reliability.png`: Backend performance curves
- `nsn_multilingual_heatmap.png`: Language-rank accuracy matrix
- `nsn_subspace_containment.png`: Subspace nesting visualization
- `nsn_pareto_frontier.png`: Compute-performance frontier
- `nsn_leaderboard_rankings.png`: Top contributor rankings
- `nsn_uncertainty_analysis.png`: Uncertainty reduction analysis
- `nsn_comprehensive_dashboard.png`: Multi-panel dashboard
- `limit_graph_nsn_results.json`: Benchmark results

## Key Concepts

### Nested Subspace Networks (NSNs)

NSNs represent model parameters in nested subspaces of increasing rank:
- **Low Rank (r=8-16)**: Fast inference, lower accuracy, suitable for noisy backends
- **Medium Rank (r=32-64)**: Balanced performance
- **High Rank (r=128-256)**: Maximum accuracy, high compute, requires stable backends

### Backend-Aware Selection

Quantum backend characteristics determine optimal rank:
- **Qubit Count**: More qubits → higher rank capacity
- **Error Rate**: Lower error → higher rank feasibility
- **Gate Fidelity**: Higher fidelity → better high-rank performance
- **Coherence Time**: Longer coherence → supports complex circuits

### Multilingual Subspace Containment

Low-resource language edits often nest within high-resource language subspaces:
- **Indonesian → English**: ~85% containment at rank 128
- **Swahili → English**: ~80% containment at rank 128
- **Vietnamese → Chinese**: ~75% containment at rank 64

This enables transfer learning and cross-lingual edit propagation.

## Integration with Existing Components

### REPAIR Integration

```python
from quantum_integration.social_science_extensions import REPAIRInferenceWrapper
from quantum_integration.nsn_integration import BackendAwareRankSelector

# Select rank based on backend
selector = BackendAwareRankSelector()
rank_config = selector.select_rank(BackendType.IBM_WASHINGTON)

# Use rank in REPAIR inference
# (REPAIR wrapper can be extended to accept rank parameter)
```

### Quantum Health Monitoring

```python
from quantum_integration import quantum_health_checker
from quantum_integration.nsn_integration import BackendAwareRankSelector

# Check backend health
health = quantum_health_checker.check_backend_health('ibm_washington')

# Adjust rank based on health
if health['status'] == 'degraded':
    # Use lower rank for stability
    rank = 32
else:
    # Use optimal rank
    rank = selector.select_rank(BackendType.IBM_WASHINGTON).rank
```

## Performance Metrics

### Benchmark Results (Example)

| Backend | Rank | Accuracy | Uncertainty | FLOPs | Inference Time |
|---------|------|----------|-------------|-------|----------------|
| IBM Manila | 8 | 0.76 | 0.18 | 6.4e5 | 10ms |
| IBM Washington | 128 | 0.95 | 0.05 | 1.6e8 | 160ms |
| Russian Simulator | 256 | 0.97 | 0.03 | 6.6e8 | 320ms |

### Multilingual Performance

| Language | Resource Level | Rank 8 | Rank 32 | Rank 128 |
|----------|---------------|--------|---------|----------|
| English | High | 0.90 | 0.93 | 0.96 |
| Chinese | High | 0.89 | 0.92 | 0.95 |
| Russian | Medium | 0.78 | 0.85 | 0.91 |
| Indonesian | Low | 0.65 | 0.75 | 0.85 |
| Swahili | Low | 0.62 | 0.72 | 0.83 |

## Contributing

To contribute to NSN integration:

1. **Submit Edits**: Use the leaderboard system to submit your edits
2. **Evaluate Across Ranks**: Test your edits at multiple NSN ranks
3. **Optimize Efficiency**: Aim for the Pareto frontier (high accuracy, low FLOPs)
4. **Document Results**: Share your findings and techniques

## Citation

If you use this NSN integration in your research, please cite:

```bibtex
@software{nsn_limit_graph_integration,
  title={NSN Integration with LIMIT-Graph and REPAIR},
  author={AI Research Agent Team},
  year={2025},
  url={https://github.com/your-repo/quantum_integration/nsn_integration}
}
```

## License

This integration is part of the LIMIT-Graph project and follows the same license terms.

## Support

For questions or issues:
- Open an issue on GitHub
- Check the demo scripts for usage examples
- Review the comprehensive documentation in each module

## Roadmap

- [ ] Real-time rank adaptation based on backend telemetry
- [ ] Automated hyperparameter tuning for rank selection
- [ ] Extended language support (50+ languages)
- [ ] Integration with Hugging Face Spaces for public leaderboard
- [ ] Multi-backend ensemble inference
- [ ] Quantum circuit optimization for rank-specific operations
