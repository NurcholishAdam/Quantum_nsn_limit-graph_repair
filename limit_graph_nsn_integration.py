# -*- coding: utf-8 -*-
"""
LIMIT-Graph NSN Integration
Embeds NSN rank-selection logic into LIMIT-Graph benchmarking harness
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from quantum_integration.nsn_integration import (
    BackendAwareRankSelector,
    BackendType,
    MultilingualNSNEvaluator
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for LIMIT-Graph benchmark with NSN"""
    backend_type: BackendType
    languages: List[str]
    target_reliability: float = 0.85
    compute_budget: float = 1e8
    enable_rank_adaptation: bool = True
    enable_multilingual_weighting: bool = True


class LIMITGraphNSNBenchmark:
    """
    LIMIT-Graph benchmarking harness with NSN integration
    """
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark harness
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.rank_selector = BackendAwareRankSelector()
        self.multilingual_evaluator = MultilingualNSNEvaluator()
        
        # Select optimal rank for backend
        self.selected_rank = self.rank_selector.select_rank(
            backend_type=config.backend_type,
            target_reliability=config.target_reliability
        )
        
        logger.info(f"Initialized LIMIT-Graph NSN Benchmark")
        logger.info(f"Backend: {config.backend_type.value}")
        logger.info(f"Selected Rank: {self.selected_rank.rank}")
        logger.info(f"Expected Reliability: {self.selected_rank.expected_reliability:.3f}")
    
    def run_benchmark(self, test_cases: List[Dict[str, Any]]) -> Dict:
        """
        Run benchmark with NSN-aware evaluation
        
        Args:
            test_cases: List of test case dictionaries
            
        Returns:
            Benchmark results
        """
        logger.info(f"Running benchmark with {len(test_cases)} test cases...")
        
        results = {
            'config': {
                'backend': self.config.backend_type.value,
                'rank': self.selected_rank.rank,
                'languages': self.config.languages
            },
            'test_results': [],
            'language_performance': {},
            'overall_metrics': {}
        }
        
        # Run test cases
        for i, test_case in enumerate(test_cases):
            language = test_case.get('language', 'english')
            
            # Evaluate with NSN
            eval_result = self.multilingual_evaluator.evaluate_language_edit(
                language=language,
                rank=self.selected_rank.rank,
                edit_text=test_case.get('text', '')
            )
            
            test_result = {
                'test_id': i,
                'language': language,
                'rank': self.selected_rank.rank,
                'accuracy': eval_result.edit_accuracy,
                'uncertainty': eval_result.uncertainty,
                'flops': eval_result.flops,
                'resource_level': eval_result.resource_level
            }
            
            results['test_results'].append(test_result)
            
            # Aggregate by language
            if language not in results['language_performance']:
                results['language_performance'][language] = {
                    'count': 0,
                    'total_accuracy': 0.0,
                    'total_uncertainty': 0.0
                }
            
            results['language_performance'][language]['count'] += 1
            results['language_performance'][language]['total_accuracy'] += eval_result.edit_accuracy
            results['language_performance'][language]['total_uncertainty'] += eval_result.uncertainty
        
        # Compute overall metrics
        if results['test_results']:
            results['overall_metrics'] = {
                'mean_accuracy': sum(r['accuracy'] for r in results['test_results']) / len(results['test_results']),
                'mean_uncertainty': sum(r['uncertainty'] for r in results['test_results']) / len(results['test_results']),
                'total_flops': sum(r['flops'] for r in results['test_results']),
                'num_tests': len(results['test_results'])
            }
        
        # Compute language averages
        for lang, perf in results['language_performance'].items():
            perf['avg_accuracy'] = perf['total_accuracy'] / perf['count']
            perf['avg_uncertainty'] = perf['total_uncertainty'] / perf['count']
        
        logger.info(f"Benchmark completed: {len(results['test_results'])} tests")
        logger.info(f"Overall accuracy: {results['overall_metrics']['mean_accuracy']:.3f}")
        
        return results
    
    def visualize_benchmark_results(self, results: Dict, save_path: Optional[str] = None):
        """
        Visualize benchmark results with NSN dashboard
        
        Args:
            results: Benchmark results from run_benchmark
            save_path: Optional path to save visualization
        """
        from quantum_integration.nsn_integration import NSNDashboard
        import matplotlib.pyplot as plt
        
        dashboard = NSNDashboard()
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Accuracy by language
        ax1 = axes[0, 0]
        languages = list(results['language_performance'].keys())
        accuracies = [results['language_performance'][lang]['avg_accuracy'] for lang in languages]
        ax1.bar(languages, accuracies, color='skyblue', edgecolor='black')
        ax1.set_ylabel('Average Accuracy', fontweight='bold')
        ax1.set_title('Accuracy by Language', fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Uncertainty by language
        ax2 = axes[0, 1]
        uncertainties = [results['language_performance'][lang]['avg_uncertainty'] for lang in languages]
        ax2.bar(languages, uncertainties, color='salmon', edgecolor='black')
        ax2.set_ylabel('Average Uncertainty', fontweight='bold')
        ax2.set_title('Uncertainty by Language', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 3: Accuracy vs Uncertainty scatter
        ax3 = axes[1, 0]
        for test in results['test_results']:
            ax3.scatter(test['uncertainty'], test['accuracy'], 
                       alpha=0.6, s=100, edgecolors='black')
        ax3.set_xlabel('Uncertainty', fontweight='bold')
        ax3.set_ylabel('Accuracy', fontweight='bold')
        ax3.set_title('Accuracy-Uncertainty Trade-off', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary metrics
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary_text = f"""
        BENCHMARK SUMMARY
        
        Backend: {results['config']['backend']}
        Rank: {results['config']['rank']}
        
        Overall Metrics:
        • Mean Accuracy: {results['overall_metrics']['mean_accuracy']:.3f}
        • Mean Uncertainty: {results['overall_metrics']['mean_uncertainty']:.3f}
        • Total FLOPs: {results['overall_metrics']['total_flops']:.2e}
        • Num Tests: {results['overall_metrics']['num_tests']}
        
        Languages Tested: {len(languages)}
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.suptitle('LIMIT-Graph NSN Benchmark Results', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved benchmark visualization to {save_path}")
        
        plt.show()
        return fig
    
    def export_results(self, results: Dict, filepath: str):
        """Export benchmark results to JSON"""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Exported results to {filepath}")
    
    def compare_backends(self, test_cases: List[Dict[str, Any]]) -> Dict:
        """
        Compare performance across different quantum backends
        
        Args:
            test_cases: List of test cases
            
        Returns:
            Comparison results
        """
        backends = [
            BackendType.IBM_MANILA,
            BackendType.IBM_WASHINGTON,
            BackendType.RUSSIAN_SIMULATOR
        ]
        
        comparison = {
            'backends': {},
            'test_cases': test_cases
        }
        
        for backend in backends:
            logger.info(f"\nBenchmarking {backend.value}...")
            
            # Create config for this backend
            config = BenchmarkConfig(
                backend_type=backend,
                languages=self.config.languages,
                target_reliability=self.config.target_reliability,
                compute_budget=self.config.compute_budget
            )
            
            # Create benchmark instance
            benchmark = LIMITGraphNSNBenchmark(config)
            
            # Run benchmark
            results = benchmark.run_benchmark(test_cases)
            
            comparison['backends'][backend.value] = {
                'selected_rank': benchmark.selected_rank.rank,
                'expected_reliability': benchmark.selected_rank.expected_reliability,
                'overall_metrics': results['overall_metrics'],
                'language_performance': results['language_performance']
            }
        
        logger.info("\nBackend comparison completed")
        return comparison


def create_limit_graph_nsn_benchmark(config: BenchmarkConfig) -> LIMITGraphNSNBenchmark:
    """Factory function to create LIMIT-Graph NSN benchmark"""
    return LIMITGraphNSNBenchmark(config)


def demo_limit_graph_integration():
    """Demo LIMIT-Graph NSN integration"""
    logger.info("=" * 80)
    logger.info("LIMIT-GRAPH NSN INTEGRATION DEMO")
    logger.info("=" * 80)
    
    # Create configuration
    config = BenchmarkConfig(
        backend_type=BackendType.IBM_WASHINGTON,
        languages=['english', 'chinese', 'indonesian', 'swahili'],
        target_reliability=0.85,
        compute_budget=1e8
    )
    
    # Create benchmark
    benchmark = create_limit_graph_nsn_benchmark(config)
    
    # Create test cases
    test_cases = [
        {'language': 'english', 'text': 'The capital of France is Paris'},
        {'language': 'english', 'text': 'Python is a programming language'},
        {'language': 'chinese', 'text': '北京是中国的首都'},
        {'language': 'chinese', 'text': '机器学习是人工智能的一部分'},
        {'language': 'indonesian', 'text': 'Jakarta adalah ibu kota Indonesia'},
        {'language': 'swahili', 'text': 'Nairobi ni mji mkuu wa Kenya'}
    ]
    
    # Run benchmark
    results = benchmark.run_benchmark(test_cases)
    
    # Visualize results
    benchmark.visualize_benchmark_results(
        results,
        save_path='limit_graph_nsn_benchmark_results.png'
    )
    
    # Export results
    benchmark.export_results(results, 'limit_graph_nsn_results.json')
    
    # Compare backends
    logger.info("\n" + "=" * 80)
    logger.info("BACKEND COMPARISON")
    logger.info("=" * 80)
    
    comparison = benchmark.compare_backends(test_cases[:3])  # Use subset for demo
    
    logger.info("\n--- Backend Comparison Summary ---")
    for backend_name, backend_data in comparison['backends'].items():
        logger.info(f"\n{backend_name}:")
        logger.info(f"  Selected Rank: {backend_data['selected_rank']}")
        logger.info(f"  Expected Reliability: {backend_data['expected_reliability']:.3f}")
        logger.info(f"  Mean Accuracy: {backend_data['overall_metrics']['mean_accuracy']:.3f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("INTEGRATION DEMO COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    demo_limit_graph_integration()
