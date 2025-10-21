# -*- coding: utf-8 -*-
"""
Complete NSN Integration Demo
Demonstrates all three stages of NSN integration with LIMIT-Graph and REPAIR
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from quantum_integration.nsn_integration import (
    BackendAwareRankSelector,
    BackendType,
    MultilingualNSNEvaluator,
    NSNLeaderboard,
    NSNDashboard
)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_stage_1_backend_aware_rank_selection():
    """
    Stage 1: Backend-Aware Rank Selection
    Dynamically adjust model rank based on quantum backend constraints
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: Backend-Aware Rank Selection")
    logger.info("=" * 80)
    
    selector = BackendAwareRankSelector()
    
    # Test different backends
    backends = [
        BackendType.IBM_MANILA,
        BackendType.IBM_WASHINGTON,
        BackendType.RUSSIAN_SIMULATOR
    ]
    
    backend_curves = {}
    
    for backend in backends:
        logger.info(f"\n--- Testing {backend.value} ---")
        
        # Get rank recommendation
        recommendation = selector.get_rank_recommendation(
            backend_type=backend,
            compute_budget=1e8,
            min_reliability=0.85
        )
        
        logger.info(f"Recommended Rank: {recommendation['recommended_rank']}")
        logger.info(f"Expected Reliability: {recommendation['expected_reliability']:.3f}")
        logger.info(f"FLOPs: {recommendation['flops']:.2e}")
        logger.info(f"Rationale: {recommendation['rationale']}")
        
        # Compute FLOPs vs reliability curve
        curve = selector.compute_flops_vs_reliability(backend)
        backend_curves[backend.value] = curve
        
        logger.info(f"Performance curve: {len(curve)} points")
    
    return backend_curves


def demo_stage_2_multilingual_edit_reliability():
    """
    Stage 2: Multilingual Edit Reliability via NSNs
    Evaluate how rank affects correction accuracy across languages
    """
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2: Multilingual Edit Reliability")
    logger.info("=" * 80)
    
    evaluator = MultilingualNSNEvaluator()
    
    # Test languages
    test_languages = [
        'english', 'chinese', 'spanish',  # High-resource
        'russian', 'arabic', 'japanese',  # Medium-resource
        'indonesian', 'vietnamese', 'swahili'  # Low-resource
    ]
    
    logger.info(f"\nEvaluating {len(test_languages)} languages across ranks...")
    
    # Comprehensive analysis
    analysis = evaluator.analyze_rank_language_matrix(test_languages)
    
    logger.info("\n--- Accuracy Matrix Summary ---")
    for lang in test_languages[:3]:  # Show first 3
        logger.info(f"{lang.capitalize()}:")
        for rank in [8, 32, 128]:
            acc = analysis['accuracy_matrix'][lang][rank]['accuracy']
            unc = analysis['accuracy_matrix'][lang][rank]['uncertainty']
            logger.info(f"  Rank {rank}: accuracy={acc:.3f}, uncertainty={unc:.3f}")
    
    logger.info("\n--- Subspace Containment Analysis ---")
    for cont in analysis['containment_analysis'][:3]:  # Show first 3
        logger.info(f"{cont['source']} -> {cont['target']} (rank {cont['rank']}): "
                   f"containment={cont['containment']:.3f}, overlap={cont['overlap']:.3f}")
    
    logger.info("\n--- Uncertainty Weights for Balanced Training ---")
    for lang, weight in list(analysis['uncertainty_weights'].items())[:5]:
        logger.info(f"{lang.capitalize()}: {weight:.3f}")
    
    # Optimal rank per language
    optimal_ranks = evaluator.get_optimal_rank_per_language(
        target_accuracy=0.85,
        max_flops=1e8
    )
    
    logger.info("\n--- Optimal Ranks per Language ---")
    for lang in test_languages:
        logger.info(f"{lang.capitalize()}: Rank {optimal_ranks[lang]}")
    
    return analysis, evaluator


def demo_stage_3_contributor_challenges():
    """
    Stage 3: Contributor Challenges with Rank-Aware Evaluation
    Design leaderboard tasks with compute-performance frontier
    """
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 3: Contributor Challenges & Leaderboard")
    logger.info("=" * 80)
    
    leaderboard = NSNLeaderboard()
    
    # Create a challenge
    challenge = leaderboard.create_challenge(
        challenge_id="multilingual_edit_2025",
        title="Multilingual Model Editing Challenge",
        description="Optimize edit accuracy across languages and ranks",
        languages=['english', 'chinese', 'indonesian', 'swahili'],
        ranks=[8, 16, 32, 64, 128, 256]
    )
    
    logger.info(f"\nCreated Challenge: {challenge.title}")
    logger.info(f"Languages: {', '.join(challenge.languages)}")
    logger.info(f"Ranks to evaluate: {challenge.ranks_to_evaluate}")
    
    # Simulate contributor submissions
    contributors = [
        ('contributor_001', 'english'),
        ('contributor_002', 'chinese'),
        ('contributor_003', 'indonesian'),
        ('contributor_004', 'swahili'),
        ('contributor_005', 'english')
    ]
    
    logger.info(f"\n--- Simulating {len(contributors)} Submissions ---")
    
    for contributor_id, language in contributors:
        # Simulate results across ranks
        rank_results = {}
        for rank in [8, 32, 64, 128]:
            # Simulate metrics (in real scenario, these come from actual evaluation)
            base_acc = 0.70 + (rank / 256) * 0.25
            accuracy = base_acc + (hash(contributor_id) % 10) / 100
            uncertainty = 0.20 - (rank / 256) * 0.15
            flops = (rank ** 2) * 1e4
            
            rank_results[rank] = {
                'accuracy': accuracy,
                'uncertainty': uncertainty,
                'flops': flops,
                'efficiency': accuracy / (flops / 1e6)
            }
        
        submission = leaderboard.submit_edit(
            challenge_id=challenge.challenge_id,
            contributor_id=contributor_id,
            language=language,
            edit_description=f"Optimized edit for {language}",
            rank_results=rank_results
        )
        
        logger.info(f"Submitted: {contributor_id} ({language}) - "
                   f"Best rank: {submission.get_best_rank()[0]}")
    
    # Get leaderboard
    rankings = leaderboard.get_leaderboard(challenge.challenge_id)
    
    logger.info("\n--- Leaderboard Rankings ---")
    for entry in rankings[:5]:
        logger.info(f"#{entry['position']}: {entry['contributor_id']} - "
                   f"Score: {entry['score']:.3f}, "
                   f"Best: Rank {entry['best_rank']} ({entry['best_accuracy']:.2%})")
    
    # Compute Pareto frontier
    frontier_data = leaderboard.compute_pareto_frontier(challenge.challenge_id)
    logger.info(f"\n--- Pareto Frontier ---")
    logger.info(f"Frontier points: {len(frontier_data['frontier'])}")
    for flops, acc in frontier_data['frontier'][:3]:
        logger.info(f"  FLOPs: {flops:.2e}, Accuracy: {acc:.3f}")
    
    # Generate feedback for first submission
    if rankings:
        feedback = leaderboard.generate_feedback(rankings[0]['submission_id'])
        logger.info(f"\n--- Feedback for Top Contributor ---")
        logger.info(f"Contributor: {feedback['contributor_id']}")
        logger.info("Recommendations:")
        for rec in feedback['recommendations']:
            logger.info(f"  - {rec}")
    
    return leaderboard, frontier_data, rankings


def demo_visualization_dashboard(backend_curves, multilingual_analysis, 
                                 evaluator, frontier_data, rankings):
    """
    Demonstrate NSN Dashboard visualizations
    """
    logger.info("\n" + "=" * 80)
    logger.info("NSN DASHBOARD VISUALIZATIONS")
    logger.info("=" * 80)
    
    dashboard = NSNDashboard()
    
    # 1. FLOPs vs Reliability
    logger.info("\nGenerating FLOPs vs Reliability plot...")
    dashboard.plot_flops_vs_reliability(
        backend_curves=backend_curves,
        save_path='nsn_flops_vs_reliability.png'
    )
    
    # 2. Multilingual Heatmap
    logger.info("Generating Multilingual Accuracy Heatmap...")
    accuracy_matrix = {}
    for lang, rank_data in multilingual_analysis['accuracy_matrix'].items():
        accuracy_matrix[lang] = {
            rank: data['accuracy'] for rank, data in rank_data.items()
        }
    
    dashboard.plot_multilingual_heatmap(
        accuracy_matrix=accuracy_matrix,
        save_path='nsn_multilingual_heatmap.png'
    )
    
    # 3. Subspace Containment
    logger.info("Generating Subspace Containment visualization...")
    dashboard.plot_subspace_containment(
        containment_data=multilingual_analysis['containment_analysis'],
        save_path='nsn_subspace_containment.png'
    )
    
    # 4. Pareto Frontier
    logger.info("Generating Pareto Frontier plot...")
    dashboard.plot_pareto_frontier(
        frontier_data=frontier_data,
        save_path='nsn_pareto_frontier.png'
    )
    
    # 5. Leaderboard Rankings
    logger.info("Generating Leaderboard Rankings...")
    dashboard.plot_leaderboard_rankings(
        leaderboard=rankings,
        top_n=5,
        save_path='nsn_leaderboard_rankings.png'
    )
    
    # 6. Uncertainty Analysis
    logger.info("Generating Uncertainty Analysis...")
    language_results = {}
    for lang in ['english', 'indonesian', 'swahili']:
        results = evaluator.evaluate_across_ranks(lang)
        language_results[lang] = [
            {
                'rank': r.rank,
                'accuracy': r.edit_accuracy,
                'uncertainty': r.uncertainty
            }
            for r in results
        ]
    
    dashboard.plot_uncertainty_analysis(
        language_results=language_results,
        save_path='nsn_uncertainty_analysis.png'
    )
    
    # 7. Comprehensive Dashboard
    logger.info("Generating Comprehensive Dashboard...")
    dashboard.create_comprehensive_dashboard(
        backend_curves=backend_curves,
        accuracy_matrix=accuracy_matrix,
        containment_data=multilingual_analysis['containment_analysis'],
        frontier_data=frontier_data,
        leaderboard=rankings,
        save_path='nsn_comprehensive_dashboard.png'
    )
    
    logger.info("\nAll visualizations generated successfully!")


def main():
    """
    Run complete NSN integration demo
    """
    logger.info("=" * 80)
    logger.info("NSN INTEGRATION WITH LIMIT-GRAPH AND REPAIR")
    logger.info("Complete Demo: All Three Stages")
    logger.info("=" * 80)
    
    try:
        # Stage 1: Backend-Aware Rank Selection
        backend_curves = demo_stage_1_backend_aware_rank_selection()
        
        # Stage 2: Multilingual Edit Reliability
        multilingual_analysis, evaluator = demo_stage_2_multilingual_edit_reliability()
        
        # Stage 3: Contributor Challenges
        leaderboard, frontier_data, rankings = demo_stage_3_contributor_challenges()
        
        # Visualization Dashboard
        demo_visualization_dashboard(
            backend_curves, multilingual_analysis, evaluator,
            frontier_data, rankings
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("DEMO COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("\nKey Achievements:")
        logger.info("✓ Stage 1: Backend-aware rank selection implemented")
        logger.info("✓ Stage 2: Multilingual edit reliability evaluated")
        logger.info("✓ Stage 3: Contributor challenges and leaderboard created")
        logger.info("✓ Comprehensive dashboard visualizations generated")
        logger.info("\nAll NSN integration components are operational!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
