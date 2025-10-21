# -*- coding: utf-8 -*-
"""
Test Suite for NSN Integration
Validates all three stages of NSN integration
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import unittest
from quantum_integration.nsn_integration import (
    BackendAwareRankSelector,
    BackendType,
    MultilingualNSNEvaluator,
    NSNLeaderboard,
    NSNDashboard
)


class TestBackendAwareRankSelector(unittest.TestCase):
    """Test Stage 1: Backend-Aware Rank Selection"""
    
    def setUp(self):
        self.selector = BackendAwareRankSelector()
    
    def test_rank_selection_low_qubit(self):
        """Test rank selection for low-qubit backend"""
        rank_config = self.selector.select_rank(
            BackendType.IBM_MANILA,
            target_reliability=0.85
        )
        self.assertEqual(rank_config.rank, 8, "Low-qubit backend should select rank 8")
        self.assertLess(rank_config.flops, 1e7, "Low rank should have low FLOPs")
    
    def test_rank_selection_high_fidelity(self):
        """Test rank selection for high-fidelity backend"""
        rank_config = self.selector.select_rank(
            BackendType.IBM_WASHINGTON,
            target_reliability=0.90
        )
        self.assertGreaterEqual(rank_config.rank, 64, "High-fidelity backend should support high rank")
        self.assertGreater(rank_config.expected_reliability, 0.85)
    
    def test_flops_vs_reliability_curve(self):
        """Test FLOPs vs reliability curve generation"""
        curve = self.selector.compute_flops_vs_reliability(BackendType.IBM_WASHINGTON)
        self.assertGreater(len(curve), 0, "Curve should have points")
        
        # Verify curve is monotonically increasing in FLOPs
        flops_values = [point[0] for point in curve]
        self.assertEqual(flops_values, sorted(flops_values), "FLOPs should be increasing")
    
    def test_rank_recommendation(self):
        """Test rank recommendation with constraints"""
        recommendation = self.selector.get_rank_recommendation(
            backend_type=BackendType.RUSSIAN_SIMULATOR,
            compute_budget=1e8,
            min_reliability=0.90
        )
        
        self.assertIn('recommended_rank', recommendation)
        self.assertIn('expected_reliability', recommendation)
        self.assertIn('rationale', recommendation)
        self.assertLessEqual(recommendation['flops'], 1e8, "Should respect compute budget")


class TestMultilingualNSNEvaluator(unittest.TestCase):
    """Test Stage 2: Multilingual Edit Reliability"""
    
    def setUp(self):
        self.evaluator = MultilingualNSNEvaluator()
    
    def test_language_edit_evaluation(self):
        """Test single language edit evaluation"""
        result = self.evaluator.evaluate_language_edit('english', rank=64)
        
        self.assertEqual(result.language, 'english')
        self.assertEqual(result.rank, 64)
        self.assertGreater(result.edit_accuracy, 0)
        self.assertLess(result.edit_accuracy, 1)
        self.assertGreater(result.uncertainty, 0)
    
    def test_resource_level_accuracy(self):
        """Test that high-resource languages have higher accuracy"""
        high_resource = self.evaluator.evaluate_language_edit('english', rank=64)
        low_resource = self.evaluator.evaluate_language_edit('swahili', rank=64)
        
        self.assertGreater(high_resource.edit_accuracy, low_resource.edit_accuracy,
                          "High-resource language should have higher accuracy")
    
    def test_rank_scaling(self):
        """Test that higher rank improves accuracy"""
        low_rank = self.evaluator.evaluate_language_edit('indonesian', rank=8)
        high_rank = self.evaluator.evaluate_language_edit('indonesian', rank=128)
        
        self.assertGreater(high_rank.edit_accuracy, low_rank.edit_accuracy,
                          "Higher rank should improve accuracy")
        self.assertLess(high_rank.uncertainty, low_rank.uncertainty,
                       "Higher rank should reduce uncertainty")
    
    def test_subspace_containment(self):
        """Test subspace containment analysis"""
        containment = self.evaluator.evaluate_subspace_containment(
            source_lang='indonesian',
            target_lang='english',
            rank=64
        )
        
        self.assertEqual(containment.source_lang, 'indonesian')
        self.assertEqual(containment.target_lang, 'english')
        self.assertGreater(containment.containment_score, 0)
        self.assertLess(containment.containment_score, 1)
    
    def test_uncertainty_weights(self):
        """Test uncertainty weight computation"""
        languages = ['english', 'indonesian', 'swahili']
        weights = self.evaluator.compute_uncertainty_weights(languages)
        
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5,
                              msg="Weights should sum to 1")
        
        # Low-resource languages should have higher weights
        self.assertGreater(weights['swahili'], weights['english'])
    
    def test_rank_language_matrix(self):
        """Test comprehensive rank-language analysis"""
        languages = ['english', 'chinese', 'indonesian']
        analysis = self.evaluator.analyze_rank_language_matrix(languages)
        
        self.assertIn('accuracy_matrix', analysis)
        self.assertIn('containment_analysis', analysis)
        self.assertIn('uncertainty_weights', analysis)
        
        # Verify all languages are in matrix
        for lang in languages:
            self.assertIn(lang, analysis['accuracy_matrix'])


class TestNSNLeaderboard(unittest.TestCase):
    """Test Stage 3: Contributor Challenges"""
    
    def setUp(self):
        self.leaderboard = NSNLeaderboard()
    
    def test_challenge_creation(self):
        """Test challenge creation"""
        challenge = self.leaderboard.create_challenge(
            challenge_id="test_challenge",
            title="Test Challenge",
            description="Test description",
            languages=['english', 'chinese'],
            ranks=[8, 32, 64]
        )
        
        self.assertEqual(challenge.challenge_id, "test_challenge")
        self.assertEqual(len(challenge.languages), 2)
        self.assertEqual(len(challenge.ranks_to_evaluate), 3)
    
    def test_submission(self):
        """Test edit submission"""
        # Create challenge
        self.leaderboard.create_challenge(
            challenge_id="test_challenge",
            title="Test",
            description="Test",
            languages=['english'],
            ranks=[8, 32]
        )
        
        # Submit edit
        rank_results = {
            8: {'accuracy': 0.75, 'uncertainty': 0.20, 'flops': 6.4e5, 'efficiency': 0.012},
            32: {'accuracy': 0.88, 'uncertainty': 0.12, 'flops': 1.02e7, 'efficiency': 0.009}
        }
        
        submission = self.leaderboard.submit_edit(
            challenge_id="test_challenge",
            contributor_id="test_contributor",
            language="english",
            edit_description="Test edit",
            rank_results=rank_results
        )
        
        self.assertEqual(submission.contributor_id, "test_contributor")
        self.assertEqual(len(submission.ranks_evaluated), 2)
    
    def test_leaderboard_ranking(self):
        """Test leaderboard ranking computation"""
        # Create challenge
        self.leaderboard.create_challenge(
            challenge_id="test_challenge",
            title="Test",
            description="Test",
            languages=['english'],
            ranks=[32]
        )
        
        # Submit multiple edits
        for i in range(3):
            rank_results = {
                32: {
                    'accuracy': 0.80 + i * 0.05,
                    'uncertainty': 0.15 - i * 0.02,
                    'flops': 1e7,
                    'efficiency': 0.008 + i * 0.001
                }
            }
            
            self.leaderboard.submit_edit(
                challenge_id="test_challenge",
                contributor_id=f"contributor_{i}",
                language="english",
                edit_description=f"Edit {i}",
                rank_results=rank_results
            )
        
        # Get leaderboard
        rankings = self.leaderboard.get_leaderboard("test_challenge")
        
        self.assertEqual(len(rankings), 3)
        self.assertEqual(rankings[0]['position'], 1)
        
        # Verify descending order
        scores = [r['score'] for r in rankings]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_pareto_frontier(self):
        """Test Pareto frontier computation"""
        # Create challenge and submit edits
        self.leaderboard.create_challenge(
            challenge_id="test_challenge",
            title="Test",
            description="Test",
            languages=['english'],
            ranks=[8, 32, 64]
        )
        
        rank_results = {
            8: {'accuracy': 0.75, 'uncertainty': 0.20, 'flops': 6.4e5, 'efficiency': 0.012},
            32: {'accuracy': 0.88, 'uncertainty': 0.12, 'flops': 1.02e7, 'efficiency': 0.009},
            64: {'accuracy': 0.92, 'uncertainty': 0.08, 'flops': 4.1e7, 'efficiency': 0.007}
        }
        
        self.leaderboard.submit_edit(
            challenge_id="test_challenge",
            contributor_id="test_contributor",
            language="english",
            edit_description="Test",
            rank_results=rank_results
        )
        
        # Compute frontier
        frontier_data = self.leaderboard.compute_pareto_frontier("test_challenge")
        
        self.assertIn('frontier', frontier_data)
        self.assertIn('all_points', frontier_data)
        self.assertGreater(len(frontier_data['frontier']), 0)
    
    def test_feedback_generation(self):
        """Test feedback generation"""
        # Create challenge and submit
        self.leaderboard.create_challenge(
            challenge_id="test_challenge",
            title="Test",
            description="Test",
            languages=['english'],
            ranks=[32]
        )
        
        rank_results = {
            32: {'accuracy': 0.88, 'uncertainty': 0.12, 'flops': 1.02e7, 'efficiency': 0.009}
        }
        
        submission = self.leaderboard.submit_edit(
            challenge_id="test_challenge",
            contributor_id="test_contributor",
            language="english",
            edit_description="Test",
            rank_results=rank_results
        )
        
        # Generate feedback
        feedback = self.leaderboard.generate_feedback(submission.submission_id)
        
        self.assertIn('rank_specific_feedback', feedback)
        self.assertIn('recommendations', feedback)
        self.assertIn(32, feedback['rank_specific_feedback'])


class TestNSNDashboard(unittest.TestCase):
    """Test Dashboard Visualizations"""
    
    def setUp(self):
        self.dashboard = NSNDashboard()
    
    def test_dashboard_creation(self):
        """Test dashboard initialization"""
        self.assertIsNotNone(self.dashboard)
        self.assertEqual(self.dashboard.figsize, (15, 10))
    
    # Note: Visualization tests would require matplotlib backend setup
    # and are typically run separately or mocked


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBackendAwareRankSelector))
    suite.addTests(loader.loadTestsFromTestCase(TestMultilingualNSNEvaluator))
    suite.addTests(loader.loadTestsFromTestCase(TestNSNLeaderboard))
    suite.addTests(loader.loadTestsFromTestCase(TestNSNDashboard))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
    
    success = run_tests()
    sys.exit(0 if success else 1)
