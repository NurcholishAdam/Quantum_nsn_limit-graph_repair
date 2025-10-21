# -*- coding: utf-8 -*-
"""
NSN Leaderboard and Contributor Challenges
Rank-aware evaluation with compute-performance frontier visualization
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ContributorSubmission:
    """A contributor's edit submission"""
    contributor_id: str
    submission_id: str
    timestamp: datetime
    language: str
    edit_description: str
    ranks_evaluated: List[int]
    results: Dict[int, Dict[str, float]]  # rank -> metrics
    
    def get_best_rank(self) -> Tuple[int, float]:
        """Get rank with best accuracy"""
        best_rank = max(self.results.keys(), 
                       key=lambda r: self.results[r].get('accuracy', 0))
        best_acc = self.results[best_rank]['accuracy']
        return best_rank, best_acc
    
    def get_pareto_frontier_point(self) -> List[Tuple[float, float]]:
        """Get (FLOPs, accuracy) points for Pareto frontier"""
        points = []
        for rank, metrics in self.results.items():
            points.append((metrics['flops'], metrics['accuracy']))
        return points


@dataclass
class ContributorChallenge:
    """A leaderboard challenge for contributors"""
    challenge_id: str
    title: str
    description: str
    languages: List[str]
    ranks_to_evaluate: List[int]
    evaluation_criteria: Dict[str, float]  # metric -> weight
    start_date: datetime
    end_date: datetime
    submissions: List[ContributorSubmission] = field(default_factory=list)
    
    def add_submission(self, submission: ContributorSubmission):
        """Add a contributor submission"""
        self.submissions.append(submission)
        logger.info(f"Added submission {submission.submission_id} to challenge {self.challenge_id}")
    
    def compute_leaderboard(self) -> List[Dict]:
        """Compute leaderboard rankings"""
        rankings = []
        
        for submission in self.submissions:
            # Compute weighted score
            score = 0.0
            for rank, metrics in submission.results.items():
                for criterion, weight in self.evaluation_criteria.items():
                    score += metrics.get(criterion, 0) * weight
            
            score /= len(submission.results)  # Average across ranks
            
            rankings.append({
                'contributor_id': submission.contributor_id,
                'submission_id': submission.submission_id,
                'score': score,
                'best_rank': submission.get_best_rank()[0],
                'best_accuracy': submission.get_best_rank()[1],
                'language': submission.language,
                'timestamp': submission.timestamp.isoformat()
            })
        
        # Sort by score descending
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        # Add rank position
        for i, entry in enumerate(rankings):
            entry['position'] = i + 1
        
        return rankings


class NSNLeaderboard:
    """
    Manages NSN-based contributor challenges and leaderboards
    """
    
    def __init__(self):
        self.challenges: Dict[str, ContributorChallenge] = {}
        self.global_submissions: List[ContributorSubmission] = []
    
    def create_challenge(self, 
                        challenge_id: str,
                        title: str,
                        description: str,
                        languages: List[str],
                        ranks: List[int] = None) -> ContributorChallenge:
        """
        Create a new contributor challenge
        
        Args:
            challenge_id: Unique challenge identifier
            title: Challenge title
            description: Challenge description
            languages: Languages to evaluate
            ranks: NSN ranks to evaluate
            
        Returns:
            Created challenge
        """
        if ranks is None:
            ranks = [8, 16, 32, 64, 128, 256]
        
        challenge = ContributorChallenge(
            challenge_id=challenge_id,
            title=title,
            description=description,
            languages=languages,
            ranks_to_evaluate=ranks,
            evaluation_criteria={
                'accuracy': 0.5,
                'efficiency': 0.3,  # FLOPs efficiency
                'uncertainty': 0.2  # Lower is better
            },
            start_date=datetime.now(),
            end_date=datetime.now()  # Set appropriately
        )
        
        self.challenges[challenge_id] = challenge
        logger.info(f"Created challenge: {challenge_id}")
        
        return challenge
    
    def submit_edit(self,
                   challenge_id: str,
                   contributor_id: str,
                   language: str,
                   edit_description: str,
                   rank_results: Dict[int, Dict[str, float]]) -> ContributorSubmission:
        """
        Submit an edit for evaluation
        
        Args:
            challenge_id: Challenge to submit to
            contributor_id: Contributor identifier
            language: Edit language
            edit_description: Description of the edit
            rank_results: Results for each rank evaluated
            
        Returns:
            Created submission
        """
        if challenge_id not in self.challenges:
            raise ValueError(f"Challenge {challenge_id} not found")
        
        challenge = self.challenges[challenge_id]
        
        submission = ContributorSubmission(
            contributor_id=contributor_id,
            submission_id=f"{contributor_id}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            language=language,
            edit_description=edit_description,
            ranks_evaluated=list(rank_results.keys()),
            results=rank_results
        )
        
        challenge.add_submission(submission)
        self.global_submissions.append(submission)
        
        logger.info(f"Submitted edit from {contributor_id} for challenge {challenge_id}")
        
        return submission
    
    def get_leaderboard(self, challenge_id: str) -> List[Dict]:
        """
        Get leaderboard for a challenge
        
        Args:
            challenge_id: Challenge identifier
            
        Returns:
            Leaderboard rankings
        """
        if challenge_id not in self.challenges:
            raise ValueError(f"Challenge {challenge_id} not found")
        
        return self.challenges[challenge_id].compute_leaderboard()
    
    def compute_pareto_frontier(self, challenge_id: str) -> Dict:
        """
        Compute compute-performance Pareto frontier
        
        Args:
            challenge_id: Challenge identifier
            
        Returns:
            Pareto frontier data
        """
        if challenge_id not in self.challenges:
            raise ValueError(f"Challenge {challenge_id} not found")
        
        challenge = self.challenges[challenge_id]
        
        # Collect all points
        all_points = []
        contributor_points = {}
        
        for submission in challenge.submissions:
            points = submission.get_pareto_frontier_point()
            all_points.extend(points)
            contributor_points[submission.contributor_id] = points
        
        # Compute Pareto frontier
        pareto_frontier = self._compute_pareto_optimal(all_points)
        
        return {
            'frontier': pareto_frontier,
            'all_points': all_points,
            'contributor_points': contributor_points,
            'challenge_id': challenge_id
        }
    
    def _compute_pareto_optimal(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Compute Pareto optimal frontier (minimize FLOPs, maximize accuracy)
        
        Args:
            points: List of (FLOPs, accuracy) tuples
            
        Returns:
            Pareto optimal points
        """
        if not points:
            return []
        
        # Sort by FLOPs
        sorted_points = sorted(points, key=lambda p: p[0])
        
        pareto = []
        max_accuracy = -float('inf')
        
        for flops, accuracy in sorted_points:
            if accuracy > max_accuracy:
                pareto.append((flops, accuracy))
                max_accuracy = accuracy
        
        return pareto
    
    def generate_feedback(self, submission_id: str) -> Dict:
        """
        Generate rank-specific feedback for a submission
        
        Args:
            submission_id: Submission identifier
            
        Returns:
            Feedback dictionary
        """
        # Find submission
        submission = None
        for sub in self.global_submissions:
            if sub.submission_id == submission_id:
                submission = sub
                break
        
        if not submission:
            raise ValueError(f"Submission {submission_id} not found")
        
        feedback = {
            'submission_id': submission_id,
            'contributor_id': submission.contributor_id,
            'overall_performance': {},
            'rank_specific_feedback': {},
            'recommendations': []
        }
        
        # Analyze each rank
        for rank, metrics in submission.results.items():
            accuracy = metrics.get('accuracy', 0)
            flops = metrics.get('flops', 0)
            uncertainty = metrics.get('uncertainty', 1)
            
            # Rank-specific feedback
            rank_feedback = {
                'expressiveness': self._assess_expressiveness(rank, accuracy),
                'efficiency': self._assess_efficiency(flops, accuracy),
                'uncertainty_level': self._assess_uncertainty(uncertainty),
                'recommendation': self._generate_rank_recommendation(
                    rank, accuracy, flops, uncertainty
                )
            }
            
            feedback['rank_specific_feedback'][rank] = rank_feedback
        
        # Overall recommendations
        best_rank, best_acc = submission.get_best_rank()
        feedback['recommendations'].append(
            f"Best performance at rank {best_rank} with {best_acc:.2%} accuracy"
        )
        
        # Efficiency recommendation
        pareto_points = submission.get_pareto_frontier_point()
        if pareto_points:
            most_efficient = min(pareto_points, key=lambda p: p[0] / p[1])
            feedback['recommendations'].append(
                f"Most efficient at {most_efficient[0]:.0f} FLOPs with {most_efficient[1]:.2%} accuracy"
            )
        
        return feedback
    
    def _assess_expressiveness(self, rank: int, accuracy: float) -> str:
        """Assess model expressiveness at given rank"""
        if rank >= 128 and accuracy >= 0.90:
            return "High expressiveness - model can capture complex patterns"
        elif rank >= 64 and accuracy >= 0.80:
            return "Medium expressiveness - good for most tasks"
        else:
            return "Limited expressiveness - consider higher rank for complex edits"
    
    def _assess_efficiency(self, flops: float, accuracy: float) -> str:
        """Assess computational efficiency"""
        efficiency = accuracy / (flops / 1e6)  # Accuracy per MFLOPs
        
        if efficiency > 0.01:
            return "Excellent efficiency"
        elif efficiency > 0.005:
            return "Good efficiency"
        else:
            return "Low efficiency - consider lower rank"
    
    def _assess_uncertainty(self, uncertainty: float) -> str:
        """Assess prediction uncertainty"""
        if uncertainty < 0.1:
            return "Low uncertainty - high confidence"
        elif uncertainty < 0.2:
            return "Medium uncertainty - acceptable"
        else:
            return "High uncertainty - model may need more training"
    
    def _generate_rank_recommendation(self, rank: int, accuracy: float,
                                     flops: float, uncertainty: float) -> str:
        """Generate specific recommendation for rank"""
        if accuracy >= 0.90 and uncertainty < 0.1:
            return f"Rank {rank} is optimal for this task"
        elif accuracy < 0.80:
            return f"Consider increasing rank from {rank} to improve accuracy"
        elif flops > 1e8:
            return f"Consider decreasing rank from {rank} to reduce compute"
        else:
            return f"Rank {rank} provides good balance"
    
    def export_leaderboard(self, challenge_id: str, filepath: str):
        """Export leaderboard to JSON file"""
        leaderboard = self.get_leaderboard(challenge_id)
        
        with open(filepath, 'w') as f:
            json.dump({
                'challenge_id': challenge_id,
                'leaderboard': leaderboard,
                'exported_at': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Exported leaderboard to {filepath}")


def create_nsn_leaderboard() -> NSNLeaderboard:
    """Factory function to create NSN leaderboard"""
    return NSNLeaderboard()
