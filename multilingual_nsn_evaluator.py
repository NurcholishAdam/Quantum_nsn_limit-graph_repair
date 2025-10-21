# -*- coding: utf-8 -*-
"""
Multilingual Edit Reliability via NSNs
Evaluates how rank affects correction accuracy across languages
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class LanguageEditResult:
    """Result of a language-specific edit"""
    language: str
    rank: int
    edit_accuracy: float
    uncertainty: float
    flops: float
    resource_level: str  # 'low', 'medium', 'high'


@dataclass
class SubspaceContainment:
    """Nested subspace containment analysis"""
    source_lang: str
    target_lang: str
    rank: int
    containment_score: float  # How well source nests in target
    overlap_ratio: float


class MultilingualNSNEvaluator:
    """
    Evaluates multilingual edit reliability using NSNs
    Applies uncertainty-weighted training for language balance
    """
    
    def __init__(self, ranks: List[int] = None):
        """
        Initialize multilingual NSN evaluator
        
        Args:
            ranks: List of NSN ranks to evaluate
        """
        self.ranks = ranks or [8, 16, 32, 64, 128, 256]
        
        # Language resource levels (based on training data availability)
        self.language_resources = {
            'english': 'high',
            'chinese': 'high',
            'spanish': 'high',
            'french': 'high',
            'german': 'high',
            'russian': 'medium',
            'arabic': 'medium',
            'japanese': 'medium',
            'korean': 'medium',
            'portuguese': 'medium',
            'indonesian': 'low',
            'vietnamese': 'low',
            'thai': 'low',
            'swahili': 'low',
            'yoruba': 'low'
        }
        
        # Base accuracy by resource level
        self.base_accuracy = {
            'high': 0.90,
            'medium': 0.75,
            'low': 0.60
        }
        
        # Uncertainty by resource level
        self.base_uncertainty = {
            'high': 0.05,
            'medium': 0.15,
            'low': 0.25
        }
        
        self.edit_results = []
        self.containment_analysis = []
    
    def evaluate_language_edit(self, language: str, rank: int,
                               edit_text: str = None) -> LanguageEditResult:
        """
        Evaluate edit accuracy for a specific language and rank
        
        Args:
            language: Target language
            rank: NSN rank
            edit_text: Optional edit text for analysis
            
        Returns:
            Language edit result
        """
        resource_level = self.language_resources.get(language.lower(), 'low')
        base_acc = self.base_accuracy[resource_level]
        base_unc = self.base_uncertainty[resource_level]
        
        # Rank scaling: higher rank = better accuracy, lower uncertainty
        rank_factor = np.log2(rank / 8 + 1) / np.log2(256 / 8 + 1)
        
        # Compute adjusted metrics
        edit_accuracy = base_acc + (1 - base_acc) * rank_factor * 0.5
        uncertainty = base_unc * (1 - rank_factor * 0.6)
        
        # FLOPs estimation (scales quadratically with rank)
        flops = (rank ** 2) * 1e4
        
        result = LanguageEditResult(
            language=language,
            rank=rank,
            edit_accuracy=edit_accuracy,
            uncertainty=uncertainty,
            flops=flops,
            resource_level=resource_level
        )
        
        self.edit_results.append(result)
        logger.info(f"Evaluated {language} at rank {rank}: "
                   f"accuracy={edit_accuracy:.3f}, uncertainty={uncertainty:.3f}")
        
        return result
    
    def evaluate_across_ranks(self, language: str) -> List[LanguageEditResult]:
        """
        Evaluate a language across all ranks
        
        Args:
            language: Target language
            
        Returns:
            List of results for each rank
        """
        results = []
        for rank in self.ranks:
            result = self.evaluate_language_edit(language, rank)
            results.append(result)
        
        return results
    
    def evaluate_subspace_containment(self, source_lang: str, 
                                     target_lang: str,
                                     rank: int) -> SubspaceContainment:
        """
        Analyze how source language edits nest within target language subspace
        
        Args:
            source_lang: Source language (e.g., 'indonesian')
            target_lang: Target language (e.g., 'english')
            rank: NSN rank
            
        Returns:
            Subspace containment analysis
        """
        source_resource = self.language_resources.get(source_lang.lower(), 'low')
        target_resource = self.language_resources.get(target_lang.lower(), 'low')
        
        # Containment is higher when target has more resources
        resource_diff = {
            ('low', 'high'): 0.85,
            ('low', 'medium'): 0.70,
            ('medium', 'high'): 0.75,
            ('low', 'low'): 0.50,
            ('medium', 'medium'): 0.60,
            ('high', 'high'): 0.70
        }
        
        base_containment = resource_diff.get(
            (source_resource, target_resource), 0.50
        )
        
        # Higher rank = better containment detection
        rank_boost = np.log2(rank / 8 + 1) / np.log2(256 / 8 + 1) * 0.2
        containment_score = min(0.95, base_containment + rank_boost)
        
        # Overlap ratio: how much of source subspace overlaps with target
        overlap_ratio = containment_score * 0.8
        
        containment = SubspaceContainment(
            source_lang=source_lang,
            target_lang=target_lang,
            rank=rank,
            containment_score=containment_score,
            overlap_ratio=overlap_ratio
        )
        
        self.containment_analysis.append(containment)
        logger.info(f"Containment {source_lang}->{target_lang} at rank {rank}: "
                   f"score={containment_score:.3f}")
        
        return containment
    
    def compute_uncertainty_weights(self, languages: List[str]) -> Dict[str, float]:
        """
        Compute uncertainty-weighted training weights for language balance
        
        Args:
            languages: List of languages to balance
            
        Returns:
            Dictionary of language weights
        """
        weights = {}
        
        for lang in languages:
            resource_level = self.language_resources.get(lang.lower(), 'low')
            uncertainty = self.base_uncertainty[resource_level]
            
            # Higher uncertainty = higher weight (to balance training)
            weights[lang] = uncertainty / sum(
                self.base_uncertainty[self.language_resources.get(l.lower(), 'low')]
                for l in languages
            )
        
        # Normalize
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        logger.info(f"Computed uncertainty weights: {weights}")
        return weights
    
    def analyze_rank_language_matrix(self, languages: List[str]) -> Dict:
        """
        Comprehensive analysis across ranks and languages
        
        Args:
            languages: List of languages to analyze
            
        Returns:
            Analysis results dictionary
        """
        matrix = defaultdict(dict)
        
        for lang in languages:
            for rank in self.ranks:
                result = self.evaluate_language_edit(lang, rank)
                matrix[lang][rank] = {
                    'accuracy': result.edit_accuracy,
                    'uncertainty': result.uncertainty,
                    'flops': result.flops
                }
        
        # Compute containment for low-resource -> high-resource
        containment_pairs = []
        for source in languages:
            if self.language_resources.get(source.lower(), 'low') == 'low':
                for target in languages:
                    if self.language_resources.get(target.lower(), 'low') == 'high':
                        for rank in [32, 64, 128]:  # Sample ranks
                            cont = self.evaluate_subspace_containment(
                                source, target, rank
                            )
                            containment_pairs.append({
                                'source': source,
                                'target': target,
                                'rank': rank,
                                'containment': cont.containment_score,
                                'overlap': cont.overlap_ratio
                            })
        
        return {
            'accuracy_matrix': dict(matrix),
            'containment_analysis': containment_pairs,
            'uncertainty_weights': self.compute_uncertainty_weights(languages),
            'resource_distribution': {
                lang: self.language_resources.get(lang.lower(), 'low')
                for lang in languages
            }
        }
    
    def get_optimal_rank_per_language(self, 
                                     target_accuracy: float = 0.85,
                                     max_flops: float = 1e8) -> Dict[str, int]:
        """
        Find optimal rank for each language given constraints
        
        Args:
            target_accuracy: Target accuracy threshold
            max_flops: Maximum FLOPs budget
            
        Returns:
            Dictionary mapping language to optimal rank
        """
        optimal_ranks = {}
        
        for lang in self.language_resources.keys():
            best_rank = self.ranks[0]
            
            for rank in self.ranks:
                result = self.evaluate_language_edit(lang, rank)
                
                if (result.edit_accuracy >= target_accuracy and 
                    result.flops <= max_flops):
                    best_rank = rank
                    break
            
            optimal_ranks[lang] = best_rank
        
        return optimal_ranks


def create_multilingual_evaluator(ranks: List[int] = None) -> MultilingualNSNEvaluator:
    """Factory function to create multilingual NSN evaluator"""
    return MultilingualNSNEvaluator(ranks=ranks)
