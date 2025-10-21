# -*- coding: utf-8 -*-
"""
NSN Dashboard for Visualization and Monitoring
Interactive dashboard for NSN rank selection, multilingual evaluation, and leaderboards
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class NSNDashboard:
    """
    Comprehensive dashboard for NSN visualization and monitoring
    """
    
    def __init__(self, figsize=(15, 10)):
        """
        Initialize NSN dashboard
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = figsize
    
    def plot_flops_vs_reliability(self, 
                                  backend_curves: Dict[str, List[tuple]],
                                  save_path: Optional[str] = None):
        """
        Plot FLOPs vs Reliability curves for different backends
        
        Args:
            backend_curves: Dict mapping backend name to list of (FLOPs, reliability) tuples
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(backend_curves)))
        
        for (backend_name, curve), color in zip(backend_curves.items(), colors):
            flops = [point[0] for point in curve]
            reliability = [point[1] for point in curve]
            
            ax.plot(flops, reliability, marker='o', label=backend_name,
                   color=color, linewidth=2, markersize=8)
        
        ax.set_xlabel('FLOPs', fontsize=14, fontweight='bold')
        ax.set_ylabel('Edit Reliability', fontsize=14, fontweight='bold')
        ax.set_title('Compute-Performance Frontier: FLOPs vs Edit Reliability',
                    fontsize=16, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved FLOPs vs Reliability plot to {save_path}")
        
        plt.show()
        return fig
    
    def plot_multilingual_heatmap(self,
                                  accuracy_matrix: Dict[str, Dict[int, float]],
                                  save_path: Optional[str] = None):
        """
        Plot heatmap of accuracy across languages and ranks
        
        Args:
            accuracy_matrix: Dict mapping language to dict of rank->accuracy
            save_path: Optional path to save figure
        """
        # Convert to 2D array
        languages = list(accuracy_matrix.keys())
        ranks = sorted(list(accuracy_matrix[languages[0]].keys()))
        
        data = np.array([
            [accuracy_matrix[lang][rank] for rank in ranks]
            for lang in languages
        ])
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn',
                   xticklabels=ranks, yticklabels=languages,
                   cbar_kws={'label': 'Edit Accuracy'},
                   vmin=0.5, vmax=1.0, ax=ax)
        
        ax.set_xlabel('NSN Rank', fontsize=14, fontweight='bold')
        ax.set_ylabel('Language', fontsize=14, fontweight='bold')
        ax.set_title('Multilingual Edit Accuracy Across NSN Ranks',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved multilingual heatmap to {save_path}")
        
        plt.show()
        return fig
    
    def plot_subspace_containment(self,
                                  containment_data: List[Dict],
                                  save_path: Optional[str] = None):
        """
        Visualize nested subspace containment across languages
        
        Args:
            containment_data: List of containment analysis dicts
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Group by rank
        ranks = sorted(set(d['rank'] for d in containment_data))
        
        # Plot 1: Containment score by rank
        for rank in ranks:
            rank_data = [d for d in containment_data if d['rank'] == rank]
            pairs = [f"{d['source'][:3]}->{d['target'][:3]}" for d in rank_data]
            scores = [d['containment'] for d in rank_data]
            
            x_pos = np.arange(len(pairs))
            ax1.plot(x_pos, scores, marker='o', label=f'Rank {rank}',
                    linewidth=2, markersize=8)
        
        ax1.set_xlabel('Language Pair', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Containment Score', fontsize=12, fontweight='bold')
        ax1.set_title('Subspace Containment Across Ranks',
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Plot 2: Overlap ratio distribution
        overlap_by_rank = {rank: [] for rank in ranks}
        for d in containment_data:
            overlap_by_rank[d['rank']].append(d['overlap'])
        
        positions = np.arange(len(ranks))
        bp = ax2.boxplot([overlap_by_rank[r] for r in ranks],
                         positions=positions,
                         labels=[f'Rank {r}' for r in ranks],
                         patch_artist=True)
        
        for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0, 1, len(ranks)))):
            patch.set_facecolor(color)
        
        ax2.set_xlabel('NSN Rank', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Overlap Ratio', fontsize=12, fontweight='bold')
        ax2.set_title('Subspace Overlap Distribution',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved subspace containment plot to {save_path}")
        
        plt.show()
        return fig
    
    def plot_pareto_frontier(self,
                            frontier_data: Dict,
                            save_path: Optional[str] = None):
        """
        Plot compute-performance Pareto frontier
        
        Args:
            frontier_data: Frontier data from NSNLeaderboard
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot all points
        all_points = frontier_data['all_points']
        if all_points:
            flops_all = [p[0] for p in all_points]
            acc_all = [p[1] for p in all_points]
            ax.scatter(flops_all, acc_all, alpha=0.4, s=50,
                      label='All Submissions', color='gray')
        
        # Plot Pareto frontier
        frontier = frontier_data['frontier']
        if frontier:
            flops_frontier = [p[0] for p in frontier]
            acc_frontier = [p[1] for p in frontier]
            ax.plot(flops_frontier, acc_frontier, 'r-', linewidth=3,
                   marker='*', markersize=15, label='Pareto Frontier')
        
        # Plot contributor-specific points
        contributor_points = frontier_data.get('contributor_points', {})
        colors = plt.cm.tab10(np.linspace(0, 1, len(contributor_points)))
        
        for (contributor, points), color in zip(contributor_points.items(), colors):
            if points:
                flops_c = [p[0] for p in points]
                acc_c = [p[1] for p in points]
                ax.scatter(flops_c, acc_c, s=100, alpha=0.7,
                          label=contributor, color=color, edgecolors='black')
        
        ax.set_xlabel('FLOPs (Computational Cost)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Edit Accuracy', fontsize=14, fontweight='bold')
        ax.set_title('Compute-Performance Pareto Frontier',
                    fontsize=16, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Pareto frontier plot to {save_path}")
        
        plt.show()
        return fig
    
    def plot_leaderboard_rankings(self,
                                 leaderboard: List[Dict],
                                 top_n: int = 10,
                                 save_path: Optional[str] = None):
        """
        Visualize leaderboard rankings
        
        Args:
            leaderboard: Leaderboard data from NSNLeaderboard
            top_n: Number of top contributors to show
            save_path: Optional path to save figure
        """
        top_entries = leaderboard[:top_n]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Overall scores
        contributors = [e['contributor_id'][:15] for e in top_entries]
        scores = [e['score'] for e in top_entries]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(contributors)))
        bars1 = ax1.barh(contributors, scores, color=colors, edgecolor='black')
        
        ax1.set_xlabel('Overall Score', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Contributor', fontsize=12, fontweight='bold')
        ax1.set_title(f'Top {top_n} Contributors by Score',
                     fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, score in zip(bars1, scores):
            ax1.text(score, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', ha='left', va='center',
                    fontweight='bold', fontsize=10)
        
        # Plot 2: Best accuracy vs best rank
        best_ranks = [e['best_rank'] for e in top_entries]
        best_accs = [e['best_accuracy'] for e in top_entries]
        
        scatter = ax2.scatter(best_ranks, best_accs, s=200, c=scores,
                            cmap='viridis', alpha=0.7, edgecolors='black',
                            linewidth=2)
        
        # Add contributor labels
        for i, contributor in enumerate(contributors):
            ax2.annotate(contributor, (best_ranks[i], best_accs[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        ax2.set_xlabel('Best Rank', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Best Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Best Performance: Rank vs Accuracy',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Overall Score', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved leaderboard rankings to {save_path}")
        
        plt.show()
        return fig
    
    def plot_uncertainty_analysis(self,
                                 language_results: Dict[str, List],
                                 save_path: Optional[str] = None):
        """
        Plot uncertainty analysis across languages and ranks
        
        Args:
            language_results: Dict mapping language to list of result dicts
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Uncertainty vs Rank
        for lang, results in language_results.items():
            ranks = [r['rank'] for r in results]
            uncertainties = [r['uncertainty'] for r in results]
            ax1.plot(ranks, uncertainties, marker='o', label=lang,
                    linewidth=2, markersize=8)
        
        ax1.set_xlabel('NSN Rank', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Uncertainty', fontsize=12, fontweight='bold')
        ax1.set_title('Uncertainty Reduction Across Ranks',
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        
        # Plot 2: Accuracy vs Uncertainty scatter
        for lang, results in language_results.items():
            accuracies = [r['accuracy'] for r in results]
            uncertainties = [r['uncertainty'] for r in results]
            ax2.scatter(uncertainties, accuracies, s=100, alpha=0.6,
                       label=lang, edgecolors='black')
        
        ax2.set_xlabel('Uncertainty', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Accuracy-Uncertainty Trade-off',
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved uncertainty analysis to {save_path}")
        
        plt.show()
        return fig
    
    def create_comprehensive_dashboard(self,
                                      backend_curves: Dict,
                                      accuracy_matrix: Dict,
                                      containment_data: List,
                                      frontier_data: Dict,
                                      leaderboard: List,
                                      save_path: Optional[str] = None):
        """
        Create comprehensive multi-panel dashboard
        
        Args:
            backend_curves: Backend performance curves
            accuracy_matrix: Multilingual accuracy matrix
            containment_data: Subspace containment data
            frontier_data: Pareto frontier data
            leaderboard: Leaderboard rankings
            save_path: Optional path to save figure
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: FLOPs vs Reliability
        ax1 = fig.add_subplot(gs[0, :2])
        for backend_name, curve in backend_curves.items():
            flops = [point[0] for point in curve]
            reliability = [point[1] for point in curve]
            ax1.plot(flops, reliability, marker='o', label=backend_name, linewidth=2)
        ax1.set_xlabel('FLOPs', fontweight='bold')
        ax1.set_ylabel('Reliability', fontweight='bold')
        ax1.set_title('Backend Performance Curves', fontweight='bold', fontsize=12)
        ax1.set_xscale('log')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Leaderboard Top 5
        ax2 = fig.add_subplot(gs[0, 2])
        top5 = leaderboard[:5]
        contributors = [e['contributor_id'][:10] for e in top5]
        scores = [e['score'] for e in top5]
        ax2.barh(contributors, scores, color=plt.cm.viridis(np.linspace(0.3, 0.9, 5)))
        ax2.set_xlabel('Score', fontweight='bold', fontsize=10)
        ax2.set_title('Top 5 Contributors', fontweight='bold', fontsize=12)
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Panel 3: Multilingual Heatmap
        ax3 = fig.add_subplot(gs[1, :])
        languages = list(accuracy_matrix.keys())[:8]  # Limit for visibility
        ranks = sorted(list(accuracy_matrix[languages[0]].keys()))
        data = np.array([[accuracy_matrix[lang][rank] for rank in ranks] for lang in languages])
        sns.heatmap(data, annot=True, fmt='.2f', cmap='RdYlGn',
                   xticklabels=ranks, yticklabels=languages,
                   vmin=0.5, vmax=1.0, ax=ax3, cbar_kws={'label': 'Accuracy'})
        ax3.set_title('Multilingual Performance Matrix', fontweight='bold', fontsize=12)
        
        # Panel 4: Pareto Frontier
        ax4 = fig.add_subplot(gs[2, :2])
        all_points = frontier_data['all_points']
        if all_points:
            flops_all = [p[0] for p in all_points]
            acc_all = [p[1] for p in all_points]
            ax4.scatter(flops_all, acc_all, alpha=0.3, s=30, color='gray')
        frontier = frontier_data['frontier']
        if frontier:
            flops_f = [p[0] for p in frontier]
            acc_f = [p[1] for p in frontier]
            ax4.plot(flops_f, acc_f, 'r-', linewidth=2, marker='*', markersize=10)
        ax4.set_xlabel('FLOPs', fontweight='bold')
        ax4.set_ylabel('Accuracy', fontweight='bold')
        ax4.set_title('Compute-Performance Frontier', fontweight='bold', fontsize=12)
        ax4.set_xscale('log')
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: Containment Summary
        ax5 = fig.add_subplot(gs[2, 2])
        ranks_cont = sorted(set(d['rank'] for d in containment_data))
        avg_containment = [np.mean([d['containment'] for d in containment_data if d['rank'] == r])
                          for r in ranks_cont]
        ax5.plot(ranks_cont, avg_containment, marker='o', linewidth=2, markersize=8, color='purple')
        ax5.set_xlabel('Rank', fontweight='bold', fontsize=10)
        ax5.set_ylabel('Avg Containment', fontweight='bold', fontsize=10)
        ax5.set_title('Subspace Containment', fontweight='bold', fontsize=12)
        ax5.grid(True, alpha=0.3)
        
        fig.suptitle('NSN Comprehensive Dashboard', fontsize=18, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comprehensive dashboard to {save_path}")
        
        plt.show()
        return fig


def create_nsn_dashboard(figsize=(15, 10)) -> NSNDashboard:
    """Factory function to create NSN dashboard"""
    return NSNDashboard(figsize=figsize)
