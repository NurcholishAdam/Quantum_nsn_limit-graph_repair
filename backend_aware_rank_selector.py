#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend-Aware Rank Selection using Nested Subspace Networks (NSNs)
Dynamically adjusts model rank based on quantum backend constraints
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class BackendType(Enum):
    """Quantum backend types with different characteristics"""
    IBM_MANILA = "ibm_manila"  # Low-qubit, noisy
    IBM_WASHINGTON = "ibm_washington"  # High-fidelity
    RUSSIAN_SIMULATOR = "russian_simulator"  # Stable simulator
    IBM_SIMULATOR = "ibm_simulator"  # Standard simulator


@dataclass
class BackendConstraints:
    """Constraints for a quantum backend"""
    backend_type: BackendType
    num_qubits: int
    error_rate: float
    gate_fidelity: float
    coherence_time_us: float
    max_circuit_depth: int


@dataclass
class RankConfig:
    """NSN rank configuration"""
    rank: int
    flops: float
    expected_reliability: float
    memory_mb: float
    inference_time_ms: float


class BackendAwareRankSelector:
    """
    Selects optimal NSN rank based on quantum backend constraints
    """
    
    def __init__(self):
        # Define backend constraints
        self.backend_constraints = {
            BackendType.IBM_MANILA: BackendConstraints(
                backend_type=BackendType.IBM_MANILA,
                num_qubits=5,
                error_rate=0.05,
                gate_fidelity=0.95,
                coherence_time_us=50,
                max_circuit_depth=20
            ),
            BackendType.IBM_WASHINGTON: BackendConstraints(
                backend_type=BackendType.IBM_WASHINGTON,
                num_qubits=127,
                error_rate=0.001,
                gate_fidelity=0.999,
                coherence_time_us=200,
                max_circuit_depth=100
            ),
            BackendType.RUSSIAN_SIMULATOR: BackendConstraints(
                backend_type=BackendType.RUSSIAN_SIMULATOR,
                num_qubits=1000,
                error_rate=0.0001,
                gate_fidelity=0.9999,
                coherence_time_us=1000,
                max_circuit_depth=500
            ),
            BackendType.IBM_SIMULATOR: BackendConstraints(
                backend_type=BackendType.IBM_SIMULATOR,
                num_qubits=1000,
                error_rate=0.0001,
                gate_fidelity=0.9999,
                coherence_time_us=1000,
                max_circuit_depth=500
            )
        }
        
        # Define rank configurations (from low to high)
        self.rank_configs = [
            RankConfig(rank=8, flops=1e6, expected_reliability=0.75, 
                      memory_mb=50, inference_time_ms=10),
            RankConfig(rank=16, flops=4e6, expected_reliability=0.82,
                      memory_mb=100, inference_time_ms=20),
            RankConfig(rank=32, flops=1.6e7, expected_reliability=0.88,
                      memory_mb=200, inference_time_ms=40),
            RankConfig(rank=64, flops=6.4e7, expected_reliability=0.92,
                      memory_mb=400, inference_time_ms=80),
            RankConfig(rank=128, flops=2.56e8, expected_reliability=0.95,
                      memory_mb=800, inference_time_ms=160),
            RankConfig(rank=256, flops=1.024e9, expected_reliability=0.97,
                      memory_mb=1600, inference_time_ms=320)
        ]
    
    def select_rank(self, backend_type: BackendType, 
                   target_reliability: float = 0.85) -> RankConfig:
        """
        Select optimal rank based on backend constraints
        
        Args:
            backend_type: Type of quantum backend
            target_reliability: Target edit reliability
            
        Returns:
            Optimal rank configuration
        """
        constraints = self.backend_constraints[backend_type]
        
        # Low-qubit or noisy backends -> low rank
        if constraints.num_qubits < 10 or constraints.error_rate > 0.01:
            # Use low-rank inference
            selected_rank = self.rank_configs[0]  # rank=8
            
        # Medium-fidelity backends -> medium rank
        elif constraints.num_qubits < 50 or constraints.error_rate > 0.005:
            selected_rank = self.rank_configs[2]  # rank=32
            
        # High-fidelity backends -> high rank
        else:
            # Select rank that meets target reliability
            for rank_config in reversed(self.rank_configs):
                if rank_config.expected_reliability >= target_reliability:
                    selected_rank = rank_config
                    break
            else:
                selected_rank = self.rank_configs[-1]  # highest rank
        
        return selected_rank
    
    def compute_flops_vs_reliability(self, backend_type: BackendType) -> List[Tuple[float, float]]:
        """
        Compute FLOPs vs reliability curve for a backend
        
        Args:
            backend_type: Type of quantum backend
            
        Returns:
            List of (FLOPs, reliability) tuples
        """
        constraints = self.backend_constraints[backend_type]
        
        # Adjust reliability based on backend quality
        quality_factor = constraints.gate_fidelity * (1 - constraints.error_rate)
        
        curve = []
        for rank_config in self.rank_configs:
            adjusted_reliability = rank_config.expected_reliability * quality_factor
            curve.append((rank_config.flops, adjusted_reliability))
        
        return curve
    
    def get_rank_recommendation(self, backend_type: BackendType,
                               compute_budget: float,
                               min_reliability: float) -> Dict:
        """
        Get rank recommendation with detailed analysis
        
        Args:
            backend_type: Type of quantum backend
            compute_budget: Available compute budget (FLOPs)
            min_reliability: Minimum required reliability
            
        Returns:
            Recommendation dictionary
        """
        constraints = self.backend_constraints[backend_type]
        selected_rank = self.select_rank(backend_type, min_reliability)
        
        # Check if within budget
        within_budget = selected_rank.flops <= compute_budget
        
        # Find alternative if over budget
        alternative = None
        if not within_budget:
            for rank_config in self.rank_configs:
                if rank_config.flops <= compute_budget:
                    alternative = rank_config
        
        return {
            'backend_type': backend_type.value,
            'backend_constraints': {
                'num_qubits': constraints.num_qubits,
                'error_rate': constraints.error_rate,
                'gate_fidelity': constraints.gate_fidelity
            },
            'recommended_rank': selected_rank.rank,
            'flops': selected_rank.flops,
            'expected_reliability': selected_rank.expected_reliability,
            'memory_mb': selected_rank.memory_mb,
            'inference_time_ms': selected_rank.inference_time_ms,
            'within_budget': within_budget,
            'alternative_rank': alternative.rank if alternative else None,
            'rationale': self._generate_rationale(backend_type, selected_rank)
        }
    
    def _generate_rationale(self, backend_type: BackendType, 
                           rank_config: RankConfig) -> str:
        """Generate human-readable rationale for rank selection"""
        constraints = self.backend_constraints[backend_type]
        
        if constraints.num_qubits < 10:
            return f"Low-qubit backend ({constraints.num_qubits} qubits) requires low-rank (r={rank_config.rank}) for stability"
        elif constraints.error_rate > 0.01:
            return f"High error rate ({constraints.error_rate:.3f}) necessitates low-rank (r={rank_config.rank}) inference"
        elif constraints.gate_fidelity > 0.999:
            return f"High-fidelity backend (fidelity={constraints.gate_fidelity:.4f}) supports high-rank (r={rank_config.rank}) for maximum accuracy"
        else:
            return f"Medium-fidelity backend balanced with rank={rank_config.rank} for optimal reliability"


def create_rank_selector() -> BackendAwareRankSelector:
    """Factory function to create rank selector"""
    return BackendAwareRankSelector()
