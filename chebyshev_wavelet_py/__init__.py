"""NumPy implementation of the Chebyshev-wavelet numerical core.

The package mirrors the zero-based cell convention used by the Lean 4 proof
development: MATLAB level ``k`` corresponds to ``J = k - 1`` here.
"""

from .chebyshev_core import (
    chebyshev_u,
    cell_bounds,
    cell_coordinate,
    wavelet_scale,
    wavelet,
    weighted_inner_product,
)
from .operators import (
    integration_blocks,
    operational_matrix_of_integration,
    chebyshev_u_linearization,
    product_operational_matrix,
)
from .signal_generator import CausalWaveletSignalGenerator
from .execution_guard import CostGuardConfig, PreTradeCostGuard
from .ib_adapter import IBPaperAdapter, HistoricalBars, PositionSnapshot
from .live_orchestrator import InstrumentSpec, LiveOrchestrator, OrchestratorConfig
from .config_loader import ConfigLoader, SystemConfig, load_config
from .data_store import DataStore, DecisionRecord, DecisionStore
from .extended_signals import (
    CandleAnatomy,
    ExtendedSignalAnalyzer,
    FactorSpec,
    TechnicalSignalSnapshot,
    causal_rolling_volatility,
    combine_factors,
    nonlinear_momentum_filter,
    volatility_adjusted_signal,
)
from .price_quality import (
    PriceQualityError,
    dividend_drag_spread,
    implied_dividend_yield,
    is_total_return,
    require_total_return,
)
from .risk_manager import RiskConfig, RiskDecision, RiskManager, SettlementCheck, ShareQuantity

__all__ = [
    "CausalWaveletSignalGenerator",
    "CostGuardConfig",
    "ConfigLoader",
    "DataStore",
    "DecisionRecord",
    "DecisionStore",
    "CandleAnatomy",
    "FactorSpec",
    "PriceQualityError",
    "ExtendedSignalAnalyzer",
    "HistoricalBars",
    "IBPaperAdapter",
    "InstrumentSpec",
    "LiveOrchestrator",
    "OrchestratorConfig",
    "PositionSnapshot",
    "RiskConfig",
    "RiskDecision",
    "RiskManager",
    "SettlementCheck",
    "ShareQuantity",
    "SystemConfig",
    "TechnicalSignalSnapshot",
    "causal_rolling_volatility",
    "dividend_drag_spread",
    "implied_dividend_yield",
    "is_total_return",
    "require_total_return",
    "combine_factors",
    "PreTradeCostGuard",
    "cell_bounds",
    "cell_coordinate",
    "chebyshev_u",
    "chebyshev_u_linearization",
    "integration_blocks",
    "operational_matrix_of_integration",
    "product_operational_matrix",
    "wavelet",
    "wavelet_scale",
    "weighted_inner_product",
    "load_config",
    "nonlinear_momentum_filter",
    "volatility_adjusted_signal",
]
