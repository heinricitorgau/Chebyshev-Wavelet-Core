"""Safe, typed access to the project's single JSON configuration source."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar


class ConfigurationError(ValueError):
    """Raised when configuration is absent, malformed, or violates safety rules."""


_MISSING = object()
T = TypeVar("T")


@dataclass(frozen=True)
class SystemSettings:
    host: str
    paper_only: bool
    submit_orders: bool
    port: int
    client_id: int
    account: str | None


@dataclass(frozen=True)
class TradingRules:
    max_break_even_bps: float
    unit_shares: int
    allow_odd_lots: bool
    stop_loss_pct: float
    reference_turnover: float
    order_slippage_bps: float


@dataclass(frozen=True)
class IndicatorSettings:
    wavelet_window_size: int
    wavelet_mode_count: int
    wavelet_signal_mode: int
    wavelet_threshold: float
    sma_fast_window: int
    sma_slow_window: int
    support_resistance_window: int
    sma_trend_filter: bool


@dataclass(frozen=True)
class SystemConfig:
    """Immutable, validated snapshot of all runtime configuration sections."""

    schema_version: int
    system: SystemSettings
    trading_rules: TradingRules
    indicators: IndicatorSettings
    risk: Mapping[str, Any]
    data_store: Mapping[str, Any]
    orchestration: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConfigLoader:
    """Read JSON once and expose validated dotted-path access.

    The loader performs no environment expansion and never writes to the
    source file, so a returned configuration can be persisted beside a replay
    log without hidden process-specific changes.
    """

    def __init__(self, path: str | Path = "config.json") -> None:
        self.path = Path(path)
        self._raw = self._read()

    def get(self, dotted_path: str, default: T | object = _MISSING) -> Any | T:
        """Return a dotted-path value, or an explicit default when provided."""
        current: Any = self._raw
        for part in dotted_path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                if default is _MISSING:
                    raise ConfigurationError(f"missing required configuration key: {dotted_path}")
                return default
            current = current[part]
        return current

    def section(self, name: str) -> Mapping[str, Any]:
        """Return a shallow copy of one top-level JSON object section."""
        value = self.get(name)
        if not isinstance(value, Mapping):
            raise ConfigurationError(f"'{name}' must be a JSON object")
        return dict(value)

    @property
    def system(self) -> SystemSettings:
        return _dataclass_from(SystemSettings, self.section("system"), "system")

    @property
    def trading_rules(self) -> TradingRules:
        return _dataclass_from(TradingRules, self.section("trading_rules"), "trading_rules")

    @property
    def indicators(self) -> IndicatorSettings:
        return _dataclass_from(IndicatorSettings, self.section("indicators"), "indicators")

    def load(self) -> SystemConfig:
        """Validate cross-section invariants and return the complete snapshot."""
        config = SystemConfig(
            schema_version=_positive_int(self.get("schema_version"), "schema_version"),
            system=self.system,
            trading_rules=self.trading_rules,
            indicators=self.indicators,
            risk=self.section("risk"),
            data_store=self.section("data_store"),
            orchestration=self.section("orchestration"),
        )
        _validate(config)
        return config

    def _read(self) -> Mapping[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigurationError(f"configuration file not found: {self.path}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"invalid JSON in {self.path}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ConfigurationError("configuration root must be a JSON object")
        return raw


def load_config(path: str | Path = "config.json") -> SystemConfig:
    """Compatibility helper returning ``ConfigLoader(path).load()``."""
    return ConfigLoader(path).load()


def _dataclass_from(class_type: type[T], section: Mapping[str, Any], name: str) -> T:
    allowed = set(class_type.__dataclass_fields__)
    unknown = set(section) - allowed
    missing = allowed - set(section)
    if unknown:
        raise ConfigurationError(f"unknown settings in '{name}': {sorted(unknown)}")
    if missing:
        raise ConfigurationError(f"missing settings in '{name}': {sorted(missing)}")
    try:
        return class_type(**section)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid '{name}' settings: {exc}") from exc


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigurationError(f"'{name}' must be a positive integer")
    return value


def _validate(config: SystemConfig) -> None:
    system = config.system
    rules = config.trading_rules
    indicators = config.indicators
    if system.paper_only and system.port not in {7497, 4002}:
        raise ConfigurationError("paper_only requires IB Paper Trading port 7497 or 4002")
    if system.submit_orders and not system.paper_only:
        raise ConfigurationError("live order submission is refused by this configuration loader")
    if rules.max_break_even_bps <= 0.0 or rules.reference_turnover <= 0.0:
        raise ConfigurationError("cost thresholds must be positive")
    if rules.unit_shares < 1 or not 0.0 < rules.stop_loss_pct < 1.0:
        raise ConfigurationError("unit_shares and stop_loss_pct are invalid")
    if indicators.wavelet_window_size < 2 or indicators.wavelet_mode_count < 1:
        raise ConfigurationError("wavelet dimensions are invalid")
    if not 0 <= indicators.wavelet_signal_mode < indicators.wavelet_mode_count:
        raise ConfigurationError("wavelet_signal_mode must index a retained mode")
    if not 1 <= indicators.sma_fast_window < indicators.sma_slow_window:
        raise ConfigurationError("SMA windows must satisfy 1 <= fast < slow")
    if indicators.support_resistance_window < 2:
        raise ConfigurationError("support_resistance_window must be at least two")
    if not isinstance(indicators.sma_trend_filter, bool):
        raise ConfigurationError("sma_trend_filter must be a boolean")
