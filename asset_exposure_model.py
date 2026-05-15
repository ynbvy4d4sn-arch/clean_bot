"""Estimate or fall back to asset-factor exposures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from asset_factor_mapping import get_asset_factor_mapping


@dataclass(slots=True)
class AssetExposureModelResult:
    """Estimated asset-factor exposures and diagnostics."""

    exposure_matrix: pd.DataFrame
    residual_volatility: pd.Series
    diagnostics: list[str]


def estimate_asset_factor_exposures(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    date: pd.Timestamp | str | None = None,
    rolling_window: int = 252,
) -> AssetExposureModelResult:
    """Estimate simple linear factor exposures or fall back to qualitative priors."""

    priors = get_asset_factor_mapping()
    diagnostics: list[str] = []
    factor_columns = factor_returns.columns.tolist()
    exposure_matrix = pd.DataFrame(0.0, index=asset_returns.columns, columns=factor_columns, dtype=float)
    residual_volatility = pd.Series(0.0, index=asset_returns.columns, dtype=float)

    if asset_returns.empty or factor_returns.empty:
        diagnostics.append("Insufficient asset/factor history; using prior factor mappings.")
        for asset, mapping in priors.items():
            if asset not in exposure_matrix.index:
                continue
            for factor, value in mapping.items():
                if factor in exposure_matrix.columns:
                    exposure_matrix.loc[asset, factor] = float(value)
            residual_volatility.loc[asset] = 0.10
        return AssetExposureModelResult(exposure_matrix=exposure_matrix, residual_volatility=residual_volatility, diagnostics=diagnostics)

    as_of = pd.Timestamp(date if date is not None else asset_returns.index[-1])
    asset_hist = asset_returns.loc[:as_of].sort_index().tail(rolling_window)
    factor_hist = factor_returns.loc[:as_of].sort_index().tail(rolling_window)
    common_index = asset_hist.index.intersection(factor_hist.index)
    asset_hist = asset_hist.loc[common_index]
    factor_hist = factor_hist.loc[common_index]

    if len(common_index) < 60:
        diagnostics.append("Too little overlapping history for regression; using priors.")
        for asset, mapping in priors.items():
            if asset not in exposure_matrix.index:
                continue
            for factor, value in mapping.items():
                if factor in exposure_matrix.columns:
                    exposure_matrix.loc[asset, factor] = float(value)
            residual_volatility.loc[asset] = float(asset_hist[asset].std(ddof=0)) if asset in asset_hist else 0.10
        return AssetExposureModelResult(exposure_matrix=exposure_matrix, residual_volatility=residual_volatility, diagnostics=diagnostics)

    x = factor_hist.to_numpy(dtype=float)
    ridge = 1e-4
    xtx = x.T @ x + ridge * np.eye(x.shape[1])
    xtx_inv = np.linalg.pinv(xtx)
    for asset in asset_hist.columns:
        y = asset_hist[asset].to_numpy(dtype=float)
        beta = xtx_inv @ x.T @ y
        exposure_matrix.loc[asset] = beta
        residual = y - x @ beta
        residual_volatility.loc[asset] = float(np.std(residual, ddof=0) * np.sqrt(252.0))
    return AssetExposureModelResult(exposure_matrix=exposure_matrix, residual_volatility=residual_volatility, diagnostics=diagnostics)
