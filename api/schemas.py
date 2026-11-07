"""
schemas.py
Pydantic request/response models for the TerraWatt FastAPI service.
Includes standard telemetry, cost-optimal schedule recommendations, and pooling benefit schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any


class IngestPayload(BaseModel):
    node_id: str
    date: str
    actual_value: float


class ScheduleRequest(BaseModel):
    """Request for cost-optimal schedule recommendation."""
    node_id: str = Field(..., description="Hierarchy node ID (e.g., 'India/NR/Delhi')")
    date: Optional[str] = Field(None, description="Target date for schedule (YYYY-MM-DD); defaults to latest available date")
    normal_rate: Optional[float] = Field(
        default=None,
        description="Normal Rate in INR/MWh. Defaults to 4000 (₹4/kWh) if not provided."
    )


class ScheduleRecommendation(BaseModel):
    """Response with cost-optimal schedule recommendation."""
    node_id: str
    date: str
    point_forecast_mw: float = Field(..., description="Raw MinT-Shrink forecast in MW")
    forecast_std_mw: float = Field(..., description="Forecast uncertainty (std dev) in MW")
    optimal_schedule_mw: float = Field(..., description="Cost-optimal schedule to submit in MW")
    schedule_bias_mw: float = Field(..., description="Bias = optimal - forecast (positive = over-schedule)")
    schedule_bias_pct: float = Field(..., description="Bias as percentage of forecast")
    expected_penalty_inr: float = Field(..., description="Expected DSM penalty at optimal schedule in INR")
    expected_penalty_at_forecast_inr: float = Field(..., description="Expected penalty if submitting raw forecast")
    penalty_savings_inr: float = Field(..., description="Savings from biasing: penalty_at_forecast - penalty_at_optimal")
    bias_direction: str = Field(..., description="'over_schedule', 'under_schedule', or 'neutral'")
    normal_rate_used: float = Field(..., description="Normal Rate used for calculation in INR/MWh")


class PoolingBenefitResponse(BaseModel):
    """Response with pooling benefit analysis."""
    total_standalone_penalty_inr: float
    total_pooled_penalty_inr: float
    total_savings_inr: float
    savings_pct: float
    by_region: Dict[str, Any]
