from Framework.ForecastSystem.LSTMModel import (
    GetForecastBackendName,
    GetForecastSequenceLength,
    PredictMultiHorizonForecast,
)
from Framework.ROModule.h1_forecast_features import (
    build_h1_recent_direction_features,
    extract_h1_close_list,
)
from Framework.ROModule.h1_forecast_contract import (
    build_h1_runtime_view,
    evaluate_h1_alignment,
    has_h1_forecast_result,
)
from Framework.ROModule.h1_forecast_runtime import evaluate_h1_forecast_runtime


_MODULE_NAME = "h1_forecast"

def evaluate_h1_forecast(_h1_data, _thresholds):
    return evaluate_h1_forecast_runtime(
        _h1_data=_h1_data,
        _thresholds=_thresholds,
        _required_sequence_length_resolver=GetForecastSequenceLength,
        _forecast_engine_name_resolver=GetForecastBackendName,
        _predictor=PredictMultiHorizonForecast,
        _close_extractor=extract_h1_close_list,
        _recent_feature_builder=build_h1_recent_direction_features,
        _module_name=_MODULE_NAME,
    )
