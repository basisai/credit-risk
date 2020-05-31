from bedrock_client.bedrock.metrics.collector.type import Collector
from bedrock_client.bedrock.metrics.exporter.type import LogExporter
from bedrock_client.bedrock.metrics.service import ModelMonitoringService

from flask import Blueprint, Response, current_app, request

model_monitor = Blueprint("model_monitor", __name__)


class EmptyCollector(Collector):
    def collect(self):
        return []


class NoopLogger(LogExporter):
    def emit(self, prediction):
        pass


@model_monitor.before_app_first_request
def init_background_threads():
    """Global objects with daemon threads will be stopped by gunicorn --preload flag.
    So instantiate them here instead.
    """
    current_app.prediction_store = ModelMonitoringService(baseline_collector=EmptyCollector())
    current_app.monitor = ModelMonitoringService(log_exporter=NoopLogger())


@model_monitor.route("/metrics", methods=["GET"])
def get_metrics():
    """Returns real time feature values recorded by prometheus
    """
    body, content_type = current_app.monitor.export_http(
        params=request.args.to_dict(flat=False),
        headers=request.headers,
    )
    return Response(body, content_type=content_type)
