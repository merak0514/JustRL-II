from miles.utils.tracking_utils.sglang_metrics_utils import (
    _aggregate_worker_metrics_texts,
    parse_prometheus_metrics,
)


def test_parse_prometheus_metrics():
    text = """
# HELP sglang:gen_throughput gen
sglang:gen_throughput 1.5
sglang:num_running_reqs 2
"""
    metrics = parse_prometheus_metrics(text)
    assert metrics["gen_throughput"] == 1.5
    assert metrics["num_running_reqs"] == 2.0


def test_aggregate_worker_metrics_texts_simple():
    merged = _aggregate_worker_metrics_texts(
        [
            "sglang:gen_throughput 1.0\nsglang:num_running_reqs 2",
            "sglang:gen_throughput 3.0\nsglang:num_running_reqs 4",
        ]
    )
    assert merged["gen_throughput"] == 2.0
    assert merged["num_running_reqs"] == 6.0
    assert merged["num_engines"] == 2.0


def test_aggregate_worker_metrics_texts_multiple_label_series():
    merged = _aggregate_worker_metrics_texts(
        [
            'sglang:gen_throughput{gpu="0"} 1.0\nsglang:gen_throughput{gpu="1"} 2.0',
            'sglang:gen_throughput{gpu="0"} 3.0',
        ]
    )
    assert merged["gen_throughput"] == 2.0
    assert merged["num_engines"] == 2.0


def test_aggregate_worker_metrics_texts_empty_worker_body():
    merged = _aggregate_worker_metrics_texts(["# no sglang metrics here\nother_metric 1.0"])
    assert merged == {"num_engines": 1.0}
