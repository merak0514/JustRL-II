from unittest.mock import patch

from miles.utils.tracking_utils import swanlab_utils


def test_log_metrics_preserves_step_axes_keys():
    logged: list[tuple[dict, int | None]] = []

    def _fake_log(data, step=None):
        logged.append((data, step))

    with patch.object(swanlab_utils, "swanlab", create=True) as mock_swanlab:
        mock_swanlab.log = _fake_log
        swanlab_utils.log_metrics(
            {
                "rollout/step": 3,
                "rollout/reward": 0.5,
            },
            step_key="rollout/step",
        )

    assert logged[0][0] == {"rollout/step": 3.0}
    assert logged[0][1] == 3
    assert logged[1][0] == {"rollout/reward": 0.5}
    assert logged[1][1] == 3
