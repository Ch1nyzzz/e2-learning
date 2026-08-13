from experience_learning.model import _load_model_only_accelerator_state


class _FakeAccelerator:
    def __init__(self) -> None:
        self._optimizers = [object()]
        self.optimizers_seen_while_loading = None

    def load_state(self, checkpoint: str) -> None:
        assert checkpoint == "checkpoint"
        self.optimizers_seen_while_loading = list(self._optimizers)


def test_model_only_checkpoint_hides_and_restores_preparation_optimizer() -> None:
    accelerator = _FakeAccelerator()
    original_optimizers = accelerator._optimizers

    _load_model_only_accelerator_state(accelerator, "checkpoint")

    assert accelerator.optimizers_seen_while_loading == []
    assert accelerator._optimizers is original_optimizers


def test_model_only_checkpoint_restores_preparation_optimizer_after_failure() -> None:
    class FailingAccelerator(_FakeAccelerator):
        def load_state(self, checkpoint: str) -> None:
            super().load_state(checkpoint)
            raise RuntimeError("load failed")

    accelerator = FailingAccelerator()
    original_optimizers = accelerator._optimizers

    try:
        _load_model_only_accelerator_state(accelerator, "checkpoint")
    except RuntimeError as exc:
        assert str(exc) == "load failed"
    else:
        raise AssertionError("expected checkpoint load to fail")

    assert accelerator.optimizers_seen_while_loading == []
    assert accelerator._optimizers is original_optimizers
