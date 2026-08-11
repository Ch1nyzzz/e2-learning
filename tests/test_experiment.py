from collections.abc import Sequence
from pathlib import Path

from experience_learning.config import AppConfig
from experience_learning.experiment import OnlineExperienceExperiment
from experience_learning.judge import ExactMatchJudge
from experience_learning.types import EnvironmentState, EpisodeContext, Experience


class FakeEnvironment:
    def __init__(self, actual: str):
        self.actual = actual
        self.step_calls = 0
        self.closed = False

    def reset(self) -> EnvironmentState:
        return EnvironmentState("start", ("certain", "uncertain"))

    def step(self, action: str) -> EnvironmentState:
        self.step_calls += 1
        assert action in {"certain", "uncertain"}
        return EnvironmentState(self.actual, (), done=True)

    def close(self) -> None:
        self.closed = True


class FakeModel:
    def __init__(self, *, correct: bool):
        self.correct = correct
        self.learn_calls = 0
        self.save_calls = 0

    def predict(
        self,
        requests: Sequence[tuple[EpisodeContext, str]],
        *,
        samples_per_request: int,
        do_sample: bool,
        seed: int | None = None,
    ) -> list[list[str]]:
        del samples_per_request, seed
        results = []
        for _, action in requests:
            if not do_sample:
                results.append(["real" if self.correct else "wrong"])
            elif action == "certain":
                results.append(["same", "same", "same", "same"])
            elif self.correct:
                results.append(["real", "real", "real", "other"])
            else:
                results.append(["wrong", "wrong", "wrong", "other"])
        return results

    def learn(self, experiences: Sequence[Experience]) -> dict[str, float]:
        assert experiences
        self.learn_calls += 1
        return {"loss": 1.0, "learning_rate": 1e-6, "optimizer_step": self.learn_calls}

    def save(self, output_dir: str) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.save_calls += 1


class CountingJudge(ExactMatchJudge):
    def __init__(self) -> None:
        self.calls = 0

    def compare(self, **kwargs):
        self.calls += 1
        return super().compare(**kwargs)


def _config(tmp_path: Path) -> AppConfig:
    config = AppConfig()
    config.experiment.output_dir = str(tmp_path)
    config.experiment.max_episodes = 1
    config.experiment.max_environment_steps = 1
    config.training.checkpoint_every_environment_steps = 0
    return config


def test_one_acquisition_executes_environment_once_and_updates_on_mistake(tmp_path: Path) -> None:
    environment = FakeEnvironment("real")
    model = FakeModel(correct=False)
    result = OnlineExperienceExperiment(
        config=_config(tmp_path),
        model=model,
        is_main_process=True,
        environment=environment,
        judge=ExactMatchJudge(),
    ).run()
    assert environment.step_calls == 1
    assert model.learn_calls == 1
    assert result["environment_steps"] == 1
    assert result["mistakes"] == 1


def test_semantically_correct_greedy_prediction_skips_update(tmp_path: Path) -> None:
    environment = FakeEnvironment("real")
    model = FakeModel(correct=True)
    result = OnlineExperienceExperiment(
        config=_config(tmp_path),
        model=model,
        is_main_process=True,
        environment=environment,
        judge=ExactMatchJudge(),
    ).run()
    assert environment.step_calls == 1
    assert model.learn_calls == 0
    assert result["mistakes"] == 0


def test_random_acquisition_skips_entropy_judge_calls(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.acquisition.strategy = "random"
    judge = CountingJudge()
    OnlineExperienceExperiment(
        config=config,
        model=FakeModel(correct=True),
        is_main_process=True,
        environment=FakeEnvironment("real"),
        judge=judge,
    ).run()
    assert judge.calls == 1
