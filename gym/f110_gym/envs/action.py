from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from f110_gym.envs.dynamic_models import pid_steer, pid_accl


class CarActionEnum(Enum):
    Accl = 1
    Speed = 2

    @staticmethod
    def from_string(action: str):
        if action == "accl":
            return AcclAction
        elif action == "speed":
            return SpeedAction
        else:
            raise ValueError(f"Unknown action type {action}")


class LongitudinalAction:
    def __init__(self) -> None:
        self._longitudinal_action_type = None

    @abstractmethod
    def act(self, longitudinal_action: Any, **kwargs) -> float:
        raise NotImplementedError("longitudinal act method not implemented")

    @property
    def type(self) -> str:
        return self._longitudinal_action_type

    @property
    def space(self) -> gym.Space:
        return NotImplementedError(
            f"space method not implemented for longitudinal action type {self.type}"
        )
class SteerAction:
    def __init__(self) -> None:
        self._steer_action_type = None

    @abstractmethod
    def act(self, steer_action: Any, **kwargs) -> float:
        raise NotImplementedError("steer act method not implemented")

    @property
    def type(self) -> str:
        return self._steer_action_type

    @property
    def space(self) -> gym.Space:
        return NotImplementedError(
            f"space method not implemented for steer action type {self.type}"
        )
    
class CarAction:
    def __init__(self) -> None:
        self._steer_action = SteerAction()
        self._longitudinal_action = LongitudinalAction()

    @abstractmethod
    def act(self, action: Any, **kwargs) -> Tuple[float, float]:
        longitudinal_action = self._longitudinal_action.act(action[0], **kwargs)
        steer_action = self._steer_action.act(action[1], **kwargs)
        return steer_action, longitudinal_action

    @property
    def type(self) -> Tuple[str, str]:
        return (self._steer_action.type, self._longitudinal_action.type)

    @property
    def space(self) -> gym.Space:
        return NotImplementedError(
            f"space method not implemented for action type {self.type}"
        )


class AcclAction(CarAction):
    def __init__(self, params: Dict) -> None:
        super().__init__()
        self._steer_action_type = "speed"
        self._longitudinal_action_type = "accl"

        self.steering_low, self.steering_high = params["sv_min"], params["sv_max"]
        self.acc_low, self.acc_high = -params["a_max"], params["a_max"]

    def act(self, action: Tuple[float, float], state, params) -> Tuple[float, float]:
        return action

    @property
    def space(self) -> gym.Space:
        low = np.array([self.steering_low, self.acc_low]).astype(np.float32)
        high = np.array([self.steering_high, self.acc_high]).astype(np.float32)

        return gym.spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)


class SpeedAction(CarAction):
    def __init__(self, params: Dict) -> None:
        super().__init__()
        self._steer_action_type = "angle"
        self._longitudinal_action_type = "speed"

        self.steering_low, self.steering_high = params["s_min"], params["s_max"]
        self.velocity_low, self.velocity_high = params["v_min"], params["v_max"]

    def act(
        self, action: Tuple[float, float], state: np.ndarray, params: Dict
    ) -> Tuple[float, float]: # pid(speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v)
        accl = pid_accl(
            action[0],
            state[3],
            params["a_max"],
            params["v_max"],
            params["v_min"],
        )

        sv = pid_steer(
            action[1],
            state[2],
            params["sv_max"],
        )

        return accl, sv

    @property
    def space(self) -> gym.Space:
        low = np.array([self.steering_low, self.velocity_low]).astype(np.float32)
        high = np.array([self.steering_high, self.velocity_high]).astype(np.float32)

        return gym.spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)


def from_single_to_multi_action_space(
    single_agent_action_space: gym.spaces.Box, num_agents: int
) -> gym.spaces.Box:
    return gym.spaces.Box(
        low=single_agent_action_space.low[None].repeat(num_agents, 0),
        high=single_agent_action_space.high[None].repeat(num_agents, 0),
        shape=(num_agents, single_agent_action_space.shape[0]),
        dtype=np.float32,
    )