from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


class PO:
    """
    Perturb & Observe algorithm
    """

    def __init__(
        self,
        dc_step: float,
        dv_index: Union[int, str] = "delta_voltage",
        dp_index: Union[int, str] = "delta_power",
    ):
        self.dc_step = dc_step
        self.dv_index = dv_index
        self.dp_index = dp_index

        self.dp_in_obs = False
        self.dv_in_obs = False

        if isinstance(self.dp_index, int):
            self.dp_in_obs = True
        if isinstance(self.dv_index, int):
            self.dv_in_obs = True

    def learn(self, total_timesteps: int = 1000):
        pass

    def predict(
        self,
        obs: np.ndarray,
        info: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[np.ndarray, Any]:
        """
        Get an action according to the observation

        Parameters:
            obs: observations from the environment
            info: additional info passed to the policy
        """
        if self.dp_in_obs:
            delta_p = obs[0][self.dp_index]
        else:
            delta_p = info.get(self.dp_index, 0.0)

        if self.dv_in_obs:
            delta_v = obs[0][self.dv_index]
        else:
            delta_v = info.get(self.dv_index, 0.0)

        if delta_p >= 0:
            if delta_v > 0:
                action = -self.dc_step
            else:
                action = self.dc_step
        else:
            if delta_v >= 0:
                action = self.dc_step
            else:
                action = -self.dc_step

        action = np.array([action])
        action = np.clip(action, -1.0, 1.0)

        return action, None

    def __str__(self):
        return "POPolicy"
