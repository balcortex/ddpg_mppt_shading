from __future__ import annotations

import numbers
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src import utils
from src.pymlab import MATLABHandler

MATLAB_SOURCE_PATH = Path("src\matlab").resolve()
SIMULINK_MODEL_NAME = "pv_boost_avg_rload"
SIM_PARAMS = {
    "irradiance": [1000, 400, 300, 1000],
    "ambient_temperature": [40, 40, 40, 40],
    "capacitor_initial_voltage": 0,
    "dc_bus_voltage": 48,
    "rload_resistance": 80,
    "duty_cycle": 0,
    "stop_time": 0.1,
    "diode_state": [True] * 4,
}


class SimulinkModelOutput(NamedTuple):
    current: numbers.Real
    voltage: numbers.Real
    mod1_voltage: numbers.Real
    mod2_voltage: numbers.Real
    mod3_voltage: numbers.Real
    mod4_voltage: numbers.Real
    duty_cycle: numbers.Real

    def __str__(self):
        return ",".join(map(str, self))

    @classmethod
    def from_string(cls, string: str) -> SimulinkModelOutput:
        return cls(*map(float, string.split(",")))


class MPP(NamedTuple):
    current: numbers.Real
    voltage: numbers.Real
    power: numbers.Real
    duty_cycle: numbers.Real


class ShadedIVCurve(NamedTuple):
    current: Tuple[numbers.Real]
    voltage: Tuple[numbers.Real]
    module1: Tuple[numbers.Real]
    module2: Tuple[numbers.Real]
    module3: Tuple[numbers.Real]
    module4: Tuple[numbers.Real]
    duty_cycle: Tuple[numbers.Real]


class ShadedArray:
    """Photovoltaic System under partial shading

    Parameters:
        - mh: MATLAB handler
        - matlab_source_path: Path contaning the MATLAB model files
        - simulink_model_name: Name of the Simulink model
        - sim_params: Dictionary with the simulation paramaters.

            Example:

            sim_params = {
                "irradiance": [100, 400, 300, 200],
                "ambient_temperature": [40, 40, 40, 40],
                "capacitor_initial_voltage": 0,
                "dc_bus_voltage": 48,
                "rload_resistance": 80,
                "duty_cycle": 0,
                "stop_time": 1e-3,
                "diode_state": [True, True, True, True],
            }
    """

    def __init__(
        self,
        mh: MATLABHandler,
        matlab_source_path: Path,
        simulink_model_name: str,
        sim_params: Dict[str, Any],
    ):
        self.mh = mh
        # Get this variables from MATLAB's workspace after ruuning the simulink model
        self._vars = ("I_PV", "V_PV", "V_MOD1", "V_MOD2", "V_MOD3", "V_MOD4", "duty")
        # Variables to compose the key
        self._keys = ("irradiance", "ambient_temperature", "diode_state", "duty_cycle")

        # Initialize MATLAB
        self.mh.eval_args("cd", f"'{str(matlab_source_path)}'")
        self.mh.eval(f"model='{simulink_model_name}'")
        self.mh.eval_args("load_system", "model")
        self.sim_params = dict()
        self.set_params(sim_params)

        # self._last_cap_voltage = 0.0

        self.cache_path = matlab_source_path.joinpath(simulink_model_name + ".txt")
        if not self.cache_path.exists():
            self.cache_path.touch()
        self.cache = utils.read_dic_txt(self.cache_path)

    def set_params(self, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the simulation parameters with a Dictionary. If a Dict is not
        provided, update the paramaters in Simulink using the instance's dict.
        """
        params = params or self.sim_params
        for key, val in params.items():
            fn = self.__getattribute__("set_" + key)
            fn(val)

    def set_irradiance(self, irradiances: Sequence[numbers.Real]) -> None:
        """Set the modules' irradiance"""
        self.sim_params.update({"irradiance": tuple(irradiances)})
        command_template = "'pv_boost_avg_rload/Irradiance {}', 'Value', '{}'"
        for idx, g in enumerate(irradiances, start=1):
            self.mh.eval_args("set_param", command_template.format(idx, g))

    def set_ambient_temperature(
        self,
        temperatures: Sequence[numbers.Real],
        irradiances: Optional[Sequence[numbers.Real]] = None,
    ) -> None:
        """Provide the ambient temperature for each of the modules and estimate the modules' temperature based on the irradiance"""
        if irradiances:
            self.set_irradiance(irradiances)
        self.sim_params.update({"ambient_temperature": tuple(temperatures)})
        command_template = "'pv_boost_avg_rload/Temperature {}', 'Value', '{}'"
        for idx, t in enumerate(self.cell_t, start=1):
            self.mh.eval_args("set_param", command_template.format(idx, t))

    def set_capacitor_initial_voltage(self, voltage: numbers.Real) -> None:
        """Set the capacitor initial voltage"""
        self.sim_params.update({"capacitor_initial_voltage": voltage})
        self.mh.eval_args(
            "set_param", f"'pv_boost_avg_rload/C', 'InitialVoltage', '{voltage}'"
        )

    def set_dc_bus_voltage(self, voltage: numbers.Real) -> None:
        """Set the PV system's DC bus voltage"""
        self.sim_params.update({"dc_bus_voltage": voltage})
        self.mh.eval_args(
            "set_param", f"'pv_boost_avg_rload/DC Bus', 'Amplitude', '{voltage}'"
        )

    def set_rload_resistance(self, resistance: numbers.Real) -> None:
        """Set the value of the load resistance"""
        self.sim_params.update({"rload_resistance": resistance})
        self.mh.eval_args(
            "set_param", f"'pv_boost_avg_rload/R Load', 'Resistance', '{resistance}'"
        )

    def set_duty_cycle(self, duty_cycle: numbers.Real) -> None:
        """Set the duty cycle of the DC-DC converter"""
        self.sim_params.update({"duty_cycle": duty_cycle})
        self.mh.eval_args(
            "set_param", f"'pv_boost_avg_rload/Duty Cycle', 'Value', '{duty_cycle}'"
        )

    def set_stop_time(self, stop_time: Union[numbers.Real, str]) -> None:
        """Set the stop time of the simulation"""
        self.sim_params.update({"stop_time": stop_time})
        self.mh.eval_args(
            "set_param", f"'pv_boost_avg_rload', 'StopTime', '{stop_time}'"
        )

    def set_diode_state(self, states: Sequence[bool]) -> None:
        """Set the modules' diode state. If `False` the diode is omitted"""
        self.sim_params.update({"diode_state": tuple(states)})
        strings = ["on", "off"]
        command_template = "'pv_boost_avg_rload/PV System/Diode {}', 'Commented', '{}'"
        for idx, s in enumerate(states, start=1):
            self.mh.eval_args("set_param", command_template.format(idx, strings[s]))

    def run(self) -> SimulinkModelOutput:
        """Run the simulation and return the result"""
        self.mh.eval_args("sim", "model")
        while self.sim_status == "running":
            continue
        return self.simulation_output

    def simulate(
        self,
        duty_cycle: Optional[numbers.Real] = None,
        irradiance: Optional[Sequence[numbers.Real]] = None,
        ambient_temperature: Optional[Sequence[numbers.Real]] = None,
    ) -> SimulinkModelOutput:
        """Specify the simulation parameters and return the result"""
        if duty_cycle is not None:
            duty_cycle = min(max(duty_cycle, 0), 1)  # clip between [0, 1]
            duty_cycle = round(duty_cycle, 2)
            self.sim_params.update({"duty_cycle": duty_cycle})
        if irradiance:
            self.sim_params.update({"irradiance": tuple(irradiance)})
        if ambient_temperature:
            self.sim_params.update({"ambient_temperature": tuple(ambient_temperature)})

        if self.is_cached:
            return SimulinkModelOutput.from_string(self.cache[self.key])

        self.set_params()
        res = self.run()
        self._add_to_cache(res)
        self.save_cache_dic()

        return res

    def save_cache_dic(self, overwrite: bool = False) -> None:
        """Save the cache dictionary as a txt file

        Parameters:
            - overwrite: If `True` overwrite the file, else append to the file

        """
        utils.save_dic_txt(self.cache, self.cache_path, overwrite=overwrite)

    def get_shaded_iv_curve(
        self,
        duty_cycle: Optional[numbers.Real] = None,
        irradiance: Optional[Sequence[numbers.Real]] = None,
        ambient_temperature: Optional[Sequence[numbers.Real]] = None,
        curve_points: int = 100,
        verbose: bool = False,
    ) -> ShadedIVCurve:
        """Get the IV curve for the PV system and each of its modules"""
        # Update the parameters if given
        self.simulate(duty_cycle, irradiance, ambient_temperature)

        curves = list()
        for dc in tqdm(
            range(0, curve_points + 1)[::-1], desc="IV Curve", disable=not verbose
        ):
            res = self.simulate(duty_cycle=dc / curve_points)
            curves.append(res)

        # Tranpose to group by variable
        return ShadedIVCurve(*zip(*curves))

    def get_mpp(
        self,
        duty_cycle: Optional[numbers.Real] = None,
        irradiance: Optional[Sequence[numbers.Real]] = None,
        ambient_temperature: Optional[Sequence[numbers.Real]] = None,
        curve_points: int = 100,
        verbose: bool = False,
    ) -> SimulinkModelOutput:
        """Compute the MPP"""
        # Update the parameters if given
        self.simulate(duty_cycle, irradiance, ambient_temperature)

        curves = self.get_shaded_iv_curve(curve_points, verbose=verbose)
        power = self.power(curves.current, curves.voltage)
        idx = np.argmax(power)

        return MPP(
            current=curves.current[idx],
            voltage=curves.voltage[idx],
            power=power[idx],
            duty_cycle=curves.duty_cycle[idx],
        )

    @property
    def simulation_output(self) -> SimulinkModelOutput:
        """Return the output of the last simulation run"""
        # [-1][0]-> output is a list of lists, get last list and unpack it to a single val
        return SimulinkModelOutput(
            *(self.mh.eval(f"{var}", nargout=1)[-1][0] for var in self._vars)
        )

    @property
    def is_cached(self) -> bool:
        """Return `True` if the simulation has been computed before"""
        return self.key in self.cache.keys()

    @property
    def sim_status(self) -> str:
        """Return the status of the Simulink simulation"""
        return self.mh.eval_args(
            "get_param", "'pv_boost_avg_rload', 'SimulationStatus'", nargout=1
        )

    @property
    def key(self) -> str:
        """Compose a key for dictionary storing"""
        return ",".join(
            str(val) for (key, val) in self.sim_params.items() if key in self._keys
        )

    @property
    def g(self) -> Sequence[numbers.Real]:
        """Return the current values of the modules' irradiance"""
        return self.sim_params["irradiance"]

    @property
    def amb_t(self) -> Sequence[numbers.Real]:
        """Return the current ambient temperature for each module"""
        return self.sim_params["ambient_temperature"]

    @property
    def cell_t(self) -> Sequence[numbers.Real]:
        """Return the modules' temperature"""
        return self.cell_temp_from_ambient(self.g, self.amb_t)

    @property
    def num_modules(self) -> int:
        """"Number of PV Modules in the system"""
        return len(self.g)

    @staticmethod
    def power(
        voltage: Union[numbers.Real, Sequence[numbers.Real]],
        current: Union[numbers.Real, Sequence[numbers.Real]],
    ) -> np.ndarray:
        """Compute the electric power"""
        return np.array(voltage) * np.array(current)

    @staticmethod
    def cell_temp_from_ambient(
        irradiance: Union[numbers.Real, Sequence[numbers.Real]],
        ambient_temp: Union[numbers.Real, Sequence[numbers.Real]],
        decimals: int = 2,
    ) -> float:
        "Estimate cell temperature from ambient temperature and irradiance"
        noct = 45
        g_ref = 800

        return (
            np.array(ambient_temp) + (noct - 20) * (np.array(irradiance) / g_ref)
        ).round(decimals)

    @classmethod
    def get_default_array(cls) -> ShadedArray:
        return cls(
            mh=MATLABHandler("MATLAB42"),
            matlab_source_path=MATLAB_SOURCE_PATH,
            simulink_model_name=SIMULINK_MODEL_NAME,
            sim_params=SIM_PARAMS,
        )

    def _add_to_cache(self, output: SimulinkModelOutput) -> None:
        """Add the simulation result to the cache dictionary"""
        self.cache[self.key] = str(output)


if __name__ == "__main__":
    pvsyss = ShadedArray.get_default_array()
    pvsyss.simulate(
        duty_cycle=0.0,
        irradiance=[400, 400, 400, 1000],
        ambient_temperature=[20, 25, 25, 20],
    )

    mpp = pvsyss.get_mpp()
    curve = pvsyss.get_shaded_iv_curve(curve_points=100)
    plt.plot(curve.voltage, curve.current)
    plt.plot(mpp.voltage, mpp.current, "o")
    plt.show()
    plt.plot(curve.voltage, pvsyss.power(curve.voltage, curve.current))
    plt.plot(mpp.voltage, pvsyss.power(mpp.voltage, mpp.current), "o")
    plt.show()

    plt.plot(curve.duty_cycle, pvsyss.power(curve.voltage, curve.current))
    plt.show()
