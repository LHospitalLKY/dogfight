"""配置字段表：YAML 里的人读单位 ↔ 数据类的 SI 单位。

这里只做“字段名、单位换算、类型”的声明，不读取文件；
loader 依据本表解析，未知字段会直接报错而不是静默忽略。
"""
from __future__ import annotations

import math
from typing import Callable

Converter = Callable[[object], object]


class ConfigError(ValueError):
    """配置解析/校验错误。消息里始终带文件与字段路径。"""


def _float(value: object) -> float:
    return float(value)


def _int(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("bool 不能当整数")
    return int(value)


def _bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError("需要 true/false")
    return value


def _str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("需要字符串")
    return value


def _deg(value: object) -> float:
    return math.radians(float(value))


def _km(value: object) -> float:
    return float(value) * 1000.0


def _kn(value: object) -> float:
    return float(value) * 1000.0


def _deg_per_s(value: object) -> float:
    return math.radians(float(value))


# YAML 键 -> (数据类字段, 转换函数)
WEAPON_FIELDS: dict[str, tuple[str, Converter]] = {
    "name": ("name", _str),
    "mass_kg": ("mass", _float),
    "length_m": ("length", _float),
    "diameter_m": ("diameter", _float),
    "max_g": ("max_g", _float),
    "seeker_fov_deg": ("seeker_fov", _deg),
    "gimbal_fov_deg": ("gimbal_fov", _deg),
    "seeker_range_km": ("seeker_range", _km),
    "lock_time_s": ("lock_time", _float),
    "min_range_m": ("min_range", _float),
    "max_range_km": ("max_range", _km),
    "thrust_kn": ("thrust", _kn),
    "burn_time_s": ("burn_time", _float),
    "burnout_mass_kg": ("burnout_mass", _float),
    "cd0": ("cd0", _float),
    "cn_max": ("cn_max", _float),
    "warhead_damage": ("warhead_damage", _float),
    "lethal_radius_m": ("lethal_radius", _float),
    "pk_decay": ("pk_decay", _float),
    "cm_resist": ("cm_resist", _float),
    "induced_k": ("induced_k", _float),
    "life_s": ("life", _float),
    "cooldown_s": ("cooldown", _float),
    "pn_gain": ("pn_gain", _float),
    "guidance": ("guidance", _str),
    "loft": ("loft", _bool),
    "active_range_km": ("active_range", _km),
    "autopilot_tau_s": ("autopilot_tau", _float),
    "guidance_delay_s": ("guidance_delay", _float),
    "g_slew": ("g_slew", _float),
    "tvc_frac": ("tvc_frac", _float),
    "tvc_time_s": ("tvc_time", _float),
    "ignition_delay_s": ("ignition_delay", _float),
    "ignition_align_deg": ("ignition_align", _deg),
    "rcs_slew_deg_s": ("rcs_slew", _deg_per_s),
    "ignition_timeout_s": ("ignition_timeout", _float),
    "cd0_wave": ("cd0_wave", _float),
    "boost_thrust_kn": ("boost_thrust", _kn),
    "boost_time_s": ("boost_time", _float),
    "pulse2_thrust_kn": ("pulse2_thrust", _kn),
    "pulse2_time_s": ("pulse2_time", _float),
    "pulse2_gap_s": ("pulse2_gap", _float),
    "dogfight_reach_m": ("dogfight_reach", _float),
}

# 机体的机体/气动/动力/机炮参数
AIRCRAFT_FRAME_FIELDS: dict[str, tuple[str, Converter]] = {
    "base_mass_kg": ("base_mass", _float),
    "wing_area_m2": ("wing_area", _float),
    "cl_max": ("cl_max", _float),
    "n_struct_g": ("n_struct", _float),
    "thrust_max_kn": ("thrust_max", _kn),
    "thrust_mil_kn": ("thrust_mil", _kn),
    "cd0": ("cd0", _float),
    "aspect_ratio": ("aspect_ratio", _float),
    "oswald": ("oswald", _float),
    "hp_max": ("hp_max", _float),
    "gun_range_m": ("gun_range", _float),
    "gun_cone_deg": ("gun_cone", _deg),
    "gun_dps": ("gun_dps", _float),
    "ammo_max": ("ammo_max", _float),
    "rounds_per_sec": ("rounds_per_sec", _float),
    "turn_tau_s": ("turn_tau", _float),
    "cruise_mach": ("cruise_mach", _float),
    "fuel_capacity_kg": ("fuel_capacity", _float),
    "lapse_exp": ("lapse_exp", _float),
    "ram_r0": ("ram_r0", _float),
    "ram_r1": ("ram_r1", _float),
    "derate_z_m": ("derate_z", _float),
    "derate_H_m": ("derate_H", _float),
    "eta_install": ("eta_install", _float),
    "muzzle_speed": ("muzzle_speed", _float),
    "bullet_mass_kg": ("bullet_mass", _float),
    "bullet_diameter_m": ("bullet_diameter", _float),
    "gun_spread": ("gun_spread", _float),
    "length_m": ("length", _float),
    "cl_alpha": ("cl_alpha", _float),
    "cl0": ("cl0", _float),
    "alpha_stall_deg": ("alpha_stall", _deg),
    "cm_alpha": ("cm_alpha", _float),
    "cm_de": ("cm_de", _float),
    "cm_q": ("cm_q", _float),
    "cl_p": ("cl_p", _float),
    "cl_da": ("cl_da", _float),
    "cl_beta": ("cl_beta", _float),
    "cl_dr": ("cl_dr", _float),
    "cn_beta": ("cn_beta", _float),
    "cn_r": ("cn_r", _float),
    "cn_dr": ("cn_dr", _float),
    "cn_da": ("cn_da", _float),
    "cy_beta": ("cy_beta", _float),
    "tvc_deg": ("tvc", _deg),
    "tvc_alpha_max_deg": ("tvc_alpha_max", _deg),
    "tvc_cl_bonus": ("tvc_cl_bonus", _float),
    # 全动鸭翼（J-20）：几何、作动器与涡增升。
    "canard": ("canard", _bool),
    "canard_cl_bonus": ("canard_cl_bonus", _float),
    "canard_alpha_max_deg": ("canard_alpha_max", _deg),
    "canard_stall_shift_deg": ("canard_stall_shift", _deg),
    "canard_area_ratio": ("canard_area_ratio", _float),
    "canard_arm_ratio": ("canard_arm_ratio", _float),
    "canard_chord_m": ("canard_chord", _float),
    "canard_deflect_max_deg": ("canard_deflect_max", _deg),
    "canard_rate_max_deg_s": ("canard_rate_max", _deg_per_s),
    "canard_tau_s": ("canard_tau", _float),
    "canard_cl_alpha": ("canard_cl_alpha", _float),
    "canard_cl_max": ("canard_cl_max", _float),
    "canard_alpha_stall_deg": ("canard_alpha_stall", _deg),
    "canard_downwash": ("canard_downwash", _float),
    "canard_hinge_coeff": ("canard_hinge_coeff", _float),
    "canard_actuator_torque_knm": ("canard_actuator_torque", _kn),
    "canard_post_stall_alpha_deg": ("canard_post_stall_alpha", _deg),
    # 主翼后缘可动翼面（所有机型）与全动垂尾（J-20 / Su-57）。
    "elevon": ("elevon", _bool),
    "elevon_area_ratio": ("elevon_area_ratio", _float),
    "elevon_arm_ratio": ("elevon_arm_ratio", _float),
    "elevon_deflect_max_deg": ("elevon_deflect_max", _deg),
    "elevon_rate_max_deg_s": ("elevon_rate_max", _deg_per_s),
    "elevon_tau_s": ("elevon_tau", _float),
    "elevon_cl_alpha": ("elevon_cl_alpha", _float),
    "elevon_cl_max": ("elevon_cl_max", _float),
    "elevon_deflect_drag": ("elevon_deflect_drag", _float),
    "all_moving_fin": ("all_moving_fin", _bool),
    "fin_area_ratio": ("fin_area_ratio", _float),
    "fin_arm_ratio": ("fin_arm_ratio", _float),
    "fin_deflect_max_deg": ("fin_deflect_max", _deg),
    "fin_rate_max_deg_s": ("fin_rate_max", _deg_per_s),
    "fin_tau_s": ("fin_tau", _float),
    "fin_cn_alpha": ("fin_cn_alpha", _float),
}

# 机体的传感器/隐身参数
AIRCRAFT_SENSOR_FIELDS: dict[str, tuple[str, Converter]] = {
    "radar_range_km": ("radar_range", _km),
    "side_array": ("side_array", _bool),
    "lband": ("lband", _bool),
    "irst_range_km": ("irst_range", _km),
    "pod_irst_range_km": ("pod_irst_range", _km),
    "rcs": ("rcs", _float),
    "rcs_front": ("rcs_front", _float),
    "rcs_side": ("rcs_side", _float),
    "rcs_back": ("rcs_back", _float),
}


def apply_fields(
    raw: object, mapping: dict[str, tuple[str, Converter]], source: str, where: str,
    ignore: set[str] | frozenset[str] = frozenset(),
) -> dict:
    """按字段表转换一段 YAML 字典；未知键与类型错误都带定位信息。"""
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: {where} 必须是键值表")
    out: dict = {}
    for key, value in raw.items():
        if key in ignore:
            continue
        if key not in mapping:
            raise ConfigError(f"{source}: {where}.{key}: 未知字段")
        field, convert = mapping[key]
        try:
            out[field] = convert(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"{source}: {where}.{key}: 值 {value!r} 非法（{exc}）"
            ) from exc
    return out
