"""
Functions for converting between the following moisture variables:
* specific humidity, q
* mixing ratio, r
* vapour pressure, e
* relative humidity, RH
* dewpoint temperature, Td
* frost-point temperature, Tf
* saturation-point temperature, Ts

"""

import numpy as np
from atmos.constant import eps, cpd, cpv
from atmos.thermo import precision, max_n_iter
from atmos.thermo import latent_heat_of_vaporisation
from atmos.thermo import latent_heat_of_sublimation
from atmos.thermo import mixed_phase_latent_heat
from atmos.thermo import saturation_vapour_pressure
from atmos.thermo import saturation_specific_humidity
from atmos.thermo import saturation_mixing_ratio
from atmos.thermo import ice_fraction
from atmos.thermo import mixing_ratio as \
    mixing_ratio_from_specific_humidity
from atmos.thermo import vapour_pressure as \
    vapour_pressure_from_specific_humidity
from atmos.thermo import relative_humidity as \
    relative_humidity_from_specific_humidity
from atmos.thermo import _dewpoint_temperature_from_relative_humidity as \
    dewpoint_temperature_from_relative_humidity
from atmos.thermo import dewpoint_temperature as \
    dewpoint_temperature_from_specific_humidity
from atmos.thermo import _frost_point_temperature_from_relative_humidity as \
    frost_point_temperature_from_relative_humidity
from atmos.thermo import frost_point_temperature as \
    frost_point_temperature_from_specific_humidity
from atmos.thermo import _saturation_point_temperature_from_relative_humidity
from atmos.thermo import saturation_point_temperature as \
    saturation_point_temperature_from_specific_humidity
from atmos.thermo import wet_bulb_temperature as \
    wet_bulb_temperature_from_specific_humidity


def specific_humidity_from_mixing_ratio(r):
    """
    Computes specific humidity from water vapour mixing ratio.

    Args:
        r (float or ndarray): mixing ratio (kg/kg)

    Returns:
        q (float or ndarray): specific humidity (kg/kg)

    """
    q = r / (1 + r)

    return q


def specific_humidity_from_vapour_pressure(p, e):
    """
    Computes specific humidity from pressure and vapour pressure.

    Args:
        p (float or ndarray): pressure (Pa)
        e (float or ndarray): vapour pressure (Pa)

    Returns:
        q (float or ndarray): specific humidity (kg/kg)

    """
    q = eps * e / (p - (1 - eps) * e)

    return q


def specific_humidity_from_relative_humidity(p, T, RH, phase='liquid', 
                                             omega=0.0):
    """
    Computes specific humidity from pressure, temperature, and relative 
    humidity with respect to specified phase.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        RH (float or ndarray): relative humidity (fraction)
        phase (str, optional): condensed water phase (valid options are 
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation 
            (default is 0.0)
        
    Returns:
        q (float or ndarray): specific humidity (kg/kg)

    """
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    e = RH * es
    q = specific_humidity_from_vapour_pressure(p, e)

    return q


def specific_humidity_from_dewpoint_temperature(p, Td):
    """
    Computes specific humidity from pressure and dewpoint temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        Td (float or ndarray): dewpoint temperature (K)

    Returns:
        q (float or ndarray): specific humidity (kg/kg)

    """    
    q = saturation_specific_humidity(p, Td, phase='liquid')

    return q


def specific_humidity_from_frost_point_temperature(p, Tf):
    """
    Computes specific humidity from pressure and frost-point temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        Tf (float or ndarray): frost-point temperature (K)

    Returns:
        q (float or ndarray): specific humidity (kg/kg)

    """    
    q = saturation_specific_humidity(p, Tf, phase='ice')

    return q


def specific_humidity_from_saturation_point_temperature(p, Ts, omega):
    """
    Computes specific humidity from pressure, saturation-point temperature, and
    ice fraction at saturation.

    Args:
        p (float or ndarray): pressure (Pa)
        Ts (float or ndarray): saturation-point temperature (K)
        omega (float or ndarray): ice fraction at saturation

    Returns:
        q (float or ndarray): specific humidity (kg/kg)

    """    
    q = saturation_specific_humidity(p, Ts, phase='mixed', omega=omega)

    return q


def specific_humidity_from_wet_bulb_temperature(p, T, Tw, phase='liquid',
                                                omega=0.0):
    """
    Computes specific humidity from pressure and wet-bulb temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Tw (float or ndarray): wet-bulb temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        q (float or ndarray): specific humidity (kg/kg)

    """
    e = vapour_pressure_from_wet_bulb_temperature(p, T, Tw, phase=phase,
                                                  omega=omega)
    q = specific_humidity_from_vapour_pressure(p, e)

    return q


def mixing_ratio_from_vapour_pressure(p, e):
    """
    Computes mixing ratio from pressure and vapour pressure.

    Args:
        p (float or ndarray): pressure (Pa)
        e (float or ndarray): vapour pressure (Pa)

    Returns:
        r (float or ndarray): mixing ratio (kg/kg)

    """
    r = eps * e / (p - e)

    return r


def mixing_ratio_from_relative_humidity(p, T, RH, phase='liquid', omega=0.0):
    """
    Computes mixing ratio from pressure, temperature, and relative humidity
    with respect to specified phase.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        RH (float or ndarray): relative humidity (fraction)
        phase (str, optional): condensed water phase (valid options are 
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation 
            (default is 0.0)

    Returns:
        r (float or ndarray): mixing ratio (kg/kg)

    """
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    e = RH * es
    r = mixing_ratio_from_vapour_pressure(p, e)

    return r


def mixing_ratio_from_dewpoint_temperature(p, Td):
    """
    Computes mixing ratio from pressure and dewpoint temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        Td (float or ndarray): dewpoint temperature (K)

    Returns:
        r (float or ndarray): mixing ratio (kg/kg)

    """    
    r = saturation_mixing_ratio(p, Td, phase='liquid')

    return r


def mixing_ratio_from_frost_point_temperature(p, Tf):
    """
    Computes mixing ratio from pressure and frost-point temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        Tf (float or ndarray): frost-point temperature (K)

    Returns:
        r (float or ndarray): mixing ratio (kg/kg)

    """    
    r = saturation_mixing_ratio(p, Tf, phase='ice')

    return r


def mixing_ratio_from_saturation_point_temperature(p, Ts, omega):
    """
    Computes mixing ratio from pressure, saturation-point temperature, and ice
    fraction at saturation.

    Args:
        p (float or ndarray): pressure (Pa)
        Ts (float or ndarray): saturation-point temperature (K)
        omega (float or ndarray): ice fraction at saturation

    Returns:
        r (float or ndarray): mixing ratio (kg/kg)

    """    
    r = saturation_mixing_ratio(p, Ts, phase='mixed', omega=omega)

    return r


def mixing_ratio_from_wet_bulb_temperature(p, T, Tw, phase='liquid',
                                           omega=0.0):
    """
    Computes mixing ratio from pressure and wet-bulb temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Tw (float or ndarray): wet-bulb temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        r (float or ndarray): mixing ratio (kg/kg)

    """
    e = vapour_pressure_from_wet_bulb_temperature(p, T, Tw, phase=phase,
                                                  omega=omega)
    r = mixing_ratio_from_vapour_pressure(p, e)

    return r


def vapour_pressure_from_mixing_ratio(p, r):
    """
    Computes vapour pressure from pressure and mixing ratio.

    Args:
        p (float or ndarray): pressure (Pa)
        r (float or ndarray): mixing ratio (kg/kg)

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """
    e = p * r / (r + eps)

    return e


def vapour_pressure_from_relative_humidity(T, RH, phase='liquid', omega=0.0):
    """
    Computes vapour pressure from temperature and relative humidity with
    respect to specified phase.

    Args:
        T (float or ndarray): temperature (K)
        RH (float or ndarray): relative humidity (fraction)
        phase (str, optional): condensed water phase (valid options are 
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation 
            (default is 0.0)

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    e = RH * es

    return e


def vapour_pressure_from_dewpoint_temperature(Td):
    """
    Computes vapour pressure from dewpoint temperature.

    Args:
        Td (float or ndarray): dewpoint temperature (K)

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """
    e = saturation_vapour_pressure(Td, phase='liquid')

    return e


def vapour_pressure_from_frost_point_temperature(Tf):
    """
    Computes vapour pressure from frost-point temperature.

    Args:
        Tf (float or ndarray): frost-point temperature (K)

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """
    e = saturation_vapour_pressure(Tf, phase='ice')

    return e


def vapour_pressure_from_saturation_point_temperature(Ts, omega):
    """
    Computes vapour pressure from saturation-point temperature and ice fraction
    at saturation.

    Args:
        Ts (float or ndarray): saturation-point temperature (K)
        omega (float or ndarray): ice fraction at saturation

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """
    e = saturation_vapour_pressure(Ts, phase='mixed', omega=omega)

    return e


def vapour_pressure_from_wet_bulb_temperature(p, T, Tw, phase='liquid',
                                              omega=0.0):
    """
    Computes vapour pressure from pressure, temperature, and wet-bulb
    temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Tw (float or ndarray): wet-bulb temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """

    # Compute saturation vapour pressure at Tw
    es_Tw = saturation_vapour_pressure(Tw, phase=phase, omega=omega)

    if phase == 'liquid':

        # Compute latent heat of vaporisation at Tw
        Lv_Tw = latent_heat_of_vaporisation(Tw)

        # Compute vapour pressure
        e = p * (eps * Lv_Tw * es_Tw - cpd * (T - Tw) * (p - es_Tw)) / \
            (eps * Lv_Tw * p + (eps * cpv - cpd) * (T - Tw) * (p - es_Tw))

    if phase == 'ice':

        # Compute latent heat of sublimation at Tw
        Ls_Tw = latent_heat_of_sublimation(Tw)

        # Compute vapour pressure
        e = p * (eps * Ls_Tw * es_Tw - cpd * (T - Tw) * (p - es_Tw)) / \
            (eps * Ls_Tw * p + (eps * cpv - cpd) * (T - Tw) * (p - es_Tw))

    else:

        # Compute mixed-phase latent heat at Tw
        Lx_Tw = mixed_phase_latent_heat(Tw, omega)

        # Compute vapour pressure
        e = p * (eps * Lx_Tw * es_Tw - cpd * (T - Tw) * (p - es_Tw)) / \
            (eps * Lx_Tw * p + (eps * cpv - cpd) * (T - Tw) * (p - es_Tw))

    return e
    
    
def relative_humidity_from_mixing_ratio(p, T, r, phase='liquid', omega=0.0):
    """
    Computes relative humidity with respect to specified phase from pressure, 
    temperature, and mixing ratio.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        r (float or ndarray): mixing ratio (kg/kg)
        phase (optional): condensed water phase ('liquid', 'ice', or 'mixed')
        omega (float or ndarray, optional): ice fraction at saturation 
            (default is 0.0)

    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """
    e = vapour_pressure_from_mixing_ratio(p, r)
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    RH = e / es

    return RH
    
    
def relative_humidity_from_vapour_pressure(T, e, phase='liquid', omega=0.0):
    """
    Computes relative humidity with respect to specified phase from temperature
    and vapour pressure.

    Args:
        T (float or ndarray): temperature (K)
        e (float or ndarray): vapour pressure (Pa)
        phase (optional): condensed water phase ('liquid', 'ice', or 'mixed')
        omega (optional): ice fraction at saturation

    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    RH = e / es

    return RH
    
    
def relative_humidity_from_dewpoint_temperature(T, Td):
    """
    Computes relative humidity with respect to liquid water from temperature
    and dewpoint temperature.

    Args:
        T (float or ndarray): temperature (K)
        Td (float or ndarray): dewpoint temperature (K)

    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """
    e = saturation_vapour_pressure(Td, phase='liquid')
    es = saturation_vapour_pressure(T, phase='liquid')
    RH = e / es

    return RH
    
    
def relative_humidity_from_frost_point_temperature(T, Tf):
    """
    Computes relative humidity with respect to ice from temperature and frost-
    point temperature.

    Args:
        T (float or ndarray): temperature (K)
        Tf (float or ndarray): frost-point temperature (K)

    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """
    e = saturation_vapour_pressure(Tf, phase='ice')
    es = saturation_vapour_pressure(T, phase='ice')
    RH = e / es

    return RH


def relative_humidity_from_saturation_point_temperature(T, Ts, omega):
    """
    Computes mixed-phase relative humidity from temperature, saturation-point
    temperature, and ice fraction at saturation.

    Args:
        T (float or ndarray): temperature (K)
        Ts (float or ndarray): saturation-point temperature (K)
        omega (float or ndarray): ice fraction at saturation

    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """
    e = saturation_vapour_pressure(Ts, phase='mixed', omega=omega)
    es = saturation_vapour_pressure(T, phase='mixed', omega=omega)
    RH = e / es

    return RH


def relative_humidity_from_wet_bulb_temperature(p, T, Tw, phase='liquid',
                                                omega=0.0):
    """
    Computes relative humidity from pressure, temperature, and wet-bulb
    temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Tw (float or ndarray): wet-bulb temperature (K)
        phase (optional): condensed water phase ('liquid', 'ice', or 'mixed')
        omega (float or ndarray): ice fraction at saturation

    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """
    e = vapour_pressure_from_wet_bulb_temperature(p, T, Tw, phase=phase,
                                                  omega=omega)
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    RH = e / es

    return RH
    

def dewpoint_temperature_from_mixing_ratio(p, T, r):
    """
    Computes dewpoint temperature from pressure, temperature, and mixing ratio.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        r (float or ndarray): mixing ratio (kg/kg)

    Returns:
        Td (float or ndarray): dewpoint temperature (K)

    """
    RH = relative_humidity_from_mixing_ratio(p, T, r, phase='liquid')
    Td = dewpoint_temperature_from_relative_humidity(T, RH)

    return Td


def dewpoint_temperature_from_vapour_pressure(T, e):
    """
    Computes dewpoint temperature from temperature and vapour pressure.

    Args:
        T (float or ndarray): temperature (K)
        e (float or ndarray): vapour pressure (Pa)

    Returns:
        Td (float or ndarray): dewpoint temperature (K)

    """
    RH = relative_humidity_from_vapour_pressure(T, e, phase='liquid')
    Td = dewpoint_temperature_from_relative_humidity(T, RH)

    return Td


def dewpoint_temperature_from_frost_point_temperature(T, Tf):
    """
    Computes dewpoint temperature from temperature and frost-point temperature.

    Args:
        T (float or ndarray): temperature (K)
        Tf (float or ndarray): frost-point temperature (K)

    Returns:
        Td (float or ndarray): dewpoint temperature (K)

    """

    # Compute relative humidity over ice
    RHi = relative_humidity_from_frost_point_temperature(T, Tf)

    # Convert to relative humidity over liquid water
    RHl = convert_relative_humidity(T, RHi, phase_in='ice', phase_out='liquid')

    # Compute dewpoint temperature
    Td = dewpoint_temperature_from_relative_humidity(T, RHl)

    return Td


def dewpoint_temperature_from_saturation_point_temperature(T, Ts, omega):
    """
    Computes dewpoint temperature from temperature, saturation-point
    temperature, and ice fraction at saturation.

    Args:
        T (float or ndarray): temperature (K)
        Ts (float or ndarray): saturation-point temperature (K)
        omega (float or ndarray): ice fraction at saturation

    Returns:
        Td (float or ndarray): dewpoint temperature (K)

    """

    # Compute mixed-phase relative humidity
    RHx = relative_humidity_from_saturation_point_temperature(T, Ts, omega)

    # Convert to relative humidity over liquid water
    RHl = convert_relative_humidity(T, RHx, phase_in='mixed',
                                    phase_out='liquid', omega=omega)
    
    # Compute dewpoint temperature
    Td = dewpoint_temperature_from_relative_humidity(T, RHl)

    return Td


def dewpoint_temperature_from_wet_bulb_temperature(p, T, Tw):
    """
    Computes dewpoint temperature from pressure, temperature, and wet-bulb
    temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Tw (float or ndarray): wet-bulb temperature (K)

    Returns:
        Td (float or ndarray): dewpoint temperature (K)

    """
    RH = relative_humidity_from_wet_bulb_temperature(p, T, Tw, phase='liquid')
    Td = dewpoint_temperature_from_relative_humidity(T, RH)

    return Td


def frost_point_temperature_from_mixing_ratio(p, T, r):
    """
    Computes frost-point temperature from pressure, temperature, and mixing
    ratio.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        r (float or ndarray): mixing ratio (kg/kg)

    Returns:
        Tf (float or ndarray): frost-point temperature (K)

    """
    RH = relative_humidity_from_mixing_ratio(p, T, r, phase='ice')
    Tf = frost_point_temperature_from_relative_humidity(T, RH)

    return Tf


def frost_point_temperature_from_vapour_pressure(T, e):
    """
    Computes frost-point temperature from temperature and vapour pressure.

    Args:
        T (float or ndarray): temperature (K)
        e (float or ndarray): vapour pressure (Pa)

    Returns:
        Tf (float or ndarray): frost-point temperature (K)

    """
    RH = relative_humidity_from_vapour_pressure(T, e, phase='ice')
    Tf = frost_point_temperature_from_relative_humidity(T, RH)

    return Tf


def frost_point_temperature_from_dewpoint_temperature(T, Td):
    """
    Computes frost-point temperature from temperature and dewpoint temperature.

    Args:
        T (float or ndarray): temperature (K)
        Td (float or ndarray): dewpoint temperature (K)

    Returns:
        Tf (float or ndarray): frost-point temperature (K)

    """

    # Compute relative humidity over liquid water
    RHl = relative_humidity_from_dewpoint_temperature(T, Td)

    # Convert to relative humidity over ice
    RHi = convert_relative_humidity(T, RHl, phase_in='liquid', phase_out='ice')

    # Compute frost-point temperature
    Tf = frost_point_temperature_from_relative_humidity(T, RHi)

    return Tf


def frost_point_temperature_from_saturation_point_temperature(T, Ts, omega):
    """
    Computes frost-point temperature from temperature, saturation-point
    temperature, and ice fraction at saturation.

    Args:
        T (float or ndarray): temperature (K)
        Ts (float or ndarray): saturation-point temperature (K)
        omega (float or ndarray): ice fraction at saturation

    Returns:
        Tf (float or ndarray): frost-point temperature (K)

    """
    # Compute mixed-phase relative humidity
    RHx = relative_humidity_from_saturation_point_temperature(T, Ts, omega)

    # Convert to relative humidity over ice
    RHi = convert_relative_humidity(T, RHx, phase_in='mixed', phase_out='ice',
                                    omega=omega)
    
    # Compute frost-point temperature
    Tf = frost_point_temperature_from_relative_humidity(T, RHi)

    return Tf


def frost_point_temperature_from_wet_bulb_temperature(p, T, Tw):
    """
    Computes frost-point temperature from pressure, temperature, and wet-bulb
    temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Tw (float or ndarray): wet-bulb temperature (K)

    Returns:
        Tf (float or ndarray): frost-point temperature (K)

    """
    RH = relative_humidity_from_wet_bulb_temperature(p, T, Tw, phase='ice')
    Tf = frost_point_temperature_from_relative_humidity(T, RH)

    return Tf


def saturation_point_temperature_from_mixing_ratio(p, T, r):
    """
    Computes saturation-point temperature from pressure, temperature, and 
    mixing ratio.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        r (float or ndarray): mixing ratio (kg/kg)

    Returns:
        Ts (float or ndarray): saturation-point temperature (K)

    """

    # Intialise the saturation point temperature as the temperature
    Ts = T

    # Iterate to convergence
    converged = False
    count = 0
    while not converged:

        # Update the previous Ts value
        Ts_prev = Ts

        # Compute the ice fraction
        omega = ice_fraction(Ts)

        # Compute mixed-phase relative humidity
        RH = relative_humidity_from_mixing_ratio(p, T, r, phase='mixed',
                                                 omega=omega)

        # Compute saturation point temperature
        Ts = _saturation_point_temperature_from_relative_humidity(T, RH, omega)

        # Check if solution has converged
        if np.max(np.abs(Ts - Ts_prev)) < precision:
            converged = True
        else:
            count += 1
            if count == max_n_iter:
                print(f"Ts not converged after {max_n_iter} iterations")
                break

    return Ts


def saturation_point_temperature_from_vapour_pressure(T, e):
    """
    Computes saturation-point temperature from temperature and vapour pressure.

    Args:
        T (float or ndarray): temperature (K)
        e (float or ndarray): vapour pressure (Pa)

    Returns:
        Ts (float or ndarray): saturation-point temperature (K)

    """

    # Intialise the saturation point temperature as the temperature
    Ts = T

    # Iterate to convergence
    converged = False
    count = 0
    while not converged:

        # Update the previous Ts value
        Ts_prev = Ts

        # Compute the ice fraction
        omega = ice_fraction(Ts)

        # Compute mixed-phase relative humidity
        RH = relative_humidity_from_vapour_pressure(T, e, phase='mixed',
                                                    omega=omega)

        # Compute saturation point temperature
        Ts = _saturation_point_temperature_from_relative_humidity(T, RH, omega)

        # Check if solution has converged
        if np.max(np.abs(Ts - Ts_prev)) < precision:
            converged = True
        else:
            count += 1
            if count == max_n_iter:
                print(f"Ts not converged after {max_n_iter} iterations")
                break

    return Ts
    
    
def saturation_point_temperature_from_relative_humidity(T, RH):
    """
    Computes saturation-point temperature from temperature and mixed-phase
    relative humidity.

    Args:
        T (float or ndarray): temperature (K)
        RH (float or ndarray): relative humidity (fraction)
    
    Returns:
        Ts (float or ndarray): saturation-point temperature (K)

    """
    
    # Intialise the saturation point temperature as the temperature
    Ts = T

    # Iterate to convergence
    converged = False
    count = 0
    while not converged:

        # Update the previous Ts value
        Ts_prev = Ts

        # Compute the ice fraction
        omega = ice_fraction(Ts)

        # Compute saturation point temperature
        Ts = _saturation_point_temperature_from_relative_humidity(T, RH, omega)

        # Check if solution has converged
        if np.max(np.abs(Ts - Ts_prev)) < precision:
            converged = True
        else:
            count += 1
            if count == max_n_iter:
                print(f"Ts not converged after {max_n_iter} iterations")
                break

    return Ts


def saturation_point_temperature_from_dewpoint_temperature(T, Td):
    """
    Computes saturation-point temperature from temperature and dewpoint
    temperature.

    Args:
        T (float or ndarray): temperature (K)
        Td (float or ndarray): dewsaturation-point temperature (K)

    Returns:
        Ts (float or ndarray): saturation-point temperature (K)

    """

    # Compute relative humidity over liquid water
    RHl = relative_humidity_from_dewpoint_temperature(T, Td)

    # Intialise the saturation point temperature as the temperature
    Ts = T

    # Iterate to convergence
    converged = False
    count = 0
    while not converged:

        # Update the previous Ts value
        Ts_prev = Ts

        # Compute the ice fraction
        omega = ice_fraction(Ts)

        # Compute mixed-phase relative humidity
        RHx = convert_relative_humidity(T, RHl, phase_in='liquid', 
                                        phase_out='mixed', omega=omega)

        # Compute saturation point temperature
        Ts = _saturation_point_temperature_from_relative_humidity(T, RHx, omega)

        # Check if solution has converged
        if np.max(np.abs(Ts - Ts_prev)) < precision:
            converged = True
        else:
            count += 1
            if count == max_n_iter:
                print(f"Ts not converged after {max_n_iter} iterations")
                break

    return Ts


def saturation_point_temperature_from_frost_point_temperature(T, Tf):
    """
    Computes saturation-point temperature from temperature and frost-point
    temperature.

    Args:
        T (float or ndarray): temperature (K)
        Tf (float or ndarray): frost-point temperature (K)

    Returns:
        Ts (float or ndarray): saturation-point temperature (K)

    """

    # Compute relative humidity over ice
    RHi = relative_humidity_from_frost_point_temperature(T, Tf)

    # Intialise the saturation point temperature as the temperature
    Ts = T

    # Iterate to convergence
    converged = False
    count = 0
    while not converged:

        # Update the previous Ts value
        Ts_prev = Ts

        # Compute the ice fraction
        omega = ice_fraction(Ts)

        # Compute mixed-phase relative humidity
        RHx = convert_relative_humidity(T, RHi, phase_in='ice', 
                                        phase_out='mixed', omega=omega)

        # Compute saturation point temperature
        Ts = _saturation_point_temperature_from_relative_humidity(T, RHx, omega)

        # Check if solution has converged
        if np.max(np.abs(Ts - Ts_prev)) < precision:
            converged = True
        else:
            count += 1
            if count == max_n_iter:
                print(f"Ts not converged after {max_n_iter} iterations")
                break

    return Ts


def saturation_point_temperature_from_wet_bulb_temperature(p, T, Tw):
    """
    Computes saturation-point temperature from pressure, temperature, and
    wet-bulb temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Tw (float or ndarray): wet-bulb temperature (K)

    Returns:
        Ts (float or ndarray): saturation-point temperature (K)

    """

    # Compute ice fraction corresponding to wet-bulb temperature
    omega_Tw = ice_fraction(Tw)

    # Compute specific humidity from wet-bulb temperature
    q = specific_humidity_from_wet_bulb_temperature(p, T, Tw, phase='mixed',
                                                    omega=omega_Tw)

    # Intialise the saturation point temperature as the temperature
    Ts = T

    # Iterate to convergence
    converged = False
    count = 0
    while not converged:

        # Update the previous Ts value
        Ts_prev = Ts

        # Compute the ice fraction
        omega = ice_fraction(Ts)

        # Compute relative humidity from specific humidity
        RH = relative_humidity_from_specific_humidity(p, T, q, phase='mixed',
                                                      omega=omega)

        # Compute saturation point temperature
        Ts = _saturation_point_temperature_from_relative_humidity(T, RH, omega)

        # Check if solution has converged
        if np.max(np.abs(Ts - Ts_prev)) < precision:
            converged = True
        else:
            count += 1
            if count == max_n_iter:
                print(f"Ts not converged after {max_n_iter} iterations")
                break

    return Ts


def wet_bulb_temperature_from_mixing_ratio(p, T, r, phase='liquid',
                                           isobaric_method='Romps'):
    """
    Computes isobaric wet-bulb temperature from pressure, temperature, and
    mixing ratio.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        r (float or ndarray): mixing ratio (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        isobaric_method (str, optional): method used to calculate isobaric
            wet-bulb temperature (valid options are 'Warren' or 'Romps';
            default is 'Romps')

    Returns:
        Tw (float or ndarray): isobaric wet-bulb temperature (K)

    """
    q = specific_humidity_from_mixing_ratio(r)
    Tw = wet_bulb_temperature_from_specific_humidity(
        p, T, q, phase=phase, variant='isobaric',
        isobaric_method=isobaric_method
    )

    return Tw


def wet_bulb_temperature_from_vapour_pressure(p, T, e, phase='liquid',
                                              isobaric_method='Romps'):
    """
    Computes isobaric wet-bulb temperature from pressure, temperature, and
    vapour pressure.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        e (float or ndarray): vapour pressure (Pa)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        isobaric_method (str, optional): method used to calculate isobaric
            wet-bulb temperature (valid options are 'Warren' or 'Romps';
            default is 'Romps')

    Returns:
        Tw (float or ndarray): isobaric wet-bulb temperature (K)

    """
    q = specific_humidity_from_vapour_pressure(p, e)
    Tw = wet_bulb_temperature_from_specific_humidity(
        p, T, q, phase=phase, variant='isobaric',
        isobaric_method=isobaric_method
    )

    return Tw


def wet_bulb_temperature_from_relative_humidity(p, T, RH, phase='liquid',
                                                omega=0.0,
                                                isobaric_method='Romps'):
    """
    Computes isobaric wet-bulb temperature from pressure, temperature, and
    relative humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        RH (float or ndarray): relative humidity (fraction)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray): ice fraction at saturation
        isobaric_method (str, optional): method used to calculate isobaric
            wet-bulb temperature (valid options are 'Warren' or 'Romps';
            default is 'Romps')

    Returns:
        Tw (float or ndarray): isobaric wet-bulb temperature (K)

    """
    q = specific_humidity_from_relative_humidity(p, T, RH, phase=phase,
                                                 omega=omega)
    Tw = wet_bulb_temperature_from_specific_humidity(
        p, T, q, phase=phase, variant='isobaric',
        isobaric_method=isobaric_method
    )

    return Tw


def wet_bulb_temperature_from_dewpoint_temperature(p, T, Td,
                                                   isobaric_method='Romps'):
    """
    Computes isobaric wet-bulb temperature from pressure, temperature, and
    dewpoint temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Td (float or ndarray): dewpoint temperature (K)
        isobaric_method (str, optional): method used to calculate isobaric
            wet-bulb temperature (valid options are 'Warren' or 'Romps';
            default is 'Romps')

    Returns:
        Tw (float or ndarray): isobaric wet-bulb temperature (K)

    """
    q = specific_humidity_from_dewpoint_temperature(p, Td)
    Tw = wet_bulb_temperature_from_specific_humidity(
        p, T, q, phase='liquid', variant='isobaric',
        isobaric_method=isobaric_method
    )

    return Tw


def wet_bulb_temperature_from_frost_point_temperature(p, T, Tf,
                                                      isobaric_method='Romps'):
    """
    Computes isobaric wet-bulb temperature from pressure, temperature, and
    frost-point temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Tf (float or ndarray): frost-point temperature (K)
        isobaric_method (str, optional): method used to calculate isobaric
            wet-bulb temperature (valid options are 'Warren' or 'Romps';
            default is 'Romps')

    Returns:
        Tw (float or ndarray): isobaric wet-bulb temperature (K)

    """
    q = specific_humidity_from_frost_point_temperature(p, Tf)
    Tw = wet_bulb_temperature_from_specific_humidity(
        p, T, q, phase='ice', variant='isobaric',
        isobaric_method=isobaric_method
    )

    return Tw


def wet_bulb_temperature_from_saturation_point_temperature(p, T, Ts, omega,
                                                           isobaric_method='Romps'):
    """
    Computes isobaric wet-bulb temperature from pressure, temperature, and
    saturation-point temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Ts (float or ndarray): saturation-point temperature (K)
        omega (float or ndarray): ice fraction at saturation
        isobaric_method (str, optional): method used to calculate isobaric
            wet-bulb temperature (valid options are 'Warren' or 'Romps';
            default is 'Romps')

    Returns:
        Tw (float or ndarray): isobaric wet-bulb temperature (K)

    """
    q = specific_humidity_from_saturation_point_temperature(p, Ts, omega)
    Tw = wet_bulb_temperature_from_specific_humidity(
        p, T, q, phase='mixed', variant='isobaric',
        isobaric_method=isobaric_method
    )

    return Tw


def convert_relative_humidity(T, RH_in, phase_in, phase_out, omega=0.0):
    """
    Converts relative humidity with respect to one phase to relative humidity
    with respect to another phase.

    Args:
        T (float or ndarray): temperature (K)
        RH_in (float or ndarray): input relative humidity (fraction)
        phase_in (str): input condensed water phase (valid options are 
            'liquid', 'ice', or 'mixed')
        phase_out (str): output condensed water phase (valid options are 
            'liquid', 'ice', or 'mixed')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)
    
    Returns:
        RH_out (float or ndarray): output relative humidity (fraction)

    """
    es_in = saturation_vapour_pressure(T, phase=phase_in, omega=omega)
    es_out = saturation_vapour_pressure(T, phase=phase_out, omega=omega)

    RH_out = RH_in * es_in / es_out

    return RH_out
