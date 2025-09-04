#!/bin/env python
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Matthias Auf der Maur
#
# This script fits a 1-, 2- or 3-diode model to a given solar
# cell IV curve, possibly extended also with a dark IV, and
# writes the spice netlist.
#
# ngspice is used as spice engine


import numpy as np
import matplotlib.pyplot as plt
import subprocess
import tempfile
import os
import argparse
from scipy.optimize import differential_evolution, minimize
import shutil

def load_jv_data(filename):
    try:
        data = np.loadtxt(filename, comments="#")
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {filename}: {e}")

    if data.shape[1] < 2:
        raise ValueError("Expected at least two columns for voltage and current.")

    voltage = data[:, 0]
    current = data[:, 1] * 1e-3  # convert from mA to A
    return voltage, current

def generate_spice_netlist(params, netlist_path, sweep_points, model='1-diode'):
    Vmin = min(sweep_points) - 0.05
    Vmax = max(sweep_points) + 0.05
    Vstep = 0.005

    with open(netlist_path, "w") as f:
        f.write(f"* {model} solar cell model with DC sweep\n")
        f.write("V1 N1 0 DC 0\n")

        if model == '1-diode':
            Iph, log_Is, n, Rs, inv_Rsh = params
            Is = 10**log_Is
            Rsh = 1.0 / inv_Rsh if inv_Rsh > 0 else 1e12

            f.write(f".model DModel D(IS={Is:.4e} N={n:.4f})\n")
            f.write(f"Iph N2 0 DC {-Iph:.5f}\n")
            f.write("D1 N2 0 DModel\n")
            f.write(f"Rs N1 N2 {Rs:.5f}\n")
            f.write(f"Rsh N2 0 {Rsh:.2f}\n")

        elif model == '2-diode':
            Iph, log_Is1, n1, log_Is2, n2, Rs, inv_Rsh = params
            Is1, Is2 = 10**log_Is1, 10**log_Is2
            Rsh = 1.0 / inv_Rsh if inv_Rsh > 0 else 1e12

            f.write(f".model D1Model D(IS={Is1:.4e} N={n1:.4f})\n")
            f.write(f".model D2Model D(IS={Is2:.4e} N={n2:.4f})\n")
            f.write(f"Iph N2 0 DC {-Iph:.5f}\n")
            f.write("D1 N2 0 D1Model\n")
            f.write("D2 N2 0 D2Model\n")
            f.write(f"Rs N1 N2 {Rs:.5f}\n")
            f.write(f"Rsh N2 0 {Rsh:.2f}\n")

        elif model == '3-diode':
            Iph, log_Is1, n1, log_Is2, n2, log_Is3, n3, Rs, inv_Rsh = params
            Is1, Is2, Is3 = 10**log_Is1, 10**log_Is2, 10**log_Is3
            Rsh = 1.0 / inv_Rsh if inv_Rsh > 0 else 1e12

            f.write(f".model D1Model D(IS={Is1:.4e} N={n1:.4f})\n")
            f.write(f".model D2Model D(IS={Is2:.4e} N={n2:.4f})\n")
            f.write(f".model D3Model D(IS={Is3:.4e} N={n3:.4f})\n")
            f.write(f"Iph N2 0 DC {-Iph:.5f}\n")
            f.write("D1 N2 0 D1Model\n")
            f.write("D2 N2 0 D2Model\n")
            f.write("D3 N2 0 D3Model\n")
            f.write(f"Rs N1 N2 {Rs:.5f}\n")
            f.write(f"Rsh N2 0 {Rsh:.2f}\n")

        f.write(".control\n")
        f.write(f"dc V1 {Vmin:.3f} {Vmax:.3f} {Vstep:.3f}\n")
        f.write("wrdata sim_output.txt i(V1)\n")
        f.write("quit\n")
        f.write(".endc\n")
        f.write(".end\n")

def run_spice(netlist_path):
    subprocess.run(["ngspice", "-b", netlist_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sim_data = np.loadtxt("sim_output.txt", comments="#")
    Vsim = sim_data[:, 0]
    Isim = -sim_data[:, 1]  # flip sign to match experimental convention
    return Vsim, Isim

def interpolate_simulation(Vsim, Isim, Vexp):
    return np.interp(Vexp, Vsim, Isim)

def objective_function(params, Vexp, Iexp, verbose=False, model='1-diode'):
    # Expect Vexp and Iexp to have 'light' and 'dark' attributes for separation
    if hasattr(Vexp, 'light') and hasattr(Vexp, 'dark') and hasattr(Iexp, 'light') and hasattr(Iexp, 'dark'):
        V_light, I_light = Vexp.light, Iexp.light
        V_dark, I_dark = Vexp.dark, Iexp.dark
        fit_light = len(V_light) > 0
        fit_dark = len(V_dark) > 0
    else:
        raise ValueError("Vexp and Iexp must have 'light' and 'dark' attributes.")

    # Run light JV simulation
    if fit_light:
        with tempfile.NamedTemporaryFile(delete=False, suffix="_light.cir") as netlist_light:
            light_netlist_path = netlist_light.name
        try:
            generate_spice_netlist(params, light_netlist_path, V_light, model=model)
            Vsim_light, Isim_light = run_spice(light_netlist_path)
            Isim_interp_light = interpolate_simulation(Vsim_light, Isim_light, V_light)
            error_light = np.mean((I_light - Isim_interp_light)**2)
        except Exception as e:
            print(f"Light simulation failed: {e}")
            error_light = 1e6
        finally:
            os.remove(light_netlist_path)
            if os.path.exists("sim_output.txt"):
                os.remove("sim_output.txt")
    else:
        error_light = 0

    # Run dark JV simulation (with Iph set to 0)
    if fit_dark:
        dark_params = params.copy()
        dark_params[0] = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix="_dark.cir") as netlist_dark:
            dark_netlist_path = netlist_dark.name
        try:
            generate_spice_netlist(dark_params, dark_netlist_path, V_dark, model=model)
            Vsim_dark, Isim_dark = run_spice(dark_netlist_path)
            Isim_interp_dark = interpolate_simulation(Vsim_dark, Isim_dark, V_dark)
            error_dark = np.mean((np.log10(np.abs(I_dark) + 1e-20) - np.log10(np.abs(Isim_interp_dark) + 1e-20))**2)
        except Exception as e:
            print(f"Dark simulation failed: {e}")
            error_dark = 1e6
        finally:
            os.remove(dark_netlist_path)
            if os.path.exists("sim_output.txt"):
                os.remove("sim_output.txt")
    else:
        error_dark = 0

    error = error_light + 0.5*error_dark
    
    return error

def fit_solar_cell(lightfile=None, darkfile=None, model='1-diode', plot=False, verbose=False, no_shunt=False):
    if lightfile:
        Vexp, Iexp = load_jv_data(lightfile)
    else:
        Vexp, Iexp = np.array([]), np.array([])

    if darkfile:
        Vdark_exp, Idark_exp = load_jv_data(darkfile)

        class ExpData:
            pass
        Vexp_all = ExpData()
        Iexp_all = ExpData()
        Vexp_all.light = Vexp
        Vexp_all.dark = Vdark_exp
        Iexp_all.light = Iexp
        Iexp_all.dark = Idark_exp
    else:
        class ExpData:
            pass
        Vexp_all = ExpData()
        Iexp_all = ExpData()
        Vexp_all.light = Vexp
        Vexp_all.dark = np.array([])
        Iexp_all.light = Iexp
        Iexp_all.dark = np.array([])

    if lightfile and len(Vexp) > 1:
        Isc_index = np.argmin(np.abs(Vexp))
        Isc = -Iexp[Isc_index]  # Current closest to 0 V

        zero_crossings = np.where(np.diff(np.sign(Iexp)))[0]
        if len(zero_crossings) > 0:
            i = zero_crossings[0]
            Voc = Vexp[i] - Iexp[i] * (Vexp[i+1] - Vexp[i]) / (Iexp[i+1] - Iexp[i])
            Voc_region = Vexp[(Vexp > 0.9 * Voc) & (Vexp <= Voc)]
            Jvoc_region = Iexp[(Vexp > 0.9 * Voc) & (Vexp <= Voc)]
        else:
            Voc = None
            Voc_region = Vexp[-2:]
            Jvoc_region = Iexp[-2:]
        
        Rs_estimate = np.abs((Voc_region[-1] - Voc_region[0]) / (Jvoc_region[-1] - Jvoc_region[0]))


        lowV_region = Vexp[(Vexp > 0) & (Vexp < 0.1)]
        lowI_region = Iexp[(Vexp > 0) & (Vexp < 0.1)]
        if len(lowV_region) >= 2:
            Rsh_estimate = np.abs((lowV_region[-1] - lowV_region[0]) / (lowI_region[-1] - lowI_region[0]))
        else:
            Rsh_estimate = 1e12 if no_shunt else 1e6  # fallback
    else:
        Isc = 0
        Rs_estimate = 0.01
        Rsh_estimate = 1e6
        Voc_region = np.array([])
        Jvoc_region = np.array([])
    
        

    if model == '1-diode':
        bounds = [(0.5 * Isc, 1.5 * Isc),
                  (np.log10(1e-16), np.log10(1e-6)),
                  (0.5, 4),
                  (0.5 * Rs_estimate, 2.0 * Rs_estimate),
                  (1 / (5 * Rsh_estimate), 5 / Rsh_estimate) if not no_shunt else (1e-12, 1e-12)]
        x0 = [Isc, -14, 1, Rs_estimate, 1/Rsh_estimate]
    elif model == '2-diode':
        bounds = [(0.5 * Isc, 1.5 * Isc),
                  (np.log10(1e-16), np.log10(1e-6)), (0.5, 3),
                  (np.log10(1e-16), np.log10(1e-5)), (2, 5),
                  (0.5 * Rs_estimate, 2.0 * Rs_estimate),
                  (1 / (5 * Rsh_estimate), 5 / Rsh_estimate) if not no_shunt else (1e-12, 1e-12)]
        x0 = [Isc, -14, 1, -13, 2, Rs_estimate, 1/Rsh_estimate]
    elif model == '3-diode':
        bounds = [(0.5 * Isc, 1.5 * Isc),
                  (np.log10(1e-15), np.log10(1e-6)), (0.5, 1.5),
                  (np.log10(1e-15), np.log10(1e-6)), (1.5, 4),
                  (np.log10(1e-15), np.log10(1e-6)), (4, 10),
                  (0.5 * Rs_estimate, 2.0 * Rs_estimate),
                  (1 / (5 * Rsh_estimate), 5 / Rsh_estimate) if not no_shunt else (1e-12, 1e-12)]
        x0 = [Isc, -14, 1, -13, 2, -13, 4, Rs_estimate, 1/Rsh_estimate]
    else:
        raise ValueError(f"Unsupported model: {model}")

    result = differential_evolution(
        lambda p: objective_function(p, Vexp_all, Iexp_all, verbose, model=model),
        bounds=bounds,
        strategy='randtobest1bin',
        maxiter=10,
        popsize=15,
        tol=1e-6,
        polish=False,
        disp=verbose,
        updating='deferred',
        callback=print_progress,
        workers=1
    )

    x0 = result.x
    result = minimize(objective_function, x0, args=(Vexp_all, Iexp_all, verbose, model), method='Powell', bounds=bounds)
    raw_params = result.x

    if lightfile and len(Vexp) > 0:
        sweep_points = Vexp
    elif darkfile and len(Vdark_exp) > 0:
        sweep_points = Vdark_exp
    else:
        raise ValueError("No valid sweep points found for simulation.")

    # Rescale fitted parameters to physical values for output
    if model == '1-diode':
        Iph, log_Is, n, Rs, inv_Rsh = raw_params
        fitted_params = [Iph, 10**log_Is, n, Rs, 1.0 / inv_Rsh if inv_Rsh > 0 else 1e12]
    elif model == '2-diode':
        Iph, log_Is1, n1, log_Is2, n2, Rs, inv_Rsh = raw_params
        fitted_params = [Iph, 10**log_Is1, n1, 10**log_Is2, n2, Rs, 1.0 / inv_Rsh if inv_Rsh > 0 else 1e12]
    elif model == '3-diode':
        Iph, log_Is1, n1, log_Is2, n2, log_Is3, n3, Rs, inv_Rsh = raw_params
        fitted_params = [Iph, 10**log_Is1, n1, 10**log_Is2, n2, 10**log_Is3, n3, Rs, 1.0 / inv_Rsh if inv_Rsh > 0 else 1e12]
    else:
        fitted_params = raw_params  # fallback in case of unexpected model

    print("Fitted parameters:")
    print(fitted_params)

    with open("fit_parameters.txt", "a") as f:
        label = os.path.basename(lightfile) if lightfile else os.path.basename(darkfile)
        f.write(f"{label}	" + "\t".join(f"{x:.5e}" for x in fitted_params) + "\n")

    if plot or verbose:
        generate_spice_netlist(raw_params, "final_fit.cir", sweep_points, model=model)
        Vsim, Isim = run_spice("final_fit.cir")
        os.rename("sim_output.txt", "final_fit.dat")

        # Generate dark JV curve (with Iph set to 0)
        dark_params = raw_params.copy()
        dark_params[0] = 0
        generate_spice_netlist(dark_params, "dark_fit.cir", sweep_points, model=model)
        Vdark, Idark = run_spice("dark_fit.cir")
        os.rename("sim_output.txt", "dark_fit.dat")

        # Create side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # JV plot
        ax1.plot(Vexp, Iexp, "o", label="Light Experimental", markersize=4)
        ax1.plot(Vsim, Isim, "-", label="Fitted Model")
        ax1.set_xlabel("Voltage (V)")
        ax1.set_ylabel("Current (A)")
        ax1.set_title(f"JV Fit ({model})")
        ax1.grid(True)
        ax1.legend()

        # Dark current plot
        if darkfile:
            ax2.semilogy(Vdark_exp, np.abs(Idark_exp), 'o', label="Dark Experimental")
        ax2.semilogy(Vdark, np.abs(Idark), label="Dark Model")
        ax2.set_xlabel("Voltage (V)")
        ax2.set_ylabel("|Current| (A)")
        ax2.set_title(f"Dark Current ({model})")
        ax2.set_ylim(1e-14, 1e-1)
        if darkfile:
            ax2.set_ylim(max(min(np.abs(Idark_exp)), 1e-9), max(np.abs(Idark_exp)))
        ax2.grid(True, which='both', ls='--')
        ax2.legend()

        fig.tight_layout()

        # Ideality factor analysis in dark JV region
        if darkfile:
            from scipy.stats import linregress
            mask = (Vdark > 0.1) & (np.abs(Idark) > 1e-10)
            if np.any(mask):
                log_I = np.log(np.abs(Idark[mask]))
                slope, intercept, r, p, stderr = linregress(Vdark[mask], log_I)
                n_eff = 1 / (slope * 0.02585)  # ideality factor assuming T=300K
                ax2.text(0.05, 0.95, f"n_eff ≈ {n_eff:.2f}", transform=ax2.transAxes, va='top')

        fig.savefig("fit_result.png", dpi=150)

        if plot:
            plt.show()

        

            

def print_progress(xk, convergence):
    print(f"Current params: {xk}, convergence: {convergence:.4e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit solar cell JV data using ngspice simulation.")
    parser.add_argument("--no-shunt", action="store_true", help="Disable shunt resistance in the model")
    parser.add_argument("--light", help="Optional light JV data file")
    parser.add_argument("--dark", help="Optional dark JV data file")
    parser.add_argument("--model", choices=["1-diode", "2-diode", "3-diode"], default="1-diode", help="Equivalent circuit model to fit")
    parser.add_argument("--plot", action="store_true", help="Plot the result")
    parser.add_argument("--verbose", action="store_true", help="Verbose output during fitting")
    args = parser.parse_args()

    fit_solar_cell(lightfile=args.light, darkfile=args.dark, model=args.model, plot=args.plot, verbose=args.verbose, no_shunt=args.no_shunt)

