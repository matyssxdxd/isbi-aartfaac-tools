#!/usr/bin/env python3
import datetime as dt
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import argparse

from utils.read_data import read_subband


@dataclass(frozen=True)
class SubbandFit:
    path: Path
    delay_sec: float
    rate_sec_per_sec: float
    weight: float
    snr: float

CLOCK_RE = re.compile(
    r"^(?P<indent>\s*)clock_early\s*=\s*"
    r"(?P<valid>[^:]+?)\s*:\s*"
    r"(?P<offset>[-+0-9.eE]+)\s*(?P<offset_unit>\w+)\s*:\s*"
    r"(?P<epoch>[^:]+?)\s*:\s*"
    r"(?P<rate>[-+0-9.eE]+)(?:\s*(?P<rate_unit>[^;]+?))?\s*;"
)

def parse_clock_line(line):
    match = CLOCK_RE.match(line)
    if not match:
        raise ValueError(f"Could not parse clock_early line: {line.rstrip()}")
    offset = float(match.group("offset"))
    if match.group("offset_unit").strip() == "sec":
        offset *= 1e6
    rate = float(match.group("rate"))
    rate_unit = (match.group("rate_unit") or "usec/sec").strip()
    if rate_unit == "sec/sec":
        rate *= 1e6
    return {
        "indent": match.group("indent"),
        "valid": match.group("valid").strip(),
        "offset_usec": offset,
        "epoch": match.group("epoch").strip(),
        "rate_usec_per_sec": rate,
    }

def format_clock_line(clock, offset_usec, rate_usec_per_sec):
    return (
        f"{clock['indent']}clock_early = {clock['valid']}  : {offset_usec:.11f} usec :  "
        f"{clock['epoch']}  : {rate_usec_per_sec:.11e} usec/sec;\n"
    )

def update_vex_clock(vex_in, vex_out, target, residual_delay, residual_rate):
    lines = Path(vex_in).read_text(encoding="utf-8").splitlines(keepends=True)
    output = []
    in_clock = False
    in_target_def = False
    old_clock = None
    new_offset = None
    new_rate = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("$"):
            in_clock = stripped.startswith("$CLOCK")
            in_target_def = False
        if in_clock and stripped.startswith("def "):
            station = stripped.partition("def ")[2].partition(";")[0].strip()
            in_target_def = station == target
        if in_clock and in_target_def and stripped.startswith("clock_early"):
            old_clock = parse_clock_line(line)
            new_offset = old_clock["offset_usec"] - residual_delay * 1e6
            new_rate = old_clock["rate_usec_per_sec"] - residual_rate * 1e6
            output.append(format_clock_line(old_clock, new_offset, new_rate))
            output.append(
                "* Modified by clock_search_step_by_step.ipynb on "
                f"{dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            )
            output.append("* " + line)
            continue
        output.append(line)

    if old_clock is None:
        raise ValueError(f"Did not find clock_early for def {target}; in $CLOCK")

    Path(vex_out).write_text("".join(output), encoding="utf-8")
    return old_clock, new_offset, new_rate

def numeric_suffix(path):
    match = re.search(r"subband_(\d+)\.out$", Path(path).name)
    return int(match.group(1)) if match else 0

def weighted_linear_fit(x, y, weights):
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if np.count_nonzero(mask) < 2:
        raise ValueError("x")
    slope, intercept = np.polyfit(x[mask], y[mask], 1, w=np.sqrt(weights[mask]))
    return float(slope), float(intercept)

def integration_times_sec(headers):
    midpoint = np.asarray([(h.start_time + h.end_time) * 0.5 for h in headers], dtype=np.float64)
    step = float(np.median(np.diff(midpoint))) if midpoint.size > 1 else 1.0
    if abs(step) > 100.0:
        midpoint = midpoint / 1000.0
    return midpoint - midpoint[0]

def fringe_snr(spec, guard=3):
    lag = np.fft.fftshift(np.fft.ifft(spec))
    lag_abs = np.abs(lag)
    peak_idx = int(np.argmax(lag_abs))
    mask = np.ones(lag_abs.size, dtype=bool)
    mask[max(0, peak_idx - guard):min(lag_abs.size, peak_idx + guard + 1)] = False
    noise = lag[mask]
    noise_rms = float(np.sqrt(np.mean(np.abs(noise) ** 2))) if noise.size else 0.0
    return float(lag_abs[peak_idx] / noise_rms) if noise_rms > 0 else 0.0

def fit_subband(path, pol_indices):
    headers, visibilities = read_subband(str(path))
    if len(headers) == 0:
        raise ValueError(f"{path}: no integrations found")
    if headers[0].nr_receivers != 2:
        raise ValueError(f"{path}: expected exactly two receivers")

    # Select the cross baseline, conjugate it, and average selected polarizations.
    pol_stack = [np.conj(visibilities[:, :, BL_CROSS, px, py]) for px, py in pol_indices]
    cross = np.mean(pol_stack, axis=0)

    # Average integrations with the cross-baseline weights.
    integration_weights = np.asarray([h.weights[BL_CROSS] for h in headers], dtype=np.float64)
    spec = np.tensordot(integration_weights, cross, axes=(0, 0)) / np.sum(integration_weights)

    # Fit delay from unwrapped phase versus frequency.
    freqs = headers[0].first_channel_frequency + np.arange(headers[0].nr_channels) * headers[0].channel_bandwidth
    center_freq = float(np.mean(freqs))
    amp = np.abs(spec)
    phase = np.unwrap(np.angle(spec))
    delay_slope, _ = weighted_linear_fit(freqs - center_freq, phase, amp)
    delay_sec = delay_slope / (2.0 * np.pi)

    # Remove delay, then fit the remaining phase drift versus time.
    delay_corrected = cross * np.exp(-1j * 2.0 * np.pi * freqs[None, :] * delay_sec)
    channel_weights = np.abs(cross)
    time_series = np.sum(delay_corrected * channel_weights, axis=1) / np.sum(channel_weights, axis=1)
    time_sec = integration_times_sec(headers)
    time_phase = np.unwrap(np.angle(time_series))
    rate_slope, _ = weighted_linear_fit(time_sec, time_phase, integration_weights * np.abs(time_series))
    rate_sec_per_sec = rate_slope / (2.0 * np.pi * center_freq)

    snr = fringe_snr(spec)
    weight = float(np.sum(integration_weights) * snr * snr)
    return SubbandFit(Path(path), delay_sec, rate_sec_per_sec, weight, snr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='',
                    description='',
                    epilog='')
    parser.add_argument("out_files")
    parser.add_argument("scan_nr")
    parser.add_argument("vex_in")
    parser.add_argument("vex_out")
    args = parser.parse_args()

    result_directory = Path(args.out_files)
    scan = args.scan_nr
    vex_in = Path(args.vex_in)
    vex_out = Path(args.vex_out)

    reference_station = "IB"
    target_station = "IR"
    pols = ["RR", "LL"]

    BL_CROSS = 1
    POL_TO_INDEX = {
        "RR": (0, 0),
        "RL": (0, 1),
        "LR": (1, 0),
        "LL": (1, 1)
    }

    subband_files = sorted(result_directory.glob("subband_*.out"), key=numeric_suffix)

    print(f"Result directory: {result_directory}")
    print(f"Subbands found: {len(subband_files)}")
    
    pol_indices = []

    for pol in pols:
        pol_indices.append(POL_TO_INDEX[pol])

    fits = [fit_subband(path, pol_indices) for path in subband_files]

    for fit in fits:
        print(
            f"{fit.path.name}: delay={fit.delay_sec * 1e9:+.3f} ns, "
            f"rate={fit.rate_sec_per_sec * 1e6:+.6e} usec/sec, "
            f"snr={fit.snr:.2f}, weight={fit.weight:.6e}"
    )
    
    weights = np.asarray([fit.weight for fit in fits], dtype=np.float64)
    residual_delay = float(np.average([fit.delay_sec for fit in fits], weights=weights))
    residual_rate = float(np.average([fit.rate_sec_per_sec for fit in fits], weights=weights))

    print(f"Combined residual delay: {residual_delay * 1e9:+.3f} ns")
    print(f"Combined residual rate:  {residual_rate * 1e6:+.6e} usec/sec")

    old_clock, new_offset, new_rate = update_vex_clock(
    vex_in, vex_out, target_station, residual_delay, residual_rate
    )

    print(f"Old {target_station} offset: {old_clock['offset_usec']:.11f} usec")
    print(f"New {target_station} offset: {new_offset:.11f} usec")
    print(f"Old {target_station} rate:   {old_clock['rate_usec_per_sec']:.11e} usec/sec")
    print(f"New {target_station} rate:   {new_rate:.11e} usec/sec")
    print(f"Wrote: {vex_out}")
