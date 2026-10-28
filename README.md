# ISBI-AARTFAAC tools

Python scripts that prepare runs for the ISBI-AARTFAAC correlator (delay
files, run commands) and read its `.out` visibility files.

## Setup

Needs numpy, matplotlib, astropy, pycalc11, and for the bundled vex parser
`ply` and `multidict`. The vex parser lives in `vex/`; either install it or put
it on the path:

    PYTHONPATH=vex python aartfaac_run.py control.json

## Generating a run: `aartfaac_run.py`

    python aartfaac_run.py control.json [--conf-dir DIR]

Writes `<scan>.conf` (ISBI's `--delayFile`) and prints the `ISBI/ISBI` command
to run from the correlator's repository. Control-file keys are documented at
the top of `aartfaac_run.py`; `control.json` is an E011 example and
`data/n26l2_control.json` a three-station n26l2 one.

Things to know:

- **Station order.** One list orders `-i`, the delay tables and the channel
  mapping: the reference station first, then the others in `data-paths` order
  (or as given in `stations`). ISBI reads exactly one delay table per station,
  in `-i` order, and subtracts the first from the rest.
- **Channel mapping.** Taken from the vex `$THREADS`, pairing each band's two
  polarisations by their actual polarisation (`$BBC` -> `$IF`). A vex without
  `$THREADS` (n26l2 has only `$TRACKS`), or a recorder that does not write
  channels in the order its vex claims, needs `channel-mapping` in the control
  file. For n26l2 the vex's order is wrong and the mapping had to be measured.
- **Clocks.** The vex `clock_early` in force *at the scan* is used, not just
  the first one listed. `clock-offsets-usec` adds to it -- or, for a vex with
  no `$CLOCK`, is the whole clock (from a fringe search, e.g. `clock_search.py`).
- **Mixed sample rates.** `-f` is the slowest station's rate; faster stations
  get `--stationSampleRate` and are decimated by ISBI. The reference must be at
  the slowest rate, since its subbands define the frequency grid.
- `-t` is rounded down to a multiple of 8, which ISBI's tensor-core correlator
  requires; `--maxDelay` is derived from the delay table.

## Reading output

`read_data.py`, `process_data.py`, `plot_data.py`, `print_outfile.py` and
`clock_search.py` handle any number of stations. Baseline `(i, j)` is at index
`j*(j+1)/2 + i` (`utils.read_data.baseline_index`). Header weights are read
**signed**: the correlator writes a negative weight for an integration with no
valid samples, which an unsigned read turns into ~4.3e9 and lets dominate any
weighted average. `extract_weights` clips them to 0.
