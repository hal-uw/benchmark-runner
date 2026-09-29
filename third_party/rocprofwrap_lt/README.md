# rocprofwrap_lt

Lightweight GPU power sampling wrapper for ROCm 7.0 systems with `amd-smi` API.

This tool runs a target application and records per-GPU power/clock samples in parallel using AMD SMI (`amdsmi`).

## Files

- `power_query.cpp`: C++ sampler binary (`amd-smi-query`)
- `wrapper.py`: Python launcher that starts/stops samplers with your workload
- `Makefile`: builds `amd-smi-query`

## What It Does

- Launches one sampler process per selected GPU
- Launches your application command
- Stops all samplers automatically when the application exits
- Writes CSV files like `profiling_result_<prefix>_<device>.csv`

## Requirements

- ROCm installed (default path in `Makefile`: `/opt/rocm-7.1.0`)
- AMD SMI runtime/library (`libamd_smi`)
- C++ compiler (`g++` or compatible)
- Python 3

## Build

From `rocprofwrap_lt/`:

```bash
make
```

This builds:

```bash
./amd-smi-query
```

If your ROCm installation is in a different location:

```bash
make ROCM_DIR=/opt/rocm
```

## Usage

Basic form:

```bash
python3 wrapper.py -d "<device_ids>" -p <prefix> -- <your_command>
```

Example:

```bash
python3 wrapper.py -d "0,1" -p run1 -- python3 gemm.py
```

## Wrapper Options (`wrapper.py`)

- `-d`, `--devices`: Comma-separated GPU device IDs (required), e.g. `"0"` or `"0,1,2"`
- `-p`, `--prefix`: Prefix used in output filenames (required)
- `--query`: Path to `amd-smi-query` binary (optional). Default: `./amd-smi-query` next to `wrapper.py`
- `--interval-ms`: Sampling interval in milliseconds (optional, default: `1`)
- `--`: Separator before the target application command (required)

## Output

For each selected device, the wrapper writes:

```bash
profiling_result_<prefix>_<device_id>.csv
```

Example:

```bash
profiling_result_run1_0.csv
profiling_result_run1_1.csv
```

CSV content includes:

- `timestamp_ns`
- `current_socket_power_W`
- `inst_power_W`
- `gfx_clock_MHz`
- `edge_temp_C`
- `hotspot_temp_C`

The sampler also prints one metadata line before the CSV header (device id, interval, energy counter resolution).

## Principle (How It Works)

### 1. Power sampler (`power_query.cpp`)

The C++ program uses AMD SMI to:

- Initialize AMD GPU management (`amdsmi_init`)
- Enumerate GPU handles
- Select one GPU by device index (`-d`)
- Read energy counters (`amdsmi_get_energy_count`)
- Read socket power (`amdsmi_get_power_info`)
- Read GFX clock frequency (`amdsmi_get_clk_freq`)
- Read edge and hotspot/junction temperature (`amdsmi_get_temp_metric`, best-effort: unsupported sensors fall back to NaN rather than aborting the sampler)
- Output samples as CSV at a fixed interval (`-i`)

It computes `inst_power_W` from the energy counter delta:

- `delta_energy` from consecutive energy counter readings
- `delta_time` from consecutive timestamps
- `power = delta_energy / delta_time`

This gives an instantaneous power estimate based on the energy counter, while `current_socket_power_W` is the direct SMI-reported value.

### 2. Wrapper (`wrapper.py`)

The Python wrapper:

- Parses the requested GPU IDs
- Starts one `amd-smi-query` process per GPU
- Redirects each sampler output to its own CSV file
- Starts your target application
- Waits for the application to finish
- Sends `SIGINT` to samplers and waits for clean shutdown
- Truncates any partial last CSV line (if a sampler was interrupted mid-write)

This design keeps the sampler simple and lets you profile any command without modifying the application.

## Notes

- Only tested on AMD HPCFund rocm-7.1.0

