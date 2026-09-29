# Vendored copy of rocprofwrap_lt

These files are copied from the `rocprofwrap_lt/` folder of
[hal-uw/rocprofwrap](https://github.com/hal-uw/rocprofwrap). They are bundled
here so the power run (`scripts/run_power.sbatch`) works from a plain clone of
this repo.

| | |
|---|---|
| Upstream repo | https://github.com/hal-uw/rocprofwrap |
| Base commit | `04a06aa` (branch `yiwei/upgrade`). `rocprofwrap_lt/` is identical to `main` at `112d9b7`. |
| Local changes | `power_query.cpp` and `README.md` add `edge_temp_C` and `hotspot_temp_C` columns (best-effort; `NaN` when a sensor is not exposed). The diff against the base commit is in `local-changes.patch`. These changes are not upstream yet. |
| Not included | The `amd-smi-query` binary. Build it for your ROCm (below). |

`scripts/build_sampling_json.py` needs `hotspot_temp_C` for the
`hotspot_temp` vector. With the unmodified upstream sampler, that vector is
skipped (the power and frequency vectors are unaffected).

## Build (once per checkout)

```bash
cd third_party/rocprofwrap_lt
module load rocm/7.2.0
make ROCM_DIR=/opt/rocm-7.2.0      # the Makefile's default is /opt/rocm-7.1.0
```

This produces `amd-smi-query` next to `wrapper.py`, which is where
`wrapper.py` looks for it. Point `PROF_LT` in your `workload.conf` at this
folder.

## Updating

If upstream changes, copy `wrapper.py`, `power_query.cpp`, `Makefile` and
`README.md` from the new commit, re-apply `local-changes.patch` if it is
still needed, and update the base commit above.
