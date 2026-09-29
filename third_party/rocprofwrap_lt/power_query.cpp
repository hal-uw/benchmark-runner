#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <csignal>
#include <atomic>
#include <thread>
#include <vector>

#include "amd_smi/amdsmi.h"

namespace {

std::atomic<bool> g_stop_requested{false};

void HandleSignal(int) {
  g_stop_requested.store(true);
}

struct Options {
  int device_id = 0;
  int interval_ms = 1;
  int samples = -1;
};

void PrintUsage(const char *prog) {
  std::cout
      << "Usage: " << prog
      << " [-d device_id] [-i interval_ms] [-n samples]\n"
      << "  -d device_id     GPU device index (default 0)\n"
      << "  -i interval_ms   sampling interval in ms (default 1)\n"
      << "  -n samples       number of samples, -1 for infinite (default)\n";
}

bool ParseArgs(int argc, char **argv, Options *opt) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
      opt->device_id = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
      opt->interval_ms = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      opt->samples = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-h") == 0 ||
               std::strcmp(argv[i], "--help") == 0) {
      PrintUsage(argv[0]);
      return false;
    } else {
      std::cerr << "Unknown option: " << argv[i] << "\n";
      PrintUsage(argv[0]);
      return false;
    }
  }
  if (opt->device_id < 0 || opt->interval_ms <= 0) {
    std::cerr << "Invalid arguments.\n";
    return false;
  }
  return true;
}

bool GetGpuHandles(std::vector<amdsmi_processor_handle> *out) {
  uint32_t socket_count = 0;
  amdsmi_status_t ret = amdsmi_get_socket_handles(&socket_count, nullptr);
  if (ret != AMDSMI_STATUS_SUCCESS || socket_count == 0) {
    std::cerr << "Failed to get socket handles, ret=" << ret << "\n";
    return false;
  }

  std::vector<amdsmi_socket_handle> sockets(socket_count);
  ret = amdsmi_get_socket_handles(&socket_count, sockets.data());
  if (ret != AMDSMI_STATUS_SUCCESS) {
    std::cerr << "Failed to get socket handles, ret=" << ret << "\n";
    return false;
  }

  for (uint32_t i = 0; i < socket_count; ++i) {
    uint32_t dev_count = 0;
    ret = amdsmi_get_processor_handles(sockets[i], &dev_count, nullptr);
    if (ret != AMDSMI_STATUS_SUCCESS || dev_count == 0) {
      continue;
    }

    std::vector<amdsmi_processor_handle> handles(dev_count);
    ret = amdsmi_get_processor_handles(sockets[i], &dev_count, handles.data());
    if (ret != AMDSMI_STATUS_SUCCESS) {
      std::cerr << "Failed to get processor handles, ret=" << ret << "\n";
      return false;
    }

    for (uint32_t j = 0; j < dev_count; ++j) {
      processor_type_t processor_type = AMDSMI_PROCESSOR_TYPE_UNKNOWN;
      ret = amdsmi_get_processor_type(handles[j], &processor_type);
      if (ret != AMDSMI_STATUS_SUCCESS) {
        std::cerr << "Failed to get processor type, ret=" << ret << "\n";
        return false;
      }
      if (processor_type == AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
        out->push_back(handles[j]);
      }
    }
  }
  return !out->empty();
}

}  // namespace

int main(int argc, char **argv) {
  Options opt;
  if (!ParseArgs(argc, argv, &opt)) {
    return 1;
  }

  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  amdsmi_status_t ret = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    std::cerr << "amdsmi_init failed, ret=" << ret << "\n";
    return 1;
  }

  std::vector<amdsmi_processor_handle> gpu_handles;
  if (!GetGpuHandles(&gpu_handles)) {
    std::cerr << "No GPU handles found.\n";
    amdsmi_shut_down();
    return 1;
  }

  if (opt.device_id >= static_cast<int>(gpu_handles.size())) {
    std::cerr << "device_id out of range. Found " << gpu_handles.size()
              << " GPU devices.\n";
    amdsmi_shut_down();
    return 1;
  }

  amdsmi_processor_handle handle = gpu_handles[opt.device_id];

  uint64_t prev_energy = 0;
  float counter_resolution = 0.0f;
  uint64_t prev_ts = 0;

  ret = amdsmi_get_energy_count(handle, &prev_energy, &counter_resolution,
                                &prev_ts);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    std::cerr << "amdsmi_get_energy_count failed, ret=" << ret << "\n";
    amdsmi_shut_down();
    return 1;
  }

  std::cout << "device_id=" << opt.device_id
            << " interval_ms=" << opt.interval_ms
            << " counter_resolution_uJ=" << counter_resolution << "\n";
  std::cout << "timestamp_ns,current_socket_power_W,inst_power_W,gfx_clock_MHz,"
                "edge_temp_C,hotspot_temp_C\n";

  int sample_idx = 0;
  while (!g_stop_requested.load() &&
         (opt.samples < 0 || sample_idx < opt.samples)) {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(opt.interval_ms));
    if (g_stop_requested.load()) {
      break;
    }

    uint64_t energy = 0;
    float resolution = counter_resolution;
    uint64_t ts = 0;

    ret = amdsmi_get_energy_count(handle, &energy, &resolution, &ts);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      std::cerr << "amdsmi_get_energy_count failed, ret=" << ret << "\n";
      break;
    }

    amdsmi_power_info_t power_info{};
    ret = amdsmi_get_power_info(handle, &power_info);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      std::cerr << "amdsmi_get_power_info failed, ret=" << ret << "\n";
      break;
    }

    double inst_power_w = 0.0;
    if (ts > prev_ts && resolution > 0.0f) {
      const double delta_energy_uj =
          static_cast<double>(energy - prev_energy) * resolution;
      const double delta_time_ns = static_cast<double>(ts - prev_ts);
      inst_power_w = (delta_energy_uj * 1e-6) / (delta_time_ns * 1e-9);
    }

    double curr_socket_power_w = std::numeric_limits<double>::quiet_NaN();
    if (power_info.current_socket_power != UINT32_MAX) {
      curr_socket_power_w = static_cast<double>(power_info.current_socket_power);
    }

    amdsmi_frequencies_t freqs{};
    ret = amdsmi_get_clk_freq(handle, AMDSMI_CLK_TYPE_GFX, &freqs);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      std::cerr << "amdsmi_get_clk_freq(GFX) failed, ret=" << ret << "\n";
      break;
    }
    double gfx_clock_mhz = std::numeric_limits<double>::quiet_NaN();
    if (freqs.num_supported > 0 && freqs.current < freqs.num_supported) {
      const double raw = static_cast<double>(freqs.frequency[freqs.current]);
      // Heuristic: treat values > 1e6 as Hz, else MHz.
      gfx_clock_mhz = (raw > 1.0e6) ? (raw / 1.0e6) : raw;
    }

    // Temperature sensors are queried best-effort: unlike energy/power/clock
    // above, a failure here (unsupported sensor on some ASICs) shouldn't kill
    // an otherwise-healthy profiling run, so we fall back to NaN instead of
    // breaking the sampling loop.
    double edge_temp_c = std::numeric_limits<double>::quiet_NaN();
    int64_t edge_temp = 0;
    if (amdsmi_get_temp_metric(handle, AMDSMI_TEMPERATURE_TYPE_EDGE,
                                AMDSMI_TEMP_CURRENT, &edge_temp) ==
        AMDSMI_STATUS_SUCCESS) {
      edge_temp_c = static_cast<double>(edge_temp);
    }

    double hotspot_temp_c = std::numeric_limits<double>::quiet_NaN();
    int64_t hotspot_temp = 0;
    if (amdsmi_get_temp_metric(handle, AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
                                AMDSMI_TEMP_CURRENT, &hotspot_temp) ==
        AMDSMI_STATUS_SUCCESS) {
      hotspot_temp_c = static_cast<double>(hotspot_temp);
    }

    if (g_stop_requested.load()) {
      break;
    }

    std::cout << ts << "," << curr_socket_power_w << "," << inst_power_w
              << "," << gfx_clock_mhz << "," << edge_temp_c << ","
              << hotspot_temp_c << '\n';
    std::cout.flush();

    prev_energy = energy;
    prev_ts = ts;
    ++sample_idx;
  }

  amdsmi_shut_down();
  return 0;
}
