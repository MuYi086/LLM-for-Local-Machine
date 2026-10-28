#pragma once

#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>

#include "render_manifest.h"

namespace llm_local {

struct RenderOptions {
    std::filesystem::path assets_dir;
    std::filesystem::path output_path;
    std::string hrtf_path;
    int frame_size{1024};
    int threads{1};
    bool force_bilinear{};
    bool unit_gain_output{};
    bool preserve_tail{};
    double distance_exponent{1.0};
    std::filesystem::path trajectory_report;
    std::function<void(std::size_t, std::size_t)> progress_callback;
};

class RenderError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct RenderStats {
    std::uint64_t output_frames{};
    std::uint64_t tail_samples{};
    std::uint64_t direct_settle_samples{};
};

RenderStats render_manifest(const RenderManifest& manifest, const RenderOptions& options);

}  // namespace llm_local
