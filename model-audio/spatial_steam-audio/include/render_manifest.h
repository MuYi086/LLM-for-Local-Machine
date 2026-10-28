#pragma once

#include "trajectory.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace llm_local {

struct SpatialPlan {
    std::string mode;
    std::string location;
    std::string distance;
    std::string movement;
    std::string occlusion;
    std::string spatial_blend;
    std::optional<Trajectory> trajectory{};
};

struct RenderSource {
    std::string id;
    std::string kind;
    std::string asset_filename;
    std::uint64_t start_ms{};
    std::uint64_t duration_ms{};
    float gain_db{};
    SpatialPlan spatial;
    bool has_position_override{};
    float position_x{};
    float position_y{};
    float position_z{};
};

struct RenderManifest {
    std::string version;
    std::uint32_t sample_rate{};
    std::uint64_t timeline_duration_ms{};
    std::uint64_t timeline_frames{};
    std::string profile;
    std::vector<RenderSource> sources;
};

class ManifestError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

RenderManifest load_manifest(const std::filesystem::path& path);

}  // namespace llm_local
