#pragma once

#include "json_value.h"
#include "spatial_mapping.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace llm_local {

class TrajectoryError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct Waypoint {
    double time_seconds{};
    Vec3 position;
};

struct Trajectory {
    std::string type;
    std::string repeat{"once"};
    std::string interpolation{"linear"};
    double duration_seconds{};
    Vec3 start;
    Vec3 end;
    double azimuth_degrees{};
    double elevation_degrees{};
    double start_distance_m{};
    double end_distance_m{};
    double radius_m{};
    double height_m{};
    double start_angle_degrees{};
    double end_angle_degrees{360};
    std::vector<Waypoint> points;
};

// 几何计算独立于音频块与 SDK；x 右、y 上、z 后，时间相对于本声源播放起点。
Trajectory parse_trajectory(const json::Value& value);
Trajectory load_trajectory(const std::filesystem::path& path);
Vec3 trajectory_position(const Trajectory& trajectory, double seconds);

}  // namespace llm_local
