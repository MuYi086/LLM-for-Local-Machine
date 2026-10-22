#pragma once

#include <string>

namespace llm_local {

struct Vec3 {
    float x{};
    float y{};
    float z{};
};

struct SpatialCoordinates {
    Vec3 start;
    Vec3 end;
};

float distance_for_label(const std::string& distance);
float blend_for_label(const std::string& blend, const std::string& profile);
Vec3 direction_for_location(const std::string& location);
SpatialCoordinates coordinates_for_plan(
    const std::string& location,
    const std::string& distance,
    const std::string& movement);
Vec3 interpolate_position(const SpatialCoordinates& coordinates, float progress);
Vec3 normalize_direction(const Vec3& position);

}  // namespace llm_local

