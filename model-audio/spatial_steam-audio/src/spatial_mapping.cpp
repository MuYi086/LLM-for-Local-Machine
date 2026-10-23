#include "spatial_mapping.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace llm_local {
namespace {

Vec3 multiply(const Vec3& value, float scalar) {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

}  // namespace

float distance_for_label(const std::string& distance) {
    static const std::map<std::string, float> distances{
        {"intimate", 0.25f},
        {"near", 0.60f},
        {"conversational", 1.50f},
        {"mid", 3.00f},
        {"far", 8.00f},
    };
    const auto value = distances.find(distance);
    if (value == distances.end()) {
        throw std::invalid_argument("unsupported distance: " + distance);
    }
    return value->second;
}

float blend_for_label(const std::string& blend, const std::string& profile) {
    static const std::map<std::string, float> blends{
        {"subtle", 0.20f}, {"medium", 0.45f}, {"strong", 0.75f}, {"full", 1.0f}};
    const auto value = blends.find(blend);
    if (value == blends.end()) {
        throw std::invalid_argument("unsupported spatial blend: " + blend);
    }
    if (profile == "standard") {
        return 0.0f;
    }
    if (profile == "balanced") {
        return value->second * 0.85f;
    }
    if (profile == "immersive") {
        return value->second;
    }
    throw std::invalid_argument("unsupported profile: " + profile);
}

Vec3 direction_for_location(const std::string& location) {
    constexpr float diagonal = 0.70710678f;
    static const std::map<std::string, Vec3> directions{
        {"front_center", {0.0f, 0.0f, -1.0f}},
        {"front_left", {-diagonal, 0.0f, -diagonal}},
        {"front_right", {diagonal, 0.0f, -diagonal}},
        {"side_left", {-1.0f, 0.0f, 0.0f}},
        {"side_right", {1.0f, 0.0f, 0.0f}},
        {"rear_left", {-diagonal, 0.0f, diagonal}},
        {"rear_center", {0.0f, 0.0f, 1.0f}},
        {"rear_right", {diagonal, 0.0f, diagonal}},
        {"above_front", {0.0f, diagonal, -diagonal}},
        {"above_rear", {0.0f, diagonal, diagonal}},
        {"below_front", {0.0f, -diagonal, -diagonal}},
    };
    const auto value = directions.find(location);
    if (value == directions.end()) {
        throw std::invalid_argument("unsupported location: " + location);
    }
    return value->second;
}

SpatialCoordinates coordinates_for_plan(
    const std::string& location,
    const std::string& distance,
    const std::string& movement) {
    const auto meters = distance_for_label(distance);
    const auto base = multiply(direction_for_location(location), meters);
    SpatialCoordinates coordinates{base, base};
    if (movement == "static") {
        return coordinates;
    }
    if (movement == "approaching") {
        coordinates.start = multiply(direction_for_location(location), std::min(8.0f, meters * 2.0f));
        coordinates.end = multiply(direction_for_location(location), std::max(0.25f, meters * 0.5f));
    } else if (movement == "receding") {
        coordinates.start = multiply(direction_for_location(location), std::max(0.25f, meters * 0.5f));
        coordinates.end = multiply(direction_for_location(location), std::min(8.0f, meters * 2.0f));
    } else if (movement == "left_to_right" || movement == "right_to_left") {
        coordinates.start = multiply(direction_for_location("front_left"), meters);
        coordinates.end = multiply(direction_for_location("front_right"), meters);
        if (movement == "right_to_left") {
            std::swap(coordinates.start, coordinates.end);
        }
    } else if (movement == "front_to_rear" || movement == "rear_to_front") {
        coordinates.start = multiply(direction_for_location("front_center"), meters);
        coordinates.end = multiply(direction_for_location("rear_center"), meters);
        if (movement == "rear_to_front") {
            std::swap(coordinates.start, coordinates.end);
        }
    } else if (movement == "rising" || movement == "falling") {
        coordinates.start = base;
        coordinates.end = base;
        coordinates.end.y = std::max(0.25f, meters * 0.7f);
        if (movement == "falling") {
            coordinates.start.y = coordinates.end.y;
            coordinates.end.y = -std::max(0.25f, meters * 0.35f);
        }
    } else {
        throw std::invalid_argument("unsupported movement: " + movement);
    }
    return coordinates;
}

Vec3 interpolate_position(const SpatialCoordinates& coordinates, float progress) {
    const auto clamped = std::clamp(progress, 0.0f, 1.0f);
    return {
        coordinates.start.x + (coordinates.end.x - coordinates.start.x) * clamped,
        coordinates.start.y + (coordinates.end.y - coordinates.start.y) * clamped,
        coordinates.start.z + (coordinates.end.z - coordinates.start.z) * clamped,
    };
}

Vec3 normalize_direction(const Vec3& position) {
    const auto length = std::sqrt(
        position.x * position.x + position.y * position.y + position.z * position.z);
    if (length <= 1e-6f) {
        return {0.0f, 0.0f, -1.0f};
    }
    return {position.x / length, position.y / length, position.z / length};
}

}  // namespace llm_local

