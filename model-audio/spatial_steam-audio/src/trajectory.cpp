#include "trajectory.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <set>

namespace llm_local {
namespace {
using Object = json::Value::Object;
constexpr double pi = 3.14159265358979323846;

void allowed(const Object& object, const std::set<std::string>& fields) {
    for (const auto& item : object) {
        if (!fields.count(item.first)) throw TrajectoryError("unknown trajectory field: " + item.first);
    }
}

const json::Value& required(const Object& object, const char* name) {
    const auto found = object.find(name);
    if (found == object.end()) throw TrajectoryError(std::string("missing trajectory field: ") + name);
    return found->second;
}

double number(const Object& object, const char* name, double low, double high) {
    const auto value = required(object, name).as_number();
    if (!std::isfinite(value) || value < low || value > high) {
        throw TrajectoryError(std::string("trajectory value out of range: ") + name);
    }
    return value;
}

double optional_number(const Object& object, const char* name, double fallback, double low, double high) {
    return object.count(name) ? number(object, name, low, high) : fallback;
}

std::string optional_choice(const Object& object, const char* name,
    const std::string& fallback, const std::set<std::string>& choices) {
    const auto value = object.count(name) ? required(object, name).as_string() : fallback;
    if (!choices.count(value)) throw TrajectoryError(std::string("unsupported trajectory ") + name + ": " + value);
    return value;
}

Vec3 point(const Object& object, const char* name) {
    const auto& values = required(object, name).as_array();
    if (values.size() != 3) throw TrajectoryError(std::string(name) + " must contain x,y,z");
    double coordinates[3];
    for (int i = 0; i < 3; ++i) {
        coordinates[i] = values[i].as_number();
        if (!std::isfinite(coordinates[i]) || std::abs(coordinates[i]) > 1000) {
            throw TrajectoryError("invalid position component");
        }
    }
    const auto distance = std::hypot(coordinates[0], coordinates[1], coordinates[2]);
    if (distance < 0.25 - 1e-6 || distance > 1000 + 1e-6) {
        throw TrajectoryError("position distance must be 0.25 to 1000 meters");
    }
    return {static_cast<float>(coordinates[0]), static_cast<float>(coordinates[1]), static_cast<float>(coordinates[2])};
}

bool same_point(const Vec3& a, const Vec3& b) {
    return std::hypot(static_cast<double>(a.x) - b.x, static_cast<double>(a.y) - b.y,
        static_cast<double>(a.z) - b.z) <= 1e-6;
}

void segment(const Vec3& a, const Vec3& b) {
    const double dx = static_cast<double>(b.x) - a.x;
    const double dy = static_cast<double>(b.y) - a.y;
    const double dz = static_cast<double>(b.z) - a.z;
    const auto length2 = dx * dx + dy * dy + dz * dz;
    const auto t = length2 ? std::clamp(-(a.x * dx + a.y * dy + a.z * dz) / length2, 0.0, 1.0) : 0;
    if (std::hypot(a.x + t * dx, a.y + t * dy, a.z + t * dz) < 0.25 - 1e-6) {
        throw TrajectoryError("trajectory segment passes within 0.25 meters of listener");
    }
}

double ease(double progress, const Trajectory& trajectory) {
    return trajectory.interpolation == "smoothstep" ? progress * progress * (3 - 2 * progress) : progress;
}

Vec3 interpolate(const Vec3& a, const Vec3& b, double t) {
    return {static_cast<float>(a.x + (b.x - a.x) * t), static_cast<float>(a.y + (b.y - a.y) * t),
        static_cast<float>(a.z + (b.z - a.z) * t)};
}
}  // namespace

Trajectory parse_trajectory(const json::Value& value) {
    try {
        const auto& object = value.as_object();
        Trajectory t;
        t.type = required(object, "type").as_string();
        if (t.type == "static") {
            allowed(object, {"type", "position_m"});
            t.start = point(object, "position_m");
            return t;
        }
        t.repeat = optional_choice(object, "repeat", "once", {"once", "loop", "ping_pong"});
        t.interpolation = optional_choice(object, "interpolation", "linear", {"linear", "smoothstep"});
        if (t.type == "linear") {
            allowed(object, {"type", "repeat", "interpolation", "start_m", "end_m", "duration_seconds"});
            t.start = point(object, "start_m");
            t.end = point(object, "end_m");
            segment(t.start, t.end);
            if (t.repeat == "loop" && !same_point(t.start, t.end)) throw TrajectoryError("loop must be closed");
        } else if (t.type == "approach" || t.type == "recede") {
            allowed(object, {"type", "repeat", "interpolation", "azimuth_degrees", "elevation_degrees",
                "start_distance_m", "end_distance_m", "duration_seconds"});
            t.azimuth_degrees = optional_number(object, "azimuth_degrees", 0, -360000, 360000);
            t.elevation_degrees = optional_number(object, "elevation_degrees", 0, -90, 90);
            t.start_distance_m = number(object, "start_distance_m", 0.25, 1000);
            t.end_distance_m = number(object, "end_distance_m", 0.25, 1000);
            if ((t.type == "approach" && t.end_distance_m >= t.start_distance_m) ||
                (t.type == "recede" && t.end_distance_m <= t.start_distance_m) || t.repeat == "loop") {
                throw TrajectoryError("invalid approach/recede direction or repeat");
            }
        } else if (t.type == "circular") {
            allowed(object, {"type", "repeat", "interpolation", "radius_m", "height_m",
                "start_angle_degrees", "end_angle_degrees", "duration_seconds"});
            t.radius_m = number(object, "radius_m", 0.25, 1000);
            t.height_m = optional_number(object, "height_m", 0, -1000, 1000);
            if (std::hypot(t.radius_m, t.height_m) > 1000 + 1e-6) throw TrajectoryError("circle too far from listener");
            t.start_angle_degrees = optional_number(object, "start_angle_degrees", 0, -360000, 360000);
            t.end_angle_degrees = optional_number(object, "end_angle_degrees", 360, -360000, 360000);
            if (t.repeat == "loop" && std::abs(std::remainder(t.end_angle_degrees - t.start_angle_degrees, 360)) > 1e-6) {
                throw TrajectoryError("loop circle must contain whole turns");
            }
        } else if (t.type == "waypoints") {
            allowed(object, {"type", "repeat", "interpolation", "points"});
            const auto& points_value = required(object, "points");
            const auto& points = points_value.as_array();
            if (points.size() < 2 || points.size() > 1024) throw TrajectoryError("expected 2 to 1024 waypoints");
            for (const auto& item : points) {
                const auto& node = item.as_object();
                allowed(node, {"time_seconds", "position_m"});
                Waypoint next{number(node, "time_seconds", 0, 7200), point(node, "position_m")};
                if (t.points.empty()) {
                    if (next.time_seconds != 0) throw TrajectoryError("first waypoint time must be zero");
                } else {
                    if (next.time_seconds <= t.points.back().time_seconds) throw TrajectoryError("waypoint times must increase");
                    segment(t.points.back().position, next.position);
                }
                t.points.push_back(next);
            }
            t.duration_seconds = t.points.back().time_seconds;
            if (t.repeat == "loop" && !same_point(t.points.front().position, t.points.back().position)) {
                throw TrajectoryError("loop waypoints must be closed");
            }
        } else {
            throw TrajectoryError("unsupported trajectory type: " + t.type);
        }
        if (t.type != "waypoints") {
            t.duration_seconds = number(object, "duration_seconds", 0, 7200);
            if (t.duration_seconds <= 0) throw TrajectoryError("duration_seconds must be positive");
        }
        return t;
    } catch (const json::ParseError& error) {
        throw TrajectoryError(error.what());
    }
}

Trajectory load_trajectory(const std::filesystem::path& path) {
    if (std::filesystem::file_size(path) > 1024 * 1024) throw TrajectoryError("trajectory exceeds 1 MiB");
    std::ifstream input(path, std::ios::binary);
    if (!input) throw TrajectoryError("cannot read trajectory");
    const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    try { return parse_trajectory(json::parse(text)); }
    catch (const json::ParseError& error) { throw TrajectoryError(error.what()); }
}

Vec3 trajectory_position(const Trajectory& t, double seconds) {
    if (!std::isfinite(seconds) || seconds < 0) throw TrajectoryError("trajectory time must be finite and nonnegative");
    if (t.type == "static") return t.start;
    double time = std::min(seconds, t.duration_seconds);
    if (t.repeat == "loop") time = std::fmod(seconds, t.duration_seconds);
    if (t.repeat == "ping_pong") {
        time = std::fmod(seconds, 2 * t.duration_seconds);
        if (time > t.duration_seconds) time = 2 * t.duration_seconds - time;
    }
    if (t.type == "waypoints") {
        auto right = std::upper_bound(t.points.begin(), t.points.end(), time,
            [](double value, const Waypoint& node) { return value < node.time_seconds; });
        if (right == t.points.end()) return t.points.back().position;
        const auto& left = *(right - 1);
        return interpolate(left.position, right->position,
            ease((time - left.time_seconds) / (right->time_seconds - left.time_seconds), t));
    }
    const auto progress = ease(time / t.duration_seconds, t);
    if (t.type == "linear") return interpolate(t.start, t.end, progress);
    if (t.type == "circular") {
        const auto angle = (t.start_angle_degrees + progress * (t.end_angle_degrees - t.start_angle_degrees)) * pi / 180;
        return {static_cast<float>(t.radius_m * std::sin(angle)), static_cast<float>(t.height_m),
            static_cast<float>(-t.radius_m * std::cos(angle))};
    }
    const auto distance = t.start_distance_m + progress * (t.end_distance_m - t.start_distance_m);
    const auto azimuth = t.azimuth_degrees * pi / 180;
    const auto elevation = t.elevation_degrees * pi / 180;
    return {static_cast<float>(distance * std::cos(elevation) * std::sin(azimuth)),
        static_cast<float>(distance * std::sin(elevation)),
        static_cast<float>(-distance * std::cos(elevation) * std::cos(azimuth))};
}
}  // namespace llm_local
