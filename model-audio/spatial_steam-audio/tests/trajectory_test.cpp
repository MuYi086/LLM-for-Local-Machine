#include "trajectory.h"
#include "json_value.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void position(const llm_local::Trajectory& trajectory, double time, double x, double y, double z) {
    const auto actual = llm_local::trajectory_position(trajectory, time);
    expect(std::abs(actual.x - x) < 1e-5 && std::abs(actual.y - y) < 1e-5 &&
        std::abs(actual.z - z) < 1e-5, "unexpected position at " + std::to_string(time));
}
llm_local::Trajectory parse(const std::string& text) {
    return llm_local::parse_trajectory(llm_local::json::parse(text));
}
}

int main() {
    const auto linear = parse(R"({"type":"linear","start_m":[-4,0,-2],"end_m":[4,0,-2],"duration_seconds":8})");
    position(linear, 0, -4, 0, -2);
    position(linear, 4, 0, 0, -2);
    position(linear, 20, 4, 0, -2);
    const auto approaching = parse(R"({"type":"approach","azimuth_degrees":90,"elevation_degrees":0,"start_distance_m":8,"end_distance_m":1,"duration_seconds":8})");
    position(approaching, 0, 8, 0, 0);
    position(approaching, 4, 4.5, 0, 0);
    position(approaching, 8, 1, 0, 0);
    const auto receding = parse(R"({"type":"recede","azimuth_degrees":0,"elevation_degrees":90,"start_distance_m":1,"end_distance_m":8,"duration_seconds":7})");
    position(receding, 3, 0, 4, 0);
    const auto circular = parse(R"({"type":"circular","radius_m":2,"height_m":1,"start_angle_degrees":0,"end_angle_degrees":360,"duration_seconds":8,"repeat":"loop"})");
    position(circular, 0, 0, 1, -2);
    position(circular, 2, 2, 1, 0);
    position(circular, 4, 0, 1, 2);
    position(circular, 6, -2, 1, 0);
    position(circular, 10, 2, 1, 0);
    const auto crossing = parse(R"({"type":"circular","radius_m":1,"start_angle_degrees":170,"end_angle_degrees":190,"duration_seconds":2})");
    position(crossing, 1, 0, 0, 1);
    const auto nodes = parse(R"({"type":"waypoints","repeat":"ping_pong","points":[{"time_seconds":0,"position_m":[-2,0,-2]},{"time_seconds":2,"position_m":[0,2,-2]},{"time_seconds":4,"position_m":[2,0,-2]}]})");
    position(nodes, 1, -1, 1, -2);
    position(nodes, 2, 0, 2, -2);
    position(nodes, 5, 1, 1, -2);
    position(nodes, 8, -2, 0, -2);
    const auto smooth = parse(R"({"type":"linear","start_m":[0,0,-1],"end_m":[8,0,-1],"duration_seconds":4,"interpolation":"smoothstep"})");
    position(smooth, 1, 1.25, 0, -1);
    for (const auto& invalid : {
        R"({"type":"static","position_m":[0,0,0]})",
        R"({"type":"static","position_m":[true,0,-1]})",
        R"({"type":"static","position_m":[1,0,-1],"extra":2})",
        R"({"type":"linear","start_m":[-1,0,0],"end_m":[1,0,0],"duration_seconds":2})",
        R"({"type":"linear","start_m":[-1,0,-1],"end_m":[1,0,-1],"duration_seconds":2,"repeat":"loop"})",
        R"({"type":"circular","radius_m":1,"start_angle_degrees":0,"end_angle_degrees":180,"duration_seconds":2,"repeat":"loop"})",
        R"({"type":"approach","start_distance_m":1,"end_distance_m":8,"duration_seconds":2})",
        R"({"type":"waypoints","points":[{"time_seconds":0,"position_m":[1,0,0]},{"time_seconds":0,"position_m":[2,0,0]}]})"
    }) {
        bool rejected = false;
        try { static_cast<void>(parse(invalid)); } catch (const std::exception&) { rejected = true; }
        expect(rejected, "accepted invalid trajectory");
    }
    std::cout << "trajectory geometry and validation passed\n";
}
