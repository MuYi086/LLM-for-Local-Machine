#include "spatial_mapping.h"

#include <cassert>
#include <cmath>

int main() {
    using namespace llm_local;
    assert(std::abs(distance_for_label("intimate") - 0.25f) < 1e-6f);
    assert(std::abs(distance_for_label("far") - 8.0f) < 1e-6f);

    const auto rear_right = coordinates_for_plan("rear_right", "near", "static");
    assert(rear_right.start.x > 0.0f);
    assert(rear_right.start.z > 0.0f);

    const auto crossing = coordinates_for_plan("front_center", "mid", "left_to_right");
    assert(crossing.start.x < 0.0f);
    assert(crossing.end.x > 0.0f);
    const auto midpoint = interpolate_position(crossing, 0.5f);
    assert(std::abs(midpoint.x) < 1e-5f);

    assert(blend_for_label("full", "balanced") < blend_for_label("full", "immersive"));
    assert(blend_for_label("full", "standard") == 0.0f);
    return 0;
}
