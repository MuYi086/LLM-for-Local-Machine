#include "render_manifest.h"

#include "json_value.h"

#include <cmath>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>

namespace llm_local {
namespace {

using Object = json::Value::Object;

void reject_unknown_fields(const Object& object, const std::set<std::string>& allowed) {
    for (const auto& [key, value] : object) {
        static_cast<void>(value);
        if (allowed.find(key) == allowed.end()) {
            throw ManifestError("unknown field: " + key);
        }
    }
}

const json::Value& require_value(const Object& object, const char* key) {
    const auto value = object.find(key);
    if (value == object.end()) throw ManifestError(std::string("missing field: ") + key);
    return value->second;
}

std::string require_string(const Object& object, const char* key) {
    try {
        return require_value(object, key).as_string();
    } catch (const json::ParseError& error) {
        throw ManifestError(std::string("field ") + key + ": " + error.what());
    }
}

double require_number(const Object& object, const char* key) {
    try {
        return require_value(object, key).as_number();
    } catch (const json::ParseError& error) {
        throw ManifestError(std::string("field ") + key + ": " + error.what());
    }
}

std::uint64_t require_uint(const Object& object, const char* key, bool allow_zero) {
    const auto number = require_number(object, key);
    if (!std::isfinite(number) || number < 0.0 || std::floor(number) != number ||
        (!allow_zero && number == 0.0)) {
        throw ManifestError(std::string("field ") + key + " must be a positive integer");
    }
    return static_cast<std::uint64_t>(number);
}

void require_enum(const std::string& field, const std::string& value, const std::set<std::string>& allowed) {
    if (allowed.find(value) == allowed.end()) {
        throw ManifestError("unsupported " + field + ": " + value);
    }
}

void validate_safe_filename(const std::string& filename) {
    if (filename.empty() || filename.size() > 255 || filename == "." || filename == ".." ||
        filename.find('/') != std::string::npos || filename.find('\\') != std::string::npos ||
        filename.find(':') != std::string::npos) {
        throw ManifestError("asset_filename must be a safe basename");
    }
}

SpatialPlan parse_spatial(const json::Value& value) {
    const auto& object = value.as_object();
    reject_unknown_fields(
        object,
        {"mode", "location", "distance", "movement", "occlusion", "spatial_blend", "source", "trajectory"});
    SpatialPlan plan{
        require_string(object, "mode"),
        require_string(object, "location"),
        require_string(object, "distance"),
        require_string(object, "movement"),
        require_string(object, "occlusion"),
        require_string(object, "spatial_blend"),
    };
    require_enum("mode", plan.mode, {"dry_center", "point", "diffuse", "preserve_stereo"});
    require_enum(
        "location",
        plan.location,
        {"front_center", "front_left", "front_right", "side_left", "side_right",
         "rear_left", "rear_center", "rear_right", "above_front", "above_rear", "below_front"});
    require_enum("distance", plan.distance, {"intimate", "near", "conversational", "mid", "far"});
    require_enum(
        "movement",
        plan.movement,
        {"static", "approaching", "receding", "left_to_right", "right_to_left",
         "front_to_rear", "rear_to_front", "rising", "falling"});
    require_enum(
        "occlusion",
        plan.occlusion,
        {"none", "wooden_door", "solid_door", "drywall", "concrete_wall", "floor_ceiling"});
    require_enum("spatial_blend", plan.spatial_blend, {"subtle", "medium", "strong", "full"});
    const auto trajectory = object.find("trajectory");
    if (trajectory != object.end() && !trajectory->second.is_null()) {
        if (plan.mode != "point" || plan.movement != "static") {
            throw ManifestError("trajectory requires mode=point and movement=static");
        }
        plan.trajectory = parse_trajectory(trajectory->second);
    }
    return plan;
}

RenderSource parse_source(const json::Value& value) {
    const auto& object = value.as_object();
    reject_unknown_fields(
        object,
        {"id", "kind", "asset_id", "asset_filename", "start_ms", "trim_start_ms",
         "duration_ms", "playback_rate", "gain_db", "loop", "spatial"});
    RenderSource source;
    source.id = require_string(object, "id");
    source.kind = require_string(object, "kind");
    source.asset_filename = require_string(object, "asset_filename");
    source.start_ms = require_uint(object, "start_ms", true);
    source.duration_ms = require_uint(object, "duration_ms", false);
    const auto gain_db = require_number(object, "gain_db");
    if (!std::isfinite(gain_db) || gain_db < -60.0 || gain_db > 12.0) {
        throw ManifestError("gain_db must be finite and between -60 and 12");
    }
    source.gain_db = static_cast<float>(gain_db);
    source.spatial = parse_spatial(require_value(object, "spatial"));
    validate_safe_filename(source.asset_filename);
    require_enum("kind", source.kind, {"narrator", "dialogue", "sfx", "ambience", "bgm"});
    if (source.kind == "bgm" && source.spatial.mode != "preserve_stereo") {
        throw ManifestError("bgm must use preserve_stereo");
    }
    return source;
}

}  // namespace

RenderManifest load_manifest(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw ManifestError("cannot read manifest: " + path.string());
    const std::string contents(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (contents.size() > 8 * 1024 * 1024) throw ManifestError("manifest exceeds 8 MiB");

    try {
        const auto root = json::parse(contents);
        const auto& object = root.as_object();
        reject_unknown_fields(
            object, {"version", "sample_rate", "timeline_duration_ms", "scene", "sources"});
        RenderManifest manifest;
        manifest.version = require_string(object, "version");
        if (manifest.version != "1.0") throw ManifestError("unsupported manifest version");
        manifest.sample_rate = static_cast<std::uint32_t>(require_uint(object, "sample_rate", false));
        if (manifest.sample_rate != 48000) throw ManifestError("sample_rate must be 48000");
        manifest.timeline_duration_ms = require_uint(object, "timeline_duration_ms", false);
        if (manifest.timeline_duration_ms > 2ULL * 60 * 60 * 1000) {
            throw ManifestError("timeline exceeds 2 hours");
        }

        const auto& scene_value = require_value(object, "scene");
        const auto& scene = scene_value.as_object();
        reject_unknown_fields(scene, {"room", "listener_pose", "acoustic_quality"});
        require_enum(
            "room",
            require_string(scene, "room"),
            {"dry_studio", "bedroom", "living_room", "narrow_corridor", "stairwell",
             "basement", "warehouse", "cave", "outdoor_open"});
        require_enum(
            "listener_pose", require_string(scene, "listener_pose"), {"center_forward"});
        manifest.profile = require_string(scene, "acoustic_quality");
        require_enum("acoustic_quality", manifest.profile, {"standard", "balanced", "immersive"});

        const auto& sources_value = require_value(object, "sources");
        const auto& sources = sources_value.as_array();
        if (sources.empty() || sources.size() > 500) throw ManifestError("sources must contain 1 to 500 items");
        std::set<std::string> ids;
        manifest.sources.reserve(sources.size());
        for (const auto& item : sources) {
            auto source = parse_source(item);
            if (source.spatial.mode == "diffuse") {
                throw ManifestError("diffuse mode is not implemented in renderer v1");
            }
            if (source.spatial.occlusion != "none") {
                throw ManifestError("occlusion is not implemented in renderer v1");
            }
            if (!ids.insert(source.id).second) throw ManifestError("duplicate source id: " + source.id);
            if (source.start_ms + source.duration_ms > manifest.timeline_duration_ms) {
                throw ManifestError("source exceeds timeline: " + source.id);
            }
            manifest.sources.push_back(std::move(source));
        }
        return manifest;
    } catch (const json::ParseError& error) {
        throw ManifestError(std::string("invalid JSON: ") + error.what());
    } catch (const TrajectoryError& error) {
        throw ManifestError(std::string("invalid trajectory: ") + error.what());
    }
}

}  // namespace llm_local
