#include "render_manifest.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::filesystem::path write_manifest(const std::string& content, const std::string& name) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << content;
    return path;
}

}  // namespace

int main() {
    const std::string valid = R"json({
        "version":"1.0",
        "sample_rate":48000,
        "timeline_duration_ms":2000,
        "scene":{"room":"dry_studio","listener_pose":"center_forward","acoustic_quality":"balanced"},
        "sources":[{
            "id":"sfx_1","kind":"sfx","asset_id":"asset_1","asset_filename":"source_0000.wav",
            "start_ms":100,"trim_start_ms":0,"duration_ms":500,"playback_rate":1.0,
            "gain_db":-3,"loop":false,
            "spatial":{"mode":"point","location":"rear_right","distance":"near","movement":"static",
                       "occlusion":"none","spatial_blend":"strong","source":"rule"}
        }]
    })json";
    const auto valid_path = write_manifest(valid, "llm-local-render-manifest-valid.json");
    const auto manifest = llm_local::load_manifest(valid_path);
    assert(manifest.version == "1.0");
    assert(manifest.sample_rate == 48000);
    assert(manifest.sources.size() == 1);
    assert(manifest.sources.front().spatial.location == "rear_right");
    std::filesystem::remove(valid_path);

    const auto unsafe = valid.find("source_0000.wav");
    auto unsafe_manifest = valid;
    unsafe_manifest.replace(unsafe, std::string("source_0000.wav").size(), "../outside.wav");
    const auto unsafe_path = write_manifest(unsafe_manifest, "llm-local-render-manifest-unsafe.json");
    bool rejected = false;
    try {
        static_cast<void>(llm_local::load_manifest(unsafe_path));
    } catch (const llm_local::ManifestError&) {
        rejected = true;
    }
    assert(rejected);
    std::filesystem::remove(unsafe_path);

    auto unknown_manifest = valid;
    const auto root_close = unknown_manifest.rfind('}');
    unknown_manifest.insert(root_close, R"json(,"filter_hz":1200)json");
    const auto unknown_path = write_manifest(
        unknown_manifest, "llm-local-render-manifest-unknown.json");
    rejected = false;
    try {
        static_cast<void>(llm_local::load_manifest(unknown_path));
    } catch (const llm_local::ManifestError&) {
        rejected = true;
    }
    assert(rejected);
    std::filesystem::remove(unknown_path);
    return 0;
}
