#include "audio_io.h"
#include "binaural_renderer.h"
#include "render_manifest.h"
#include "spatial_mapping.h"

#include <phonon.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

class ArgumentError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

std::map<std::string, std::string> parse_arguments(int argc, char** argv) {
    std::map<std::string, std::string> arguments;
    const std::set<std::string> allowed{"--output", "--threads", "--hrtf", "--manifest",
        "--assets-dir", "--input", "--position", "--listener", "--sample-rate",
        "--interpolation", "--trajectory", "--trajectory-report", "--distance-exponent"};
    for (int index = 1; index < argc; index += 2) {
        const std::string key = argv[index];
        if (!allowed.count(key)) throw ArgumentError("unknown argument: " + key);
        if (key.rfind("--", 0) != 0 || index + 1 >= argc) {
            throw ArgumentError("arguments must use --name value pairs");
        }
        if (!arguments.emplace(key, argv[index + 1]).second) {
            throw ArgumentError("duplicate argument: " + key);
        }
    }
    return arguments;
}

std::string require_argument(
    const std::map<std::string, std::string>& arguments,
    const std::string& name) {
    const auto value = arguments.find(name);
    if (value == arguments.end() || value->second.empty()) {
        throw ArgumentError("missing argument: " + name);
    }
    return value->second;
}

llm_local::Vec3 parse_vector(const std::string& value, const std::string& name) {
    std::stringstream stream(value);
    llm_local::Vec3 result{};
    char comma1{};
    char comma2{};
    if (!(stream >> result.x >> comma1 >> result.y >> comma2 >> result.z) ||
        comma1 != ',' || comma2 != ',' || stream.peek() != std::char_traits<char>::eof() ||
        !std::isfinite(result.x) || !std::isfinite(result.y) || !std::isfinite(result.z)) {
        throw ArgumentError(name + " must be x,y,z with finite numbers");
    }
    return result;
}

int parse_positive_integer(const std::string& value, const std::string& name) {
    try {
        std::size_t consumed = 0;
        const auto parsed = std::stoi(value, &consumed);
        if (consumed != value.size() || parsed <= 0) throw ArgumentError(name + " must be positive");
        return parsed;
    } catch (const std::invalid_argument&) {
        throw ArgumentError(name + " must be an integer");
    } catch (const std::out_of_range&) {
        throw ArgumentError(name + " is out of range");
    }
}

double parse_distance_exponent(const std::string& value) {
    try {
        std::size_t consumed = 0;
        const auto parsed = std::stod(value, &consumed);
        if (consumed != value.size() || !std::isfinite(parsed) || parsed < 0 || parsed > 1) {
            throw ArgumentError("--distance-exponent must be finite and between 0 and 1");
        }
        return parsed;
    } catch (const std::invalid_argument&) {
        throw ArgumentError("--distance-exponent must be a number between 0 and 1");
    } catch (const std::out_of_range&) {
        throw ArgumentError("--distance-exponent is out of range");
    }
}

llm_local::RenderManifest single_file_manifest(
    const std::filesystem::path& input,
    const llm_local::Vec3& position,
    const llm_local::Vec3& listener) {
    llm_local::WavReader reader(input);
    if (reader.sample_rate() != 48000 || reader.channels() != 1) {
        throw llm_local::AudioIoError("single-file input must be 48 kHz mono WAV");
    }
    llm_local::RenderManifest manifest;
    manifest.version = "1.0";
    manifest.sample_rate = 48000;
    manifest.timeline_frames = reader.frame_count();
    manifest.timeline_duration_ms = (reader.frame_count() * 1000 + 47999) / 48000;
    manifest.profile = "immersive";
    llm_local::RenderSource source;
    source.id = "single_source";
    source.kind = "sfx";
    source.asset_filename = input.filename().string();
    source.start_ms = 0;
    source.duration_ms = manifest.timeline_duration_ms;
    source.gain_db = 0.0f;
    source.spatial = {"point", "front_center", "conversational", "static", "none", "full"};
    source.has_position_override = true;
    source.position_x = position.x - listener.x;
    source.position_y = position.y - listener.y;
    source.position_z = position.z - listener.z;
    manifest.sources.push_back(std::move(source));
    return manifest;
}

void print_report(const llm_local::RenderManifest& manifest,
    const llm_local::RenderOptions& options, const llm_local::RenderStats& stats) {
    std::cout << "{\"ok\":true,\"sample_rate\":48000,\"channels\":2,\"duration_ms\":"
              << manifest.timeline_duration_ms << ",\"sources\":" << manifest.sources.size()
              << ",\"output_frames\":" << stats.output_frames
              << ",\"tail_samples\":" << stats.tail_samples
              << ",\"direct_settle_samples\":" << stats.direct_settle_samples
              << ",\"distance_exponent\":" << options.distance_exponent
              << ",\"mix_gain\":" << (options.unit_gain_output ? 1.0 : 0.5)
              << ",\"peak_normalized\":" << (options.unit_gain_output ? "false" : "true")
              << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--info") {
            std::cout << "{\"engine\":\"steam-audio\",\"trajectory_version\":1,\"sample_rate\":48000,"
                         "\"sdk_version\":" << STEAMAUDIO_VERSION << ",\"trajectory_types\":["
                         "\"static\",\"linear\",\"approach\",\"recede\",\"circular\",\"waypoints\"],\"distance_exponent_control\":true}\n";
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--validate-trajectory") {
            static_cast<void>(llm_local::load_trajectory(argv[2]));
            std::cout << "{\"ok\":true,\"trajectory_version\":1}\n";
            return 0;
        }
        const auto arguments = parse_arguments(argc, argv);
        llm_local::RenderOptions options;
        options.output_path = require_argument(arguments, "--output");
        if (arguments.count("--distance-exponent")) {
            options.distance_exponent = parse_distance_exponent(arguments.at("--distance-exponent"));
            if (!arguments.count("--trajectory") || arguments.count("--manifest")) {
                throw ArgumentError("--distance-exponent requires single-source --trajectory mode");
            }
        }
        options.threads = arguments.count("--threads")
            ? parse_positive_integer(arguments.at("--threads"), "--threads")
            : 1;
        if (arguments.count("--hrtf") && arguments.at("--hrtf") != "default") {
            options.hrtf_path = arguments.at("--hrtf");
        }
        options.progress_callback = [](std::size_t completed, std::size_t total) {
            std::cerr << "LLM_LOCAL_PROGRESS " << completed << ' ' << total << '\n';
        };

        llm_local::RenderManifest manifest;
        if (arguments.count("--manifest")) {
            if (arguments.count("--trajectory") || arguments.count("--input")) {
                throw ArgumentError("--manifest cannot be combined with --input or --trajectory");
            }
            manifest = llm_local::load_manifest(require_argument(arguments, "--manifest"));
            options.assets_dir = require_argument(arguments, "--assets-dir");
        } else {
            const std::filesystem::path input = require_argument(arguments, "--input");
            const bool trajectory_mode = arguments.count("--trajectory");
            if (trajectory_mode && (arguments.count("--position") || arguments.count("--listener"))) {
                throw ArgumentError("trajectory uses listener-relative coordinates; omit --position and --listener");
            }
            const auto position = trajectory_mode ? llm_local::Vec3{} :
                parse_vector(require_argument(arguments, "--position"), "--position");
            const auto listener = arguments.count("--listener")
                ? parse_vector(arguments.at("--listener"), "--listener")
                : llm_local::Vec3{};
            if (arguments.count("--sample-rate") && arguments.at("--sample-rate") != "48000") {
                throw ArgumentError("--sample-rate must be 48000");
            }
            if (arguments.count("--interpolation")) {
                const auto interpolation = arguments.at("--interpolation");
                if (interpolation != "nearest" && interpolation != "bilinear") {
                    throw ArgumentError("--interpolation must be nearest or bilinear");
                }
                options.force_bilinear = interpolation == "bilinear";
            }
            manifest = single_file_manifest(input, position, listener);
            if (trajectory_mode) {
                if (manifest.timeline_frames == 0 || manifest.timeline_frames > 7200ULL * 48000) {
                    throw ArgumentError("trajectory input must be nonempty and at most two hours");
                }
                if (std::filesystem::exists(options.output_path)) {
                    throw ArgumentError("trajectory output already exists; use a new path");
                }
                auto& source = manifest.sources.front();
                source.has_position_override = false;
                source.spatial.trajectory = llm_local::load_trajectory(arguments.at("--trajectory"));
                options.unit_gain_output = true;
                options.preserve_tail = true;
            }
            options.assets_dir = input.parent_path().empty() ? "." : input.parent_path();
        }

        if (arguments.count("--trajectory-report")) {
            options.trajectory_report = arguments.at("--trajectory-report");
            if (std::filesystem::exists(options.trajectory_report) ||
                std::filesystem::absolute(options.trajectory_report).lexically_normal() ==
                    std::filesystem::absolute(options.output_path).lexically_normal()) {
                throw ArgumentError("trajectory report must use a new, distinct path");
            }
        }

        const auto stats = llm_local::render_manifest(manifest, options);
        print_report(manifest, options, stats);
        return 0;
    } catch (const ArgumentError& error) {
        std::cerr << "argument error: " << error.what() << '\n';
        return 2;
    } catch (const llm_local::ManifestError& error) {
        std::cerr << "manifest error: " << error.what() << '\n';
        return 2;
    } catch (const llm_local::TrajectoryError& error) {
        std::cerr << "trajectory error: " << error.what() << '\n';
        return 2;
    } catch (const llm_local::AudioIoError& error) {
        std::cerr << "audio I/O error: " << error.what() << '\n';
        return 3;
    } catch (const llm_local::RenderError& error) {
        std::cerr << "Steam Audio render error: " << error.what() << '\n';
        return 4;
    } catch (const std::exception& error) {
        std::cerr << "output error: " << error.what() << '\n';
        return 5;
    }
}
