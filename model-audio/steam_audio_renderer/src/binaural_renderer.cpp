#include "binaural_renderer.h"

#include "audio_io.h"
#include "spatial_mapping.h"

#include <phonon.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace llm_local {
namespace {

void require_success(IPLerror status, const std::string& action) {
    if (status != IPL_STATUS_SUCCESS) throw RenderError(action + " failed");
}

class SteamContext {
public:
    SteamContext(int sample_rate, int frame_size, const std::string& hrtf_path)
        : audio_settings_{sample_rate, frame_size} {
        IPLContextSettings context_settings{};
        context_settings.version = STEAMAUDIO_VERSION;
        require_success(iplContextCreate(&context_settings, &context_), "iplContextCreate");

        IPLHRTFSettings hrtf_settings{};
        hrtf_settings.type = hrtf_path.empty() ? IPL_HRTFTYPE_DEFAULT : IPL_HRTFTYPE_SOFA;
        hrtf_settings.sofaFileName = hrtf_path.empty() ? nullptr : hrtf_path.c_str();
        hrtf_settings.volume = 1.0f;
        hrtf_settings.normType = IPL_HRTFNORMTYPE_RMS;
        require_success(
            iplHRTFCreate(context_, &audio_settings_, &hrtf_settings, &hrtf_),
            "iplHRTFCreate");
    }

    ~SteamContext() {
        if (hrtf_) iplHRTFRelease(&hrtf_);
        if (context_) iplContextRelease(&context_);
    }

    SteamContext(const SteamContext&) = delete;
    SteamContext& operator=(const SteamContext&) = delete;

    IPLContext context() const { return context_; }
    IPLHRTF hrtf() const { return hrtf_; }
    IPLAudioSettings* audio_settings() { return &audio_settings_; }

private:
    IPLContext context_{};
    IPLHRTF hrtf_{};
    IPLAudioSettings audio_settings_{};
};

class SourceEffects {
public:
    explicit SourceEffects(SteamContext& steam) : steam_(steam) {
        IPLDirectEffectSettings direct_settings{};
        direct_settings.numChannels = 1;
        require_success(
            iplDirectEffectCreate(
                steam.context(), steam.audio_settings(), &direct_settings, &direct_effect_),
            "iplDirectEffectCreate");
        IPLBinauralEffectSettings binaural_settings{};
        binaural_settings.hrtf = steam.hrtf();
        require_success(
            iplBinauralEffectCreate(
                steam.context(), steam.audio_settings(), &binaural_settings, &binaural_effect_),
            "iplBinauralEffectCreate");
    }

    ~SourceEffects() {
        if (binaural_effect_) iplBinauralEffectRelease(&binaural_effect_);
        if (direct_effect_) iplDirectEffectRelease(&direct_effect_);
    }

    IPLDirectEffect direct() const { return direct_effect_; }
    IPLBinauralEffect binaural() const { return binaural_effect_; }

private:
    SteamContext& steam_;
    IPLDirectEffect direct_effect_{};
    IPLBinauralEffect binaural_effect_{};
};

float db_to_linear(float db) {
    return std::pow(10.0f, db / 20.0f);
}

SpatialCoordinates coordinates_for_source(const RenderSource& source) {
    if (source.has_position_override) {
        const Vec3 position{source.position_x, source.position_y, source.position_z};
        return {position, position};
    }
    return coordinates_for_plan(
        source.spatial.location, source.spatial.distance, source.spatial.movement);
}

std::filesystem::path resolve_asset(
    const std::filesystem::path& assets_dir,
    const std::string& filename) {
    const auto root = std::filesystem::canonical(assets_dir);
    const auto candidate = assets_dir / filename;
    if (std::filesystem::is_symlink(candidate)) throw RenderError("asset cannot be a symlink");
    const auto resolved = std::filesystem::canonical(candidate);
    if (resolved.parent_path() != root) throw RenderError("asset escapes assets directory");
    return resolved;
}

void mix_preserved_source(
    WavReader& reader,
    const RenderSource& source,
    FloatWavMixer& mixer,
    int frame_size) {
    if (reader.channels() != 2) throw RenderError("preserve_stereo source must be stereo");
    std::vector<float> samples;
    std::uint64_t source_frame = 0;
    const auto start_frame = source.start_ms * 48000 / 1000;
    const auto gain = db_to_linear(source.gain_db) * 0.5f;
    while (true) {
        const auto frames = reader.read_frames(samples, frame_size);
        if (frames == 0) break;
        mixer.mix_frames(start_frame + source_frame, samples, frames, gain);
        source_frame += frames;
    }
}

void mix_center_source(
    WavReader& reader,
    const RenderSource& source,
    FloatWavMixer& mixer,
    int frame_size) {
    if (reader.channels() != 1) throw RenderError("dry_center source must be mono");
    std::vector<float> mono;
    std::vector<float> stereo;
    std::uint64_t source_frame = 0;
    const auto start_frame = source.start_ms * 48000 / 1000;
    const auto gain = db_to_linear(source.gain_db) * 0.5f;
    while (true) {
        const auto frames = reader.read_frames(mono, frame_size);
        if (frames == 0) break;
        stereo.resize(frames * 2);
        for (std::size_t index = 0; index < frames; ++index) {
            stereo[index * 2] = mono[index];
            stereo[index * 2 + 1] = mono[index];
        }
        mixer.mix_frames(start_frame + source_frame, stereo, frames, gain);
        source_frame += frames;
    }
}

void mix_point_source(
    SteamContext& steam,
    WavReader& reader,
    const RenderSource& source,
    const RenderManifest& manifest,
    const RenderOptions& options,
    FloatWavMixer& mixer) {
    if (reader.channels() != 1) throw RenderError("point source must be mono");
    SourceEffects effects(steam);
    const auto coordinates = coordinates_for_source(source);
    const auto total_frames = std::max<std::uint64_t>(1, reader.frame_count());
    const auto start_frame = source.start_ms * 48000 / 1000;
    const auto gain = db_to_linear(source.gain_db) * 0.5f;
    const auto blend = blend_for_label(source.spatial.spatial_blend, manifest.profile);

    IPLDistanceAttenuationModel distance_model{};
    distance_model.type = IPL_DISTANCEATTENUATIONTYPE_DEFAULT;
    IPLAirAbsorptionModel air_model{};
    air_model.type = IPL_AIRABSORPTIONTYPE_DEFAULT;
    const IPLVector3 listener{0.0f, 0.0f, 0.0f};

    std::vector<float> input(options.frame_size);
    std::vector<float> direct(options.frame_size);
    std::vector<float> left(options.frame_size);
    std::vector<float> right(options.frame_size);
    std::vector<float> interleaved(options.frame_size * 2);
    float* input_channels[]{input.data()};
    float* direct_channels[]{direct.data()};
    float* output_channels[]{left.data(), right.data()};
    IPLAudioBuffer input_buffer{1, options.frame_size, input_channels};
    IPLAudioBuffer direct_buffer{1, options.frame_size, direct_channels};
    IPLAudioBuffer output_buffer{2, options.frame_size, output_channels};

    std::vector<float> decoded;
    std::uint64_t input_frame = 0;
    std::uint64_t output_frame = 0;
    while (true) {
        const auto frames = reader.read_frames(decoded, options.frame_size);
        if (frames == 0) break;
        std::fill(input.begin(), input.end(), 0.0f);
        std::copy(decoded.begin(), decoded.end(), input.begin());
        const auto progress = static_cast<float>(input_frame + frames / 2) /
            static_cast<float>(total_frames);
        const auto position = interpolate_position(coordinates, progress);
        const IPLVector3 steam_position{position.x, position.y, position.z};

        IPLDirectEffectParams direct_params{};
        direct_params.flags = static_cast<IPLDirectEffectFlags>(
            IPL_DIRECTEFFECTFLAGS_APPLYDISTANCEATTENUATION |
            IPL_DIRECTEFFECTFLAGS_APPLYAIRABSORPTION);
        direct_params.distanceAttenuation = iplDistanceAttenuationCalculate(
            steam.context(), steam_position, listener, &distance_model);
        iplAirAbsorptionCalculate(
            steam.context(), steam_position, listener, &air_model, direct_params.airAbsorption);
        iplDirectEffectApply(effects.direct(), &direct_params, &input_buffer, &direct_buffer);

        const auto direction = normalize_direction(position);
        IPLBinauralEffectParams binaural_params{};
        binaural_params.direction = IPLVector3{direction.x, direction.y, direction.z};
        binaural_params.interpolation =
            (options.force_bilinear || source.spatial.movement != "static")
            ? IPL_HRTFINTERPOLATION_BILINEAR
            : IPL_HRTFINTERPOLATION_NEAREST;
        binaural_params.spatialBlend = blend;
        binaural_params.hrtf = steam.hrtf();
        binaural_params.peakDelays = nullptr;
        iplBinauralEffectApply(
            effects.binaural(), &binaural_params, &direct_buffer, &output_buffer);
        for (int index = 0; index < options.frame_size; ++index) {
            interleaved[index * 2] = left[index];
            interleaved[index * 2 + 1] = right[index];
        }
        // 最后一帧补零后仍混入整个 effect frame，使尾部输入不会像官方最小示例一样丢失。
        mixer.mix_frames(start_frame + output_frame, interleaved, options.frame_size, gain);
        input_frame += frames;
        output_frame += options.frame_size;
    }

    auto tail_state = IPL_AUDIOEFFECTSTATE_TAILREMAINING;
    while (tail_state == IPL_AUDIOEFFECTSTATE_TAILREMAINING) {
        std::fill(left.begin(), left.end(), 0.0f);
        std::fill(right.begin(), right.end(), 0.0f);
        tail_state = iplBinauralEffectGetTail(effects.binaural(), &output_buffer);
        for (int index = 0; index < options.frame_size; ++index) {
            interleaved[index * 2] = left[index];
            interleaved[index * 2 + 1] = right[index];
        }
        mixer.mix_frames(start_frame + output_frame, interleaved, options.frame_size, gain);
        output_frame += options.frame_size;
    }
}

}  // namespace

void render_manifest(const RenderManifest& manifest, const RenderOptions& options) {
    if (manifest.sample_rate != 48000) throw RenderError("only 48 kHz manifests are supported");
    if (options.frame_size <= 0) throw RenderError("frame size must be positive");
    const auto total_frames = manifest.timeline_frames
        ? manifest.timeline_frames
        : (manifest.timeline_duration_ms * 48000 + 999) / 1000;
    FloatWavMixer mixer(options.output_path, 48000, 2, total_frames);
    SteamContext steam(48000, options.frame_size, options.hrtf_path);

    std::size_t completed_sources = 0;
    for (const auto& source : manifest.sources) {
        WavReader reader(resolve_asset(options.assets_dir, source.asset_filename));
        if (reader.sample_rate() != 48000) throw RenderError("normalized asset is not 48 kHz");
        if (source.spatial.mode == "preserve_stereo") {
            mix_preserved_source(reader, source, mixer, options.frame_size);
        } else if (source.spatial.mode == "dry_center" || manifest.profile == "standard") {
            mix_center_source(reader, source, mixer, options.frame_size);
        } else {
            mix_point_source(steam, reader, source, manifest, options, mixer);
        }
        ++completed_sources;
        if (options.progress_callback) {
            options.progress_callback(completed_sources, manifest.sources.size());
        }
    }
    mixer.finalize();
}

}  // namespace llm_local
