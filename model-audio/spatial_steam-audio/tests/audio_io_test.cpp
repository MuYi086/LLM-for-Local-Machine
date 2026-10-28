#include "audio_io.h"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <vector>

int main() {
    const auto path = std::filesystem::temp_directory_path() / "llm-local-audio-io-test.wav";
    {
        llm_local::FloatWavMixer mixer(path, 48000, 2, 3);
        const std::vector<float> four_frames(8, 1.0f);
        // 只允许最后一帧写入，验证严格裁剪总时间线。
        mixer.mix_frames(2, four_frames, 4, 1.0f);
        mixer.mix_frames(2, four_frames, 1, 1.0f);
        mixer.finalize();
    }
    const auto audio = llm_local::read_wav(path);
    assert(audio.sample_rate == 48000);
    assert(audio.channels == 2);
    assert(audio.frame_count() == 3);
    assert(std::abs(audio.samples[0]) < 1e-6f);
    assert(std::abs(audio.samples[3]) < 1e-6f);
    assert(std::abs(audio.samples[4] - 0.98f) < 1e-5f);
    assert(std::abs(audio.samples[5] - 0.98f) < 1e-5f);
    std::filesystem::remove(path);
    {
        llm_local::FloatWavMixer mixer(path, 48000, 2, 1);
        mixer.mix_frames(0, {2.0f, -3.0f}, 1, 1.0f);
        mixer.finalize(false);
    }
    const auto linear = llm_local::read_wav(path);
    if (linear.samples != std::vector<float>{2.0f, -3.0f}) {
        throw std::runtime_error("linear float output must retain headroom without scaling");
    }
    std::filesystem::remove(path);
    return 0;
}
