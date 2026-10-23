#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace llm_local {

struct AudioData {
    std::uint32_t sample_rate{};
    std::uint16_t channels{};
    std::vector<float> samples;

    std::size_t frame_count() const;
};

class AudioIoError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class WavReader {
public:
    explicit WavReader(const std::filesystem::path& path);

    std::uint32_t sample_rate() const;
    std::uint16_t channels() const;
    std::uint64_t frame_count() const;
    std::size_t read_frames(std::vector<float>& interleaved, std::size_t max_frames);

private:
    std::ifstream input_;
    std::uint16_t format_code_{};
    std::uint16_t channels_{};
    std::uint32_t sample_rate_{};
    std::uint16_t block_align_{};
    std::uint16_t bits_per_sample_{};
    std::uint64_t data_bytes_{};
    std::uint64_t bytes_read_{};
};

class FloatWavMixer {
public:
    FloatWavMixer(
        const std::filesystem::path& output_path,
        std::uint32_t sample_rate,
        std::uint16_t channels,
        std::uint64_t total_frames);
    ~FloatWavMixer();

    FloatWavMixer(const FloatWavMixer&) = delete;
    FloatWavMixer& operator=(const FloatWavMixer&) = delete;

    void mix_frames(
        std::uint64_t start_frame,
        const std::vector<float>& interleaved,
        std::size_t frame_count,
        float gain);
    void finalize();

private:
    std::filesystem::path output_path_;
    std::filesystem::path temporary_path_;
    std::fstream file_;
    std::uint16_t channels_{};
    std::uint64_t total_frames_{};
    bool finalized_{};
};

AudioData read_wav(const std::filesystem::path& path);
void write_float_wav_atomic(const std::filesystem::path& path, const AudioData& audio);

}  // namespace llm_local
