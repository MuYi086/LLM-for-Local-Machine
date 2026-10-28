#include "audio_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>

namespace llm_local {
namespace {

constexpr std::uint16_t kPcmFormat = 1;
constexpr std::uint16_t kFloatFormat = 3;
constexpr std::uint16_t kExtensibleFormat = 0xfffe;
constexpr std::uint64_t kWavHeaderSize = 44;

std::uint16_t read_u16(const unsigned char* bytes) {
    return static_cast<std::uint16_t>(bytes[0] | (bytes[1] << 8));
}

std::uint32_t read_u32(const unsigned char* bytes) {
    return static_cast<std::uint32_t>(
        bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24));
}

void write_u16(std::ostream& output, std::uint16_t value) {
    const std::array<char, 2> bytes{
        static_cast<char>(value & 0xff), static_cast<char>((value >> 8) & 0xff)};
    output.write(bytes.data(), bytes.size());
}

void write_u32(std::ostream& output, std::uint32_t value) {
    const std::array<char, 4> bytes{
        static_cast<char>(value & 0xff),
        static_cast<char>((value >> 8) & 0xff),
        static_cast<char>((value >> 16) & 0xff),
        static_cast<char>((value >> 24) & 0xff),
    };
    output.write(bytes.data(), bytes.size());
}

float decode_sample(const unsigned char* bytes, std::uint16_t format, std::uint16_t bits) {
    if (format == kFloatFormat && bits == 32) {
        float value{};
        std::memcpy(&value, bytes, sizeof(value));
        return value;
    }
    if (format != kPcmFormat) throw AudioIoError("unsupported WAV sample format");
    if (bits == 16) {
        const auto value = static_cast<std::int16_t>(read_u16(bytes));
        return static_cast<float>(value) / 32768.0f;
    }
    if (bits == 24) {
        std::int32_t value = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16);
        if (value & 0x800000) value |= ~0xffffff;
        return static_cast<float>(value) / 8388608.0f;
    }
    if (bits == 32) {
        const auto raw = read_u32(bytes);
        const auto value = static_cast<std::int32_t>(raw);
        return static_cast<float>(static_cast<double>(value) / 2147483648.0);
    }
    throw AudioIoError("unsupported PCM bit depth");
}

std::filesystem::path temporary_sibling(const std::filesystem::path& path) {
    std::random_device device;
    return path.parent_path() /
        ("." + path.filename().string() + "." + std::to_string(device()) + ".part");
}

void write_header(
    std::ostream& output,
    std::uint32_t sample_rate,
    std::uint16_t channels,
    std::uint64_t total_frames) {
    const auto data_bytes_64 = total_frames * channels * sizeof(float);
    if (data_bytes_64 > std::numeric_limits<std::uint32_t>::max() - 36ULL) {
        throw AudioIoError("WAV exceeds RIFF 4 GiB limit");
    }
    const auto data_bytes = static_cast<std::uint32_t>(data_bytes_64);
    output.write("RIFF", 4);
    write_u32(output, data_bytes + 36);
    output.write("WAVE", 4);
    output.write("fmt ", 4);
    write_u32(output, 16);
    write_u16(output, kFloatFormat);
    write_u16(output, channels);
    write_u32(output, sample_rate);
    write_u32(output, sample_rate * channels * sizeof(float));
    write_u16(output, channels * sizeof(float));
    write_u16(output, 32);
    output.write("data", 4);
    write_u32(output, data_bytes);
}

}  // namespace

std::size_t AudioData::frame_count() const {
    return channels ? samples.size() / channels : 0;
}

WavReader::WavReader(const std::filesystem::path& path) : input_(path, std::ios::binary) {
    if (!input_) throw AudioIoError("cannot open WAV: " + path.string());
    std::array<unsigned char, 12> riff{};
    input_.read(reinterpret_cast<char*>(riff.data()), riff.size());
    if (input_.gcount() != static_cast<std::streamsize>(riff.size()) ||
        std::memcmp(riff.data(), "RIFF", 4) != 0 ||
        std::memcmp(riff.data() + 8, "WAVE", 4) != 0) {
        throw AudioIoError("input is not a RIFF/WAVE file");
    }

    bool found_format = false;
    while (input_) {
        std::array<unsigned char, 8> chunk_header{};
        input_.read(reinterpret_cast<char*>(chunk_header.data()), chunk_header.size());
        if (input_.gcount() != static_cast<std::streamsize>(chunk_header.size())) break;
        const auto chunk_size = read_u32(chunk_header.data() + 4);
        if (std::memcmp(chunk_header.data(), "fmt ", 4) == 0) {
            if (chunk_size < 16 || chunk_size > 1024) throw AudioIoError("invalid WAV fmt chunk");
            std::vector<unsigned char> format(chunk_size);
            input_.read(reinterpret_cast<char*>(format.data()), chunk_size);
            if (input_.gcount() != static_cast<std::streamsize>(chunk_size)) {
                throw AudioIoError("truncated WAV fmt chunk");
            }
            format_code_ = read_u16(format.data());
            channels_ = read_u16(format.data() + 2);
            sample_rate_ = read_u32(format.data() + 4);
            block_align_ = read_u16(format.data() + 12);
            bits_per_sample_ = read_u16(format.data() + 14);
            if (format_code_ == kExtensibleFormat) {
                if (chunk_size < 40) throw AudioIoError("invalid extensible WAV fmt chunk");
                format_code_ = read_u16(format.data() + 24);
            }
            found_format = true;
        } else if (std::memcmp(chunk_header.data(), "data", 4) == 0) {
            if (!found_format) throw AudioIoError("WAV data chunk precedes fmt chunk");
            data_bytes_ = chunk_size;
            break;
        } else {
            input_.seekg(chunk_size, std::ios::cur);
        }
        if (chunk_size & 1U) input_.seekg(1, std::ios::cur);
    }
    if (!found_format || data_bytes_ == 0) throw AudioIoError("WAV is missing audio data");
    if ((format_code_ != kPcmFormat && format_code_ != kFloatFormat) ||
        (bits_per_sample_ != 16 && bits_per_sample_ != 24 && bits_per_sample_ != 32)) {
        throw AudioIoError("unsupported WAV encoding");
    }
    if (channels_ == 0 || channels_ > 2 || sample_rate_ == 0 ||
        block_align_ != channels_ * (bits_per_sample_ / 8)) {
        throw AudioIoError("invalid WAV channel or block alignment");
    }
}

std::uint32_t WavReader::sample_rate() const { return sample_rate_; }
std::uint16_t WavReader::channels() const { return channels_; }
std::uint64_t WavReader::frame_count() const { return data_bytes_ / block_align_; }

std::size_t WavReader::read_frames(std::vector<float>& interleaved, std::size_t max_frames) {
    const auto remaining_frames = (data_bytes_ - bytes_read_) / block_align_;
    const auto frames = static_cast<std::size_t>(std::min<std::uint64_t>(remaining_frames, max_frames));
    if (frames == 0) {
        interleaved.clear();
        return 0;
    }
    std::vector<unsigned char> raw(frames * block_align_);
    input_.read(reinterpret_cast<char*>(raw.data()), raw.size());
    if (input_.gcount() != static_cast<std::streamsize>(raw.size())) {
        throw AudioIoError("truncated WAV data chunk");
    }
    bytes_read_ += raw.size();
    interleaved.resize(frames * channels_);
    const auto bytes_per_sample = bits_per_sample_ / 8;
    for (std::size_t index = 0; index < interleaved.size(); ++index) {
        const auto value = decode_sample(raw.data() + index * bytes_per_sample, format_code_, bits_per_sample_);
        if (!std::isfinite(value)) throw AudioIoError("WAV contains NaN or Inf");
        interleaved[index] = value;
    }
    return frames;
}

FloatWavMixer::FloatWavMixer(
    const std::filesystem::path& output_path,
    std::uint32_t sample_rate,
    std::uint16_t channels,
    std::uint64_t total_frames)
    : output_path_(output_path),
      temporary_path_(temporary_sibling(output_path)),
      channels_(channels),
      total_frames_(total_frames) {
    if (channels != 2 || total_frames == 0) throw AudioIoError("mixer requires non-empty stereo output");
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    file_.open(temporary_path_, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    if (!file_) throw AudioIoError("cannot create output WAV");
    write_header(file_, sample_rate, channels, total_frames);
    const auto data_bytes = total_frames * channels * sizeof(float);
    file_.seekp(static_cast<std::streamoff>(kWavHeaderSize + data_bytes - 1));
    file_.put('\0');
    file_.flush();
}

FloatWavMixer::~FloatWavMixer() {
    if (file_.is_open()) file_.close();
    if (!finalized_) {
        std::error_code ignored;
        std::filesystem::remove(temporary_path_, ignored);
    }
}

void FloatWavMixer::mix_frames(
    std::uint64_t start_frame,
    const std::vector<float>& interleaved,
    std::size_t frame_count,
    float gain) {
    if (!std::isfinite(gain)) throw AudioIoError("mix gain is not finite");
    if (start_frame >= total_frames_ || frame_count == 0) return;
    const auto clipped_frames = static_cast<std::size_t>(
        std::min<std::uint64_t>(frame_count, total_frames_ - start_frame));
    if (interleaved.size() < clipped_frames * channels_) throw AudioIoError("mix buffer is too small");
    std::vector<float> existing(clipped_frames * channels_);
    const auto offset = static_cast<std::streamoff>(
        kWavHeaderSize + start_frame * channels_ * sizeof(float));
    file_.clear();
    file_.seekg(offset);
    file_.read(reinterpret_cast<char*>(existing.data()), existing.size() * sizeof(float));
    if (file_.gcount() != static_cast<std::streamsize>(existing.size() * sizeof(float))) {
        throw AudioIoError("cannot read output mix region");
    }
    for (std::size_t index = 0; index < existing.size(); ++index) {
        const auto mixed = existing[index] + interleaved[index] * gain;
        if (!std::isfinite(mixed)) throw AudioIoError("mix produced NaN or Inf");
        existing[index] = mixed;
    }
    file_.clear();
    file_.seekp(offset);
    file_.write(reinterpret_cast<const char*>(existing.data()), existing.size() * sizeof(float));
    if (!file_) throw AudioIoError("cannot write output mix region");
}

void FloatWavMixer::finalize(bool normalize_peak) {
    file_.flush();
    file_.close();
    std::fstream scan(temporary_path_, std::ios::binary | std::ios::in | std::ios::out);
    if (!scan) throw AudioIoError("cannot reopen output WAV");
    scan.seekg(kWavHeaderSize);
    std::vector<float> buffer(64 * 1024);
    float peak = 0.0f;
    std::uint64_t remaining = total_frames_ * channels_;
    while (remaining > 0) {
        const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(remaining, buffer.size()));
        scan.read(reinterpret_cast<char*>(buffer.data()), count * sizeof(float));
        if (scan.gcount() != static_cast<std::streamsize>(count * sizeof(float))) {
            throw AudioIoError("truncated output during peak scan");
        }
        for (std::size_t index = 0; index < count; ++index) {
            if (!std::isfinite(buffer[index])) throw AudioIoError("output contains NaN or Inf");
            peak = std::max(peak, std::abs(buffer[index]));
        }
        remaining -= count;
    }
    if (normalize_peak && peak > 0.98f) {
        const auto scale = 0.98f / peak;
        remaining = total_frames_ * channels_;
        std::uint64_t sample_offset = 0;
        while (remaining > 0) {
            const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(remaining, buffer.size()));
            scan.clear();
            scan.seekg(static_cast<std::streamoff>(kWavHeaderSize + sample_offset * sizeof(float)));
            scan.read(reinterpret_cast<char*>(buffer.data()), count * sizeof(float));
            for (std::size_t index = 0; index < count; ++index) buffer[index] *= scale;
            scan.clear();
            scan.seekp(static_cast<std::streamoff>(kWavHeaderSize + sample_offset * sizeof(float)));
            scan.write(reinterpret_cast<const char*>(buffer.data()), count * sizeof(float));
            remaining -= count;
            sample_offset += count;
        }
    }
    scan.flush();
    scan.close();
    std::error_code ignored;
    std::filesystem::remove(output_path_, ignored);
    std::filesystem::rename(temporary_path_, output_path_);
    finalized_ = true;
}

AudioData read_wav(const std::filesystem::path& path) {
    WavReader reader(path);
    AudioData audio{reader.sample_rate(), reader.channels(), {}};
    std::vector<float> chunk;
    while (reader.read_frames(chunk, 64 * 1024) > 0) {
        audio.samples.insert(audio.samples.end(), chunk.begin(), chunk.end());
    }
    return audio;
}

void write_float_wav_atomic(const std::filesystem::path& path, const AudioData& audio) {
    if (audio.channels == 0 || audio.frame_count() == 0) throw AudioIoError("audio is empty");
    FloatWavMixer mixer(path, audio.sample_rate, audio.channels, audio.frame_count());
    mixer.mix_frames(0, audio.samples, audio.frame_count(), 1.0f);
    mixer.finalize();
}

}  // namespace llm_local
