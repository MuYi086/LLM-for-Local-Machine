#!/usr/bin/env python3
"""Mel-Band RoFormer audio.cpp HTTP 服务入口。"""

from audio_cpp_runtime import run_service

if __name__ == "__main__":
    run_service("mel_band_roformer")
