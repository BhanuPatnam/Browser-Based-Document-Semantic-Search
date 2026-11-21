---
project_name: Speech Recognition and Edge Voice AI with Whisper
project_category: Audio / NLP
document_type: model_card
tags: whisper, asr, speech-to-text, audio-processing, wer, voice-ai, edge-computing
url: https://github.com/BhanuPatnam/audio-nlp-projects/tree/main/10-speech-recognition-whisper
---
# Speech Recognition and Edge Voice AI with Whisper

## Objective
Deploy a robust automatic speech recognition (ASR) pipeline for voice-driven industrial operations and hands-free laboratory maintenance. The system transcribes spoken technical jargon, equipment part numbers, and error codes under noisy acoustic conditions.

## Acoustic modeling and decoding
The system leverages OpenAI Whisper (base and small checkpoints) fine-tuned on specialized technical voice corpora. Key components include:
- Audio feature extraction using 80-channel log-Mel filterbanks resampled at 16 kHz.
- Voice Activity Detection (VAD) via Silero VAD to strip ambient background hum and compressor noise.
- Temperature fallback and beam search decoding to minimize repetition loops and phantom hallucinated phrases during silent pauses.

## Evaluation and edge deployment
The pipeline is evaluated on test recordings with synthetic industrial factory noise SNR ranges (0 dB to 20 dB):
- Word Error Rate (WER) and Character Error Rate (CER) on domain-specific vocabulary.
- Real-Time Factor (RTF) across CPU, CUDA, and Apple Silicon Metal Performance Shaders (MPS).
- Memory-efficient edge deployment packaged via whisper.cpp for offline, air-gapped embedded device execution.
