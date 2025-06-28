# Sentience

A production-grade cognition engine that brings your MacBook to life. Sentience loads Google's Gemma 3n, a multimodal transformer model, and performs continuous perception and reasoning based on webcam video and microphone audio input.

## Overview

Sentience is an offline cognition engine that:
- Captures webcam frames and microphone audio at ~5Hz
- Processes images and audio using Google's int4-quantized Gemma 3n model
- Produces a continuous stream of scene descriptions and action recommendations
- Processes visual and auditory information simultaneously
- Operates autonomously without requiring user prompts

## System Requirements

- Apple Silicon MacBook (M1/M2/M3)
- macOS 13 or newer
- Python 3.10 or 3.11
- At least 4GB of free RAM (recommended 8GB+)
- Integrated GPU (uses Apple MPS backend)
- Working webcam and microphone (permissions will be requested)

## Installation

1. Create and activate a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Upgrade pip and install dependencies:
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Run Sentience:
   ```
   python -m sentience
   ```

   Additional command-line options:
   ```
   # Run without audio processing
   python -m sentience --no-audio
   
   # Run in test mode (synthetic inputs)
   python -m sentience --test
   
   # Set logging verbosity
   python -m sentience --log-level DEBUG
   ```

**Note:** On first run, Sentience will automatically download the Gemma 3n model (approximately 5GB) into the `weights/gemma_e2b_int4` directory. This download happens only once and the model will be stored locally for future use. The system will verify that you have sufficient memory before attempting to download.

## Performance Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Cold load | <9s | Time from Python start to first thought output |
| Memory usage | ≤3.0 GB | Steady-state RAM consumption after 10 iterations |
| Throughput | ≥4 thoughts/sec | Sustained output rate for 60+ seconds |
| Stability | 30+ min | Continuous operation without errors |

Exact performance may vary depending on your specific MacBook model and system load.

## Build Information

- Package: `sentience-0.1.0-py3-none-macosx_11_0_arm64.whl`
- SHA256: `[SHA will be added after build completion]`

To build the wheel file:
```
python setup.py bdist_wheel
```

## Architectural Components

- **runtime.py**: Device detection, model loading, continuous inference loop
- **vision.py**: Camera capture and image preprocessing
- **audio.py**: Microphone capture and audio processing
- **model_interface.py**: Gemma 3n multimodal model wrapper
- **streamer.py**: Thought formatting and output
- **downloader.py**: Asset management and model downloading
- **assets/mission.txt**: Core mission prompt that drives autonomous reasoning

## Acknowledgements & Licensing

### Code License
This project is licensed under the Apache License 2.0.

### Gemma 3n Model
The Gemma 3n model is used in accordance with Google's model usage terms. Gemma is a family of lightweight, state-of-the-art open models from Google based on the same research and technology used to create the Gemini models.

```
@article{gemma2024,
    title = {Gemma: Open Models Based on Gemini Research and Technology},
    author = {Google Gemma Team},
    year = {2024}
}
```

## Important Notes

- Sentience operates fully offline without any telemetry or data collection
- Performance is optimized specifically for Apple Silicon using PyTorch MPS
- For any issues or questions, please file a bug report
