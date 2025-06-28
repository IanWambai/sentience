# Sentience

A production-grade cognition engine that brings your MacBook to life. Sentience loads Google's Gemma 3n, a multimodal transformer model, and performs continuous perception and reasoning based on webcam input.

## Overview

Sentience is an offline cognition engine that:
- Captures webcam frames at ~5Hz
- Processes images using Google's int4-quantized Gemma 3n model
- Produces a continuous stream of scene descriptions and action recommendations
- Operates autonomously without requiring user prompts

## System Requirements

- Apple Silicon MacBook (M1/M2/M3)
- macOS 13 or newer
- Python 3.10 or 3.11
- At least 4GB of free RAM
- Integrated GPU (uses Apple MPS backend)

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

## Performance Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Cold load | <9s | Time from Python start to first thought output |
| Memory usage | ≤2.5 GB | Steady-state RAM consumption after 10 iterations |
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

- **runtime.py**: Device detection, model loading, continuous loop
- **vision.py**: Camera capture and image preprocessing
- **model_interface.py**: Gemma 3n model wrapper
- **streamer.py**: Thought formatting and output
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
