# Sentience: Autonomous Multimodal AI Cognition Engine

A production-grade, offline cognition engine that brings your MacBook to life through continuous perception and reasoning. Sentience loads Google's Gemma 3n multimodal transformer model and performs real-time analysis of webcam video and microphone audio input, providing autonomous scene understanding and action recommendations.

## 🧠 Overview

Sentience is an advanced autonomous AI system that:

- **Captures multimodal input** at ~5Hz from webcam and microphone simultaneously
- **Processes visual and auditory information** using Google's int4-quantized Gemma 3n-E2B model
- **Generates continuous insights** through scene descriptions and action recommendations
- **Operates fully offline** without requiring internet connectivity or user prompts
- **Optimized for Apple Silicon** using PyTorch MPS backend for optimal performance
- **Provides real-time streaming** of thoughts and observations with timestamped output

### Core Capabilities

- **Visual Perception**: Analyzes webcam frames to understand visual context, objects, people, activities, and environmental conditions
- **Audio Analysis**: Processes microphone input to detect sounds, speech, ambient noise, and audio events
- **Multimodal Fusion**: Combines visual and auditory information for comprehensive scene understanding
- **Autonomous Reasoning**: Generates insights and recommendations without user intervention
- **Continuous Operation**: Maintains persistent awareness and tracks environmental changes over time

## 🖥️ System Requirements

### Hardware Requirements
- **Processor**: Apple Silicon MacBook (M1/M2/M3 series)
- **Memory**: Minimum 8GB RAM (16GB+ recommended for optimal performance)
- **Storage**: At least 10GB free space for model download and operation
- **Camera**: Built-in or external webcam with macOS compatibility
- **Microphone**: Built-in or external microphone with macOS compatibility

### Software Requirements
- **Operating System**: macOS 13 (Ventura) or newer
- **Python**: Version 3.10 or 3.11 (3.12 not yet supported due to dependency compatibility)
- **Permissions**: Camera and microphone access permissions will be requested on first run

### Performance Expectations
- **Cold Start**: <9 seconds from launch to first thought output
- **Memory Usage**: ≤3.0 GB steady-state RAM consumption
- **Throughput**: ≥4 thoughts per second sustained output rate
- **Stability**: 30+ minutes continuous operation without errors
- **Model Size**: ~2GB Gemma 3n-E2B model (automatically downloaded)

## 🚀 Installation

### Prerequisites
Ensure you have Python 3.10 or 3.11 installed on your system. You can check your Python version with:
```bash
python3 --version
```

### Step-by-Step Installation

1. **Clone or download the repository**:
   ```bash
   git clone <repository-url>
   cd sentience
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Upgrade pip and install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run Sentience**:
   ```bash
   python -m sentience
   ```

### First Run Experience

On first run, Sentience will:
1. **Check system compatibility** (Apple Silicon, macOS version, memory)
2. **Download the Gemma 3n model** (~2GB, takes 10-30 minutes depending on connection)
3. **Request camera and microphone permissions** (required for operation)
4. **Initialize the AI engine** and begin continuous perception

The model download happens only once and is stored locally in the `weights/gemma_e2b` directory for future use.

## 🎮 Usage

### Basic Operation
```bash
# Standard operation with both video and audio
python -m sentience

# Run without audio processing (visual-only mode)
python -m sentience --no-audio

# Run in test mode with synthetic inputs (no camera/microphone required)
python -m sentience --test

# Set logging verbosity for debugging
python -m sentience --log-level DEBUG
```

### Command Line Options

| Option | Description | Use Case |
|--------|-------------|----------|
| `--no-audio` | Disable audio input processing | When microphone is unavailable or unwanted |
| `--test` | Use synthetic test inputs | Development, testing, or when hardware unavailable |
| `--log-level` | Set logging verbosity | Debugging and performance monitoring |

### Output Format

Sentience provides real-time streaming output in the following format:
```
[HH:MM:SS.mmm] SCENE: [Detailed scene description] | ACTION: [Recommended action]
```

Example output:
```
[14:23:45.123] SCENE: I can see a person sitting at a desk working on a laptop. The room appears to be a home office with natural lighting from a window. I can hear the sound of typing on a keyboard. | ACTION: Continue monitoring for any changes in activity or environment.
[14:23:45.456] SCENE: The person has stopped typing and is now looking at their phone. The room lighting remains consistent. I can hear a notification sound from the phone. | ACTION: Note the change in activity and continue observing for further interactions.
```

## 🏗️ Architecture

### Core Components

#### 1. **Runtime System** (`core/runtime.py`)
- **Device Detection**: Automatically detects Apple Silicon and configures MPS backend
- **System Initialization**: Handles model loading, camera setup, and audio initialization
- **Main Loop**: Orchestrates the continuous perception-inference cycle
- **Error Handling**: Robust error recovery and graceful degradation
- **Performance Monitoring**: Tracks throughput, memory usage, and system health

#### 2. **Vision Processing** (`core/vision.py`)
- **Camera Management**: Handles webcam access with OpenCV and fallback methods
- **Frame Capture**: Grabs frames at ~5Hz with consistent timing
- **Image Preprocessing**: Converts frames to tensor format for model input
- **Error Recovery**: Handles camera disconnections and permission issues
- **Test Mode**: Provides synthetic image generation for testing

#### 3. **Audio Processing** (`core/audio.py`)
- **Microphone Capture**: Non-blocking audio capture using PyAudio
- **Audio Buffering**: Maintains rolling 1-second buffer of recent audio
- **Silence Detection**: Filters out silent periods to reduce processing load
- **Format Conversion**: Converts audio to 16kHz mono format for model input
- **Device Management**: Handles audio device selection and permissions

#### 4. **AI Model Interface** (`core/model_interface.py`)
- **Gemma 3n Integration**: Wraps the Google Gemma 3n-E2B multimodal model
- **Hybrid Execution**: Optimized CPU/MPS execution strategy for Apple Silicon
- **Multimodal Processing**: Handles simultaneous visual and audio inputs
- **Scene Description**: Generates detailed textual descriptions of observed scenes
- **Action Planning**: Produces actionable recommendations based on context
- **Performance Optimization**: Implements efficient token generation and caching

#### 5. **Thought Streaming** (`core/streamer.py`)
- **Output Formatting**: Formats and timestamps all AI outputs
- **Color Coding**: Provides colored terminal output for different types of information
- **Error Handling**: Graceful handling of output errors
- **Throughput Monitoring**: Tracks and reports performance metrics

#### 6. **Asset Management** (`core/downloader.py`)
- **Model Download**: Automatic downloading of Gemma 3n model from Hugging Face
- **System Validation**: Checks hardware compatibility and memory requirements
- **Progress Tracking**: Shows download progress with estimated completion time
- **Integrity Verification**: Validates downloaded model files
- **Error Recovery**: Handles network issues and incomplete downloads

### Technical Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Camera Feed   │    │  Audio Stream   │    │   Mission File  │
│   (5Hz frames)  │    │  (16kHz audio)  │    │  (Core prompt)  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Gemma 3n Multimodal Model                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Vision Tower│  │ Audio Tower │  │   Language Model (MPS)  │  │
│  │   (CPU)     │  │   (CPU)     │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Thought Processing                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Scene Description│  │ Action Planning │  │   Output Stream │  │
│  │                 │  │                 │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Performance Optimization

#### Apple Silicon Optimization
- **MPS Backend**: Uses Metal Performance Shaders for GPU acceleration
- **Hybrid Execution**: Vision/audio towers on CPU, language model on MPS GPU
- **Memory Management**: Efficient tensor placement and transfer strategies
- **Quantization**: Uses int4-quantized model for reduced memory footprint

#### Model Architecture
- **Gemma 3n-E2B**: 2B parameter model optimized for multimodal tasks
- **256-dim Attention**: Compatible with current MPS SDPA kernel limitations
- **Multimodal Fusion**: True multimodal processing with shared attention
- **Efficient Inference**: Optimized token generation and caching strategies

## ⚙️ Configuration

### Environment Variables
```bash
# Disable MPS fallback for optimal performance
export PYTORCH_ENABLE_MPS_FALLBACK=0

# Set logging level
export SENTIENCE_LOG_LEVEL=INFO
```

### Model Configuration
The system automatically uses the optimal model configuration for your hardware:
- **E2B Model**: 2B parameters, 256-dim attention heads (MPS compatible)
- **Future E4B Support**: Will automatically upgrade when PyTorch adds 512-dim support

### Performance Tuning
- **Frame Rate**: Adjustable from 1-10 Hz (default: 5 Hz)
- **Audio Buffer**: Configurable buffer size (default: 1 second)
- **Memory Usage**: Automatic memory management with configurable limits

## 🐛 Troubleshooting

### Common Issues

#### Camera Permission Denied
```bash
# Reset camera permissions
sudo killall VDCAssistant
# Then restart Sentience
```

#### Microphone Not Working
```bash
# Check audio permissions in System Preferences
# Ensure microphone access is granted to Terminal/your Python environment
```

#### Model Download Fails
```bash
# Check internet connection
# Verify sufficient disk space (at least 10GB free)
# Try manual download from Hugging Face
```

#### Memory Issues
```bash
# Close other applications
# Ensure at least 8GB RAM available
# Check Activity Monitor for memory usage
```

#### Performance Issues
```bash
# Run with --log-level DEBUG to identify bottlenecks
# Check CPU/GPU usage in Activity Monitor
# Ensure no other GPU-intensive applications are running
```

### Debug Mode
```bash
# Enable detailed logging
python -m sentience --log-level DEBUG

# Run in test mode to isolate hardware issues
python -m sentience --test
```

## 📊 Performance Metrics

### Target Performance
| Metric | Target | Description |
|--------|--------|-------------|
| Cold Start | <9s | Time from launch to first thought |
| Memory Usage | ≤3.0 GB | Steady-state RAM consumption |
| Throughput | ≥4 thoughts/sec | Sustained output rate |
| Stability | 30+ min | Continuous operation time |
| Model Load | <5s | Time to load model into memory |

### Real-World Performance
Actual performance may vary based on:
- **MacBook Model**: M1/M2/M3 performance differences
- **System Load**: Background applications and processes
- **Environment**: Lighting conditions and audio complexity
- **Model Warmup**: First few generations may be slower

## 🔒 Privacy & Security

### Data Handling
- **Local Processing**: All processing occurs on-device, no data leaves your MacBook
- **No Telemetry**: No usage data collection or analytics
- **No Cloud Dependencies**: Fully offline operation after initial model download
- **Temporary Storage**: Audio and video data not permanently stored

### Permissions
- **Camera Access**: Required for visual perception
- **Microphone Access**: Required for audio analysis
- **File System**: Read access for model files, write access for downloads

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd sentience

# Create development environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Code Structure
```
sentience/
├── core/                    # Main application code
│   ├── __init__.py
│   ├── runtime.py          # Main runtime and entry point
│   ├── model_interface.py  # AI model wrapper
│   ├── vision.py           # Camera and image processing
│   ├── audio.py            # Microphone and audio processing
│   ├── streamer.py         # Output formatting and streaming
│   ├── downloader.py       # Model and asset management
│   └── assets/             # Static assets
│       ├── __init__.py
│       └── mission.txt     # Core mission prompt
├── assets/                 # Public assets
│   └── placeholder.jpg     # Test image
├── weights/                # Model storage (auto-created)
├── requirements.txt        # Python dependencies
├── setup.py               # Package configuration
└── README.md              # This file
```

### Testing
```bash
# Run in test mode
python -m sentience --test

# Run with debug logging
python -m sentience --log-level DEBUG --test
```

## 📄 License

### Code License
This project is licensed under the Apache License 2.0.

### Model License
The Gemma 3n model is used in accordance with Google's model usage terms. Gemma is a family of lightweight, state-of-the-art open models from Google based on the same research and technology used to create the Gemini models.

### Citation
If you use Sentience in your research or projects, please cite:
```
@article{gemma2024,
    title = {Gemma: Open Models Based on Gemini Research and Technology},
    author = {Google Gemma Team},
    year = {2024}
}
```

## 🙏 Acknowledgements

- **Google Gemma Team**: For the excellent multimodal model
- **Hugging Face**: For model hosting and transformers library
- **PyTorch Team**: For MPS backend and Apple Silicon support
- **OpenCV**: For computer vision capabilities
- **PyAudio**: For audio capture functionality

## 📞 Support

### Getting Help
- **Documentation**: This README contains comprehensive usage information
- **Debug Mode**: Use `--log-level DEBUG` for detailed error information
- **Test Mode**: Use `--test` flag to isolate hardware issues
- **Issues**: File bug reports with detailed system information and error logs

### System Information
When reporting issues, please include:
- macOS version
- MacBook model (M1/M2/M3)
- Python version
- Available RAM
- Error messages and logs

---

**Sentience** - Bringing autonomous AI cognition to your MacBook. Experience the future of personal AI assistants, running entirely on your device with no cloud dependencies.
