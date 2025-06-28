from setuptools import setup

setup(
    name="sentience",
    version="0.1.0",
    author="Cascade AI",
    description="A multimodal AI agent for real-time perception and action.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=['sentience'],
    package_dir={'sentience': 'core'},
    package_data={
        'sentience': ['assets/*'],
    },
    include_package_data=True,
    install_requires=[
        'torch>=2.3.0',
        'transformers>=4.44.0',
        'accelerate>=0.29.0',
        'pillow>=10.3.0',
        'opencv-python>=4.9.0',
        'psutil>=5.9.0',
        'huggingface_hub>=0.20.0',
        'pyaudio>=0.2.13',
        'tqdm>=4.66.1',
        'torchvision>=0.18.0',
        'timm>=0.9.12',
    ],
    entry_points={
        'console_scripts': [
            'sentience = sentience.runtime:run',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
    ],
    python_requires='>=3.9',
)
