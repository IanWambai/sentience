"""
Setup script for Sentience package.
"""

from setuptools import setup, find_packages

setup(
    name="sentience",
    version="0.1.0",
    packages=['sentience'],
    package_dir={'sentience': 'core'},
    package_data={'sentience': ['assets/mission.txt']},
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
    ],
    python_requires='>=3.9',
    platforms=["MacOS"],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: MacOS',
    ],
    description="Sentience - A production-grade multimodal cognition engine for Apple Silicon MacBooks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Sentience Team",
    entry_points={
        'console_scripts': [
            'sentience=sentience.runtime:run',
        ],
    },
)
