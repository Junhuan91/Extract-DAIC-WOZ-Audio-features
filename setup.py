from setuptools import setup, find_packages

setup(
    name="daic-audio-features",
    version="0.1.0",
    packages=find_packages(),
    package_dir={"": "."},
    install_requires=[
        "torch",
        "torchaudio",
        "transformers",
        "librosa",
        "numpy",
        "pandas",
        "tqdm",
        "pyyaml",
        "scikit-learn",
        "speechbrain",
        "openai-whisper"
    ],
    python_requires=">=3.7",
)
