# Run inside the project venv. This replaces CPU-only torch with a CUDA-enabled build.
# If this index is unavailable for your environment, use the exact CUDA wheel selected by the official PyTorch installer.
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
