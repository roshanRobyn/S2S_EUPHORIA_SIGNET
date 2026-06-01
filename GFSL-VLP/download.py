from huggingface_hub import snapshot_download

print("Downloading raw files straight to disk (bypassing RAM spikes)...")
snapshot_download(
    repo_id="facebook/mbart-large-cc25",
    local_dir="./mbart_local",
    local_dir_use_symlinks=False,  # Fixes the Windows symlink warning
    ignore_patterns=["*.msgpack", "*.h5", "*.ot"]  # Skip unnecessary non-PyTorch files
)
print("✅ Model successfully downloaded and saved to ./mbart_local")