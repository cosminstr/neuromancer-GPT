import os
from pathlib import Path
from typing import Optional

import modal
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# create a Volume, or retrieve it if it exists
volume = modal.Volume.from_name("hf", create_if_missing=True)
MODEL_DIR = Path("/models")

# define dependencies for downloading model
download_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub", "python-dotenv")
    .env(
        {"HF_TOKEN": os.environ["HF_TOKEN"], "HF_XET_HIGH_PERFORMANCE": "1"}
    )  # enable fast data transfer
)
app = modal.App()


@app.function(
    volumes={
        MODEL_DIR.as_posix(): volume
    },  # "mount" the Volume, sharing it with your function
    image=download_image,  # only download dependencies needed here
    timeout=60 * 60 * 24,
)
def download_model(
    repo_id: str = "openai/gpt-oss-20b",
    revision: Optional[str] = None,  # include a revision to prevent surprises!
):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, local_dir=MODEL_DIR / repo_id, revision=revision)
    print(f"Model downloaded to {MODEL_DIR / repo_id}")
    volume.commit()
