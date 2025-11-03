from dotenv import load_dotenv
import os
from huggingface_hub import HfApi, upload_file, hf_hub_download
from datetime import datetime
from tadaconv.utils.misc import find_dotenv_in_parents



env_path = find_dotenv_in_parents()
if env_path:
    load_dotenv(env_path)

def upload_checkpoint_to_huggingface(cfg, checkpoint_path):
    repo_id = cfg.HUGGINGFACE.REPO
    try:
        hf_api = HfApi(token=os.getenv("HuggingFace"))
        hf_api.create_repo(repo_id=repo_id, exist_ok=True)
        files_in_repo = hf_api.list_repo_files(repo_id)
    except Exception as e:
        print(f"Error checking files: {e}")
        return None
    date_time_str = datetime.now().strftime("%d-%m-%H%M")
    repo_file_name = cfg.HUGGINGFACE.FILE_NAME + date_time_str + ".pyth"
    assert repo_file_name not in files_in_repo, f"File {repo_file_name} already exists in repo {repo_id}"

    upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo=repo_file_name,
        repo_id=repo_id,
        token=os.getenv("HuggingFace"),
    )

def download_checkpoint_from_huggingface(cfg, hf_identifier, checkpoint_path):
    repo_id = cfg.HUGGINGFACE.REPO
    return hf_hub_download(repo_id=repo_id, filename=hf_identifier, cache_dir=checkpoint_path, token=os.getenv("HuggingFace"))


def hf_ckpt_available(cfg):
    repo_id = cfg.HUGGINGFACE.REPO
    try:
        hf_api = HfApi(token=os.getenv("HuggingFace"))
        hf_api.create_repo(repo_id=repo_id, exist_ok=True)
        files_in_repo = hf_api.list_repo_files(repo_id)
    except Exception as e:
        print(f"Error checking files: {e}")
        return None
    ckpts = [f for f in files_in_repo if cfg.HUGGINGFACE.FILE_NAME in f]
    if len(ckpts) == 0:
        return None
    ckpt = sorted(ckpts, reverse=True)[0]
    return ckpt
    
    
    