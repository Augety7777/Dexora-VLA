import argparse
import sys
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub.utils import get_token
import concurrent.futures

def upload_task_with_retry(repo_id: str, task_dir: Path, path_in_repo: str, max_retries: int = 3):
    """Uploads a single task folder with retries."""
    api = HfApi()
    # Use get_token() which works across versions
    token = get_token() 
    if not token:
        raise ValueError("Hugging Face token not found. Please login using 'huggingface-cli login'.")

    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}: Uploading {task_dir.name}...")
            api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(task_dir),
                path_in_repo=path_in_repo,
                commit_message=f"Upload task: {task_dir.name}",
                ignore_patterns=["*.jsonl"],  # Ignore mapping files as requested
                allow_patterns=None, # Ensure no conflicting patterns
                token=token,
            )
            print(f"✔ Successfully uploaded {task_dir.name}")
            return task_dir.name, True
        except Exception as e:
            print(f"  [ERROR] Attempt {attempt + 1} failed for {task_dir.name}: {e}")
            if attempt + 1 == max_retries:
                print(f"❌ Failed to upload {task_dir.name} after {max_retries} attempts.")
                return task_dir.name, False
            print("  Retrying...")
    return task_dir.name, False


def check_already_uploaded(repo_id: str, path_in_repo: str) -> bool:
    """Checks if a folder already exists in the repo."""
    api = HfApi()
    try:
        # list_repo_tree is a generator, we just need to see if it yields anything
        next(api.list_repo_tree(repo_id=repo_id, repo_type="dataset", path_in_repo=path_in_repo))
        return True
    except StopIteration:
        # The generator is empty, so the path does not exist
        return False
    except Exception:
        # Assume it's not uploaded on other errors to be safe
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload Dexora dataset tasks to Hugging Face Hub.")
    parser.add_argument("repo_id", type=str, help="Repository ID, e.g., 'Dexora/Dexora_Real-World_Dataset'")
    parser.add_argument("data_root", type=str, help="Path to the root of the dexora dataset (the one containing task folders).")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for uploading.")

    args = parser.parse_args()

    repo_id = args.repo_id
    data_root = Path(args.data_root)

    if not data_root.is_dir():
        print(f"[ERROR] Data root not found: {data_root}")
        sys.exit(1)

    all_tasks = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name != "logs"])

    if not all_tasks:
        print(f"[WARNING] No task directories found in {data_root}")
        sys.exit(0)

    print(f"Found {len(all_tasks)} tasks to potentially upload to {repo_id}.")
    print("-" * 30)

    tasks_to_upload = []
    print("Checking which tasks are already on the Hub...")
    for task_dir in all_tasks:
        path_in_repo = f"dexora/{task_dir.name}"
        if check_already_uploaded(repo_id, path_in_repo):
            print(f"  - Skipping {task_dir.name}, already exists in repo.")
        else:
            tasks_to_upload.append(task_dir)
            print(f"  - Queued {task_dir.name} for upload.")

    if not tasks_to_upload:
        print("\nAll tasks are already uploaded. Nothing to do.")
        sys.exit(0)

    print(f"\nStarting upload for {len(tasks_to_upload)} tasks using {args.workers} workers...")
    print("-" * 30)

    successful_uploads = []
    failed_uploads = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {
            executor.submit(upload_task_with_retry, repo_id, task_dir, f"dexora/{task_dir.name}"): task_dir
            for task_dir in tasks_to_upload
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task_name, success = future.result()
            if success:
                successful_uploads.append(task_name)
            else:
                failed_uploads.append(task_name)

    print("\n" + "=" * 30)
    print("Upload summary:")
    print(f"  Successfully uploaded: {len(successful_uploads)}")
    print(f"  Failed to upload: {len(failed_uploads)}")
    if failed_uploads:
        print("  Failed tasks:", ", ".join(failed_uploads))
    print("=" * 30)

if __name__ == "__main__":
    main()







