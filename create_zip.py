"""
Create a zip file for uploading to Google Colab.
Excludes models, large data files, and unnecessary directories.
"""

import zipfile
import os
from pathlib import Path

def should_exclude(path):
    """Check if a path should be excluded from the zip."""
    exclude_patterns = [
        'models/lightgbm/',  # Exclude trained models, not source
        'models/sentence_bert/',  # Exclude trained models, not source
        'models_old/',
        'venv/',
        'ENV/',
        'env/',
        '.venv/',
        '.git/',
        '__pycache__/',
        '.pytest_cache/',
        '.ipynb_checkpoints/',
        'data/synthetic_transactions_10k.csv',
        'data/synthetic_transactions_50k.csv',
        'data/synthetic_transactions_100k.csv',
        'data/synthetic_transactions_200k.csv',
        '.pyc',
        '.pyo',
        '.egg-info',
        'holmes_ai.zip',
        'demo_output.txt',
        'nul'
    ]

    path_str = str(path).replace('\\', '/')

    # Special case: Always include src/models/ source code
    if 'src/models/' in path_str and path_str.endswith('.py'):
        return False

    for pattern in exclude_patterns:
        if pattern in path_str:
            return True

    return False

def create_zip():
    """Create zip file."""
    print("Creating holmes_ai.zip...")
    print("Excluding: models, large data files, venv, .git, __pycache__")
    print()

    root_dir = Path('.')
    zip_path = 'holmes_ai.zip'

    # Remove existing zip if it exists
    if os.path.exists(zip_path):
        os.remove(zip_path)

    file_count = 0

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in root_dir.rglob('*'):
            if file_path.is_file() and not should_exclude(file_path):
                arcname = str(file_path.relative_to(root_dir))
                zipf.write(file_path, arcname)
                file_count += 1
                if file_count % 10 == 0:
                    print(f"Added {file_count} files...", end='\r')

    zip_size = os.path.getsize(zip_path) / (1024 * 1024)  # Size in MB

    print(f"\n\nSuccess! Created holmes_ai.zip")
    print(f"Total files: {file_count}")
    print(f"Zip size: {zip_size:.2f} MB")
    print(f"\nReady to upload to Google Drive!")

if __name__ == "__main__":
    create_zip()
