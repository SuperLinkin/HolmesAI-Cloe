"""List contents of holmes_ai.zip to verify all files are included."""

import zipfile
from pathlib import Path

zip_path = "holmes_ai.zip"

print("=" * 80)
print(f"Contents of {zip_path}")
print("=" * 80)
print()

with zipfile.ZipFile(zip_path, 'r') as zipf:
    all_files = zipf.namelist()

    # Filter for important files
    src_models = [f for f in all_files if f.startswith('src/models/')]
    verify_script = [f for f in all_files if 'verify_split_fix.py' in f]
    bug_fix_docs = [f for f in all_files if 'BUG_FIX' in f or 'COLAB_RETRAIN' in f]

    print(f"Total files: {len(all_files)}")
    print()

    print("src/models/ files (MUST HAVE 3 .py files):")
    for f in sorted(src_models):
        info = zipf.getinfo(f)
        size_kb = info.file_size / 1024
        print(f"  {f} ({size_kb:.1f} KB)")
    print()

    print("Verification script (NEW):")
    for f in verify_script:
        print(f"  {f}")
    print()

    print("Bug fix documentation (NEW):")
    for f in sorted(bug_fix_docs):
        print(f"  {f}")
    print()

    # Check for the critical file
    critical_file = 'src/models/lightgbm_classifier.py'
    if critical_file in all_files:
        print(f"[OK] {critical_file} is in the zip (contains the fix)")

        # Extract and check if it has the fix
        content = zipf.read(critical_file).decode('utf-8')
        if 'Split data ONCE using L1 stratification' in content:
            print(f"[OK] Bug fix detected in {critical_file}")
            print("     Comment found: 'Split data ONCE using L1 stratification'")
        else:
            print(f"[WARNING] Bug fix comment NOT found in {critical_file}")
            print("          This zip may not contain the fix!")
    else:
        print(f"[ERROR] {critical_file} is MISSING from the zip!")

    print()
    print("=" * 80)
    print("ZIP VERIFICATION COMPLETE")
    print("=" * 80)
