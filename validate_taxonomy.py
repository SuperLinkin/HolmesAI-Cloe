"""
Taxonomy Validation Tool

Helps admins validate taxonomy.json before deploying changes.
Provides clear error messages for common mistakes.
"""

import json
import sys
from pathlib import Path


def validate_taxonomy(taxonomy_path: str = "src/config/taxonomy.json"):
    """
    Validate taxonomy JSON file.

    Returns:
        (bool, str): (is_valid, error_message)
    """

    print("=" * 80)
    print("TAXONOMY VALIDATION")
    print("=" * 80)
    print()

    # Check file exists
    if not Path(taxonomy_path).exists():
        return False, f"File not found: {taxonomy_path}"

    # Check valid JSON
    try:
        with open(taxonomy_path, 'r') as f:
            taxonomy = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON syntax: {e}"

    print(f"✅ Valid JSON syntax")

    # Check required top-level fields
    if "categories" not in taxonomy:
        return False, "Missing required field: 'categories'"

    print(f"✅ Has 'categories' field")

    # Validate categories
    categories = taxonomy["categories"]
    if not isinstance(categories, list):
        return False, "'categories' must be a list"

    if len(categories) == 0:
        return False, "No categories defined"

    print(f"✅ Found {len(categories)} L1 categories")

    # Validate each L1 category
    l1_names = set()
    l1_ids = set()

    for i, cat in enumerate(categories, 1):
        # Required L1 fields
        if "l1" not in cat:
            return False, f"Category {i}: Missing 'l1' field"

        l1_name = cat["l1"]

        if "l1_id" not in cat:
            return False, f"Category '{l1_name}': Missing 'l1_id' field"

        l1_id = cat["l1_id"]

        # Check uniqueness
        if l1_name in l1_names:
            return False, f"Duplicate L1 category: '{l1_name}'"

        if l1_id in l1_ids:
            return False, f"Duplicate L1 ID: '{l1_id}'"

        l1_names.add(l1_name)
        l1_ids.add(l1_id)

        # Validate L2 subcategories
        if "l2_subcategories" not in cat:
            return False, f"Category '{l1_name}': Missing 'l2_subcategories'"

        l2_subcats = cat["l2_subcategories"]
        if not isinstance(l2_subcats, list):
            return False, f"Category '{l1_name}': 'l2_subcategories' must be a list"

        if len(l2_subcats) == 0:
            return False, f"Category '{l1_name}': No L2 subcategories defined"

        # Validate each L2
        l2_names = set()

        for j, l2_cat in enumerate(l2_subcats, 1):
            if "l2" not in l2_cat:
                return False, f"Category '{l1_name}', L2 {j}: Missing 'l2' field"

            l2_name = l2_cat["l2"]

            if "l2_id" not in l2_cat:
                return False, f"Category '{l1_name}/{l2_name}': Missing 'l2_id'"

            if l2_name in l2_names:
                return False, f"Category '{l1_name}': Duplicate L2 '{l2_name}'"

            l2_names.add(l2_name)

            # Validate L3 types
            if "l3_types" not in l2_cat:
                return False, f"Category '{l1_name}/{l2_name}': Missing 'l3_types'"

            l3_types = l2_cat["l3_types"]
            if not isinstance(l3_types, list):
                return False, f"Category '{l1_name}/{l2_name}': 'l3_types' must be a list"

            if len(l3_types) == 0:
                return False, f"Category '{l1_name}/{l2_name}': No L3 types defined"

            # Validate each L3
            l3_names = set()

            for k, l3_cat in enumerate(l3_types, 1):
                if "l3" not in l3_cat:
                    return False, f"Category '{l1_name}/{l2_name}', L3 {k}: Missing 'l3' field"

                l3_name = l3_cat["l3"]

                if "l3_id" not in l3_cat:
                    return False, f"Category '{l3_name}': Missing 'l3_id'"

                if l3_name in l3_names:
                    return False, f"Category '{l1_name}/{l2_name}': Duplicate L3 '{l3_name}'"

                l3_names.add(l3_name)

                # Validate optional fields
                if "aliases" in l3_cat:
                    if not isinstance(l3_cat["aliases"], list):
                        return False, f"Category '{l3_name}': 'aliases' must be a list"

                if "mcc_codes" in l3_cat:
                    if not isinstance(l3_cat["mcc_codes"], list):
                        return False, f"Category '{l3_name}': 'mcc_codes' must be a list"

                if "keywords" in l3_cat:
                    if not isinstance(l3_cat["keywords"], list):
                        return False, f"Category '{l3_name}': 'keywords' must be a list"

    print(f"✅ All L1 categories valid ({len(l1_names)} unique)")
    print()

    # Summary statistics
    print("=" * 80)
    print("TAXONOMY SUMMARY")
    print("=" * 80)
    print()

    total_l2 = 0
    total_l3 = 0

    for cat in categories:
        l2_count = len(cat["l2_subcategories"])
        total_l2 += l2_count

        for l2_cat in cat["l2_subcategories"]:
            l3_count = len(l2_cat["l3_types"])
            total_l3 += l3_count

    print(f"L1 Categories: {len(categories)}")
    print(f"L2 Categories: {total_l2}")
    print(f"L3 Categories: {total_l3}")
    print()

    print("✅ Taxonomy is valid!")
    print()

    return True, "Validation successful"


if __name__ == "__main__":
    is_valid, message = validate_taxonomy()

    if not is_valid:
        print("❌ VALIDATION FAILED")
        print(f"   {message}")
        sys.exit(1)
    else:
        print("=" * 80)
        print("SUCCESS! Taxonomy is ready to use.")
        print("=" * 80)
        sys.exit(0)
