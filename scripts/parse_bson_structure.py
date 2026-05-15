#!/usr/bin/env python3
"""
Script to parse and display the structure of a BSON file.

python scripts/parse_bson_structure.py --bson-file-path data/ours/action1/episode_0.bson
"""
from pathlib import Path
from typing import Any

import bson  # uv pip install pymongo
import tyro


def load_bson_file(bson_path: Path) -> dict:
    """Load a bson file and return its contents."""
    with open(bson_path, "rb") as f:
        data = bson.decode(f.read())
    return data


def print_bson_structure(data: Any, indent: str = "") -> None:
    """
    Recursively prints the structure of BSON data.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            type_str = str(type(value))
            if isinstance(value, bytes):
                print(f"{indent}Key: '{key}', Type: {type_str}, Length: {len(value)}")
            else:
                print(f"{indent}Key: '{key}', Type: {type_str}")

            if isinstance(value, dict):
                print_bson_structure(value, indent + "  ")
            elif isinstance(value, list):
                if value:
                    first_item_type_str = str(type(value[0]))
                    print(f"{indent}  List of {len(value)} items, First item type: {first_item_type_str}")
                    # Optionally, print structure of the first item if it's complex
                    if isinstance(value[0], (dict, list)):
                        print(f"{indent}  Structure of first list item:")
                        print_bson_structure(value[0], indent + "    ")
                else:
                    print(f"{indent}  Empty list")
            # Add more specific type handling if needed
    elif isinstance(data, list):
        print(f"{indent}List of {len(data)} items.")
        if data:
            first_item_type_str = str(type(data[0]))
            print(f"{indent}First item type: {first_item_type_str}")
            if isinstance(data[0], (dict, list)):
                print(f"{indent}Structure of first list item:")
                print_bson_structure(data[0], indent + "  ")
    else:
        # For basic data types at the root if the BSON is not a dict
        print(f"{indent}Value: {data}, Type: {type(data)}")


def main(bson_file_path: Path) -> None:
    """
    Main function to load and parse the BSON file.

    Args:
        bson_file_path: Path to the BSON file.
    """
    if not bson_file_path.exists():
        print(f"Error: BSON file not found at {bson_file_path}")
        return

    print(f"Parsing BSON file: {bson_file_path}\n")
    try:
        bson_data = load_bson_file(bson_file_path)
        formatted_list_str = [format(x, '.2f') for x in bson_data['data']["/observation/left_arm/joint_state"][100]["data"]["pos"]]
        print(",".join(formatted_list_str)+",")
        formatted_list_str = [format(x, '.2f') for x in bson_data['data']["/observation/right_arm/joint_state"][100]["data"]["pos"]]
        print(",".join(formatted_list_str)+",")
        exit()
        print_bson_structure(bson_data)
    except Exception as e:
        print(f"An error occurred while parsing the BSON file: {e}")


if __name__ == "__main__":
    tyro.cli(main)
