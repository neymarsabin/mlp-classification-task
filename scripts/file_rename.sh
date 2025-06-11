#!/bin/bash

# Directory containing the files
TARGET_DIR="$1"

# Check if directory is provided and exists
if [ -z "$TARGET_DIR" ] || [ ! -d "$TARGET_DIR" ]; then
  echo "Usage: $0 /path/to/directory"
  exit 1
fi

# Loop through each file in the directory
for file in "$TARGET_DIR"/*; do
  # Only process regular files (ignore directories)
  if [ -f "$file" ]; then
    dir=$(dirname "$file")
    base=$(basename "$file")
    name="${base%.*}"
    ext="${base##*.}"

    # Handle files without an extension
    if [ "$name" = "$base" ]; then
      new_name="${name}hc"
    else
      new_name="${name}hc.${ext}"
    fi

    mv "$file" "$dir/$new_name"
    echo "Renamed: $base → $new_name"
  fi
done

echo "✅ All file names updated by appending 's'."

