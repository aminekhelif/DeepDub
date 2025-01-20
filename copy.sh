#!/usr/bin/env bash

# Source directory (the one containing FILM1, FILM2, etc.)
SRC="/path/to/data"
# Destination directory where you want the filtered copy
DEST="/path/to/data_copy"

# Make sure the destination directory exists
mkdir -p "$DEST"

# Loop over each directory named FILM* inside $SRC
for film_dir in "$SRC"/FILM*/; do
  # Skip if there's no such directory
  [ -d "$film_dir" ] || continue

  # Get just the last part of the path (e.g. "FILM1", "FILM2", etc.)
  film_name=$(basename "$film_dir")

  echo "Copying $film_name ..."

  # Use rsync to copy, excluding unwanted .mp3 files
  rsync -av \
    --exclude='VF.mp3' \
    --exclude='VO_anglais.mp3' \
    --exclude='VF/output_directory/VF_audio.mp3' \
    --exclude='VF/output_directory/instrumental.mp3' \
    --exclude='VO_anglais/output_directory/VO_anglais_audio.mp3' \
    --exclude='VO_anglais/output_directory/instrumental.mp3' \
    "$film_dir" "$DEST/$film_name"

  echo "Finished copying $film_name."
  echo
done

echo "All FILM* folders copied to $DEST, skipping specified files."