#!/bin/sh
SOURCE_DIR=$1
DEST_DIR=$2
MAX_SIZE=$(awk "BEGIN {printf \"%d\", $3 * 1024 * 1024}")
TAR_NAME=$(basename "$SOURCE_DIR")
mkdir -p "$DEST_DIR"
create_tar() {
  tar_number=$1
  file_list=$2
  tar_name=$(printf "%s/${TAR_NAME}_%0${width}d.tar.gz" "$DEST_DIR" "$tar_number")
  tar -zcvf "$tar_name" -C "$SOURCE_DIR" -T "$file_list"
}
tar_number=1
current_size=0
temp_dir=$(mktemp -d)
file_list="$temp_dir/file_list_$tar_number"
echo Start indexing "file_list_$tar_number"
cd "$SOURCE_DIR" || exit 1
find . -type f | while IFS= read -r file; do
  file_size=$(du -k "$file" | cut -f1)
  if [ $(( current_size + file_size )) -gt $MAX_SIZE ]; then
    tar_number=$((tar_number + 1))
    file_list="$temp_dir/file_list_$tar_number"
    echo Start indexing "file_list_$tar_number"
    current_size=0
  fi
  echo "$file" >> "$file_list"
  current_size=$((current_size + file_size))
done
total_files=$(find "$temp_dir" -name 'file_list_*' | wc -l)
width=${
PARALLEL_PROCESSES=${4:-$total_files}
echo "Total files: $total_files"
echo "Width: $width"
echo "Parallel processes: $PARALLEL_PROCESSES"
find "$temp_dir" -name 'file_list_*' | xargs -P "$PARALLEL_PROCESSES" -I {} sh -c '
  file_list={}
  tar_number=$(basename "$file_list" | cut -d_ -f3)
  tar_name=$(printf "%s/'"$TAR_NAME"'_%0'"$width"'d.tar.gz" "'"$DEST_DIR"'" "$tar_number")
  tar -zcvf "$tar_name" -C "'"$SOURCE_DIR"'" -T "$file_list"
'
rm -rf "$temp_dir"