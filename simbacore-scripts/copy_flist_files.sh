#!/bin/bash

# Configuration 
FLIST_PATH="simbacore-work/flist.tcl"
REMOTE_HOST="cygni"
REMOTE_USER="rgeens"
REMOTE_DIR="ssm-project-work/src"
TEMP_FILE="/tmp/flist_files_$$.txt"
TEMP_FLIST="/tmp/flist_remote_$$.tcl"


# Extract ROOT path from flist.tcl
ROOT_PATH=$(grep -o 'set ROOT "[^"]*"' "$FLIST_PATH" | sed 's/set ROOT "\(.*\)"/\1/')

# Extract file paths and expand $ROOT, filter for HDL files
grep -o '"[^"]*"' "$FLIST_PATH" | sed 's/"//g' | while read -r file_path; do
    # Replace $ROOT with actual path
    if [[ "$file_path" == \$ROOT* ]]; then
        file_path="${file_path/\$ROOT/$ROOT_PATH}"
    fi
    
    # Only process HDL files (including SystemVerilog headers)
    if [[ "$file_path" =~ \.(sv|svh|v|vhd)$ ]]; then
        # Check if file exists
        if [ -f "$file_path" ]; then
            echo "$file_path"
        fi
    fi
done > "$TEMP_FILE"

# Add all files from .bender/*/include directories
echo "Adding files from .bender/*/include directories..."
INCLUDE_FILES="/tmp/include_files_$$.txt"
find "$ROOT_PATH/.bender" -path "*/.bender/*/include/*" -type f 2>/dev/null > "$INCLUDE_FILES"

if [ -s "$INCLUDE_FILES" ]; then
    include_count=$(wc -l < "$INCLUDE_FILES")
    echo "Found $include_count additional files in .bender/*/include directories"
    cat "$INCLUDE_FILES" >> "$TEMP_FILE"
fi

# Add all .svh files from hw directory tree
HW_SVH_FILES="/tmp/hw_svh_files_$$.txt"
HW_DIR="$ROOT_PATH/hw"
if [ -d "$HW_DIR" ]; then
    echo "Adding .svh files from hw directory..."
    find "$HW_DIR" -type f -name "*.svh" 2>/dev/null > "$HW_SVH_FILES"
    if [ -s "$HW_SVH_FILES" ]; then
        hw_svh_count=$(wc -l < "$HW_SVH_FILES")
        echo "Found $hw_svh_count .svh files under hw/"
        cat "$HW_SVH_FILES" >> "$TEMP_FILE"
    else
        rm -f "$HW_SVH_FILES"
    fi
fi

sort -u "$TEMP_FILE" -o "$TEMP_FILE"

file_count=$(wc -l < "$TEMP_FILE")
echo "Total files to transfer: $file_count"

if [ "$file_count" -eq 0 ]; then
    echo "No files to transfer!"
    rm -f "$TEMP_FILE"
    exit 0
fi

if ssh "$REMOTE_USER@$REMOTE_HOST" "test -d $REMOTE_DIR || mkdir -p $REMOTE_DIR 2>/dev/null; test -d $REMOTE_DIR"; then
    echo "Base directory $REMOTE_DIR verified"
else
    echo "Failed to create base directory $REMOTE_DIR"
    exit 1
fi

count=0

# Function to normalize paths by removing ".." components
normalize_path() {
    local path="$1"
    # Remove leading "./" and replace "../" with nothing, also handle "/.." at the end
    echo "$path" | sed 's|^\./||g; s|/\.\./|/|g; s|^\.\./||g; s|/\.\.$||g; s|\.\.$||g'
}

# Create SFTP batch commands file
SFTP_BATCH="/tmp/sftp_batch_$$.txt"
DIRS_CREATED="/tmp/dirs_created_$$.txt"

# Read file list and collect unique directories
while IFS= read -r file_path; do
    # Calculate relative path from ROOT and normalize it
    rel_path="${file_path#$ROOT_PATH/}"
    rel_path=$(normalize_path "$rel_path")
    remote_file_path="$REMOTE_DIR/$rel_path" 
    remote_dir_path=$(dirname "$remote_file_path")
    
    # Add directory to list (will be deduplicated later)
    echo "$remote_dir_path" >> "$DIRS_CREATED"
   
done < "$TEMP_FILE"


# Collect ALL directories (including intermediate ones) that need to be created
ALL_DIRS="/tmp/all_dirs_$$.txt"
> "$ALL_DIRS"  # Clear file

# For each directory, add all its parent directories too
sort -u "$DIRS_CREATED" | while IFS= read -r dir_path; do
    current_path=""
    # Split path and build incrementally
    echo "$dir_path" | tr '/' '\n' | while IFS= read -r part; do
        if [ -n "$part" ]; then
            if [ -z "$current_path" ]; then
                current_path="$part"
            else
                current_path="$current_path/$part"
            fi
            echo "$current_path" >> "$ALL_DIRS"
        fi
    done
done


DIRS_TO_CREATE="/tmp/dirs_to_create_$$.txt"
sort -u "$ALL_DIRS" | awk '{print length($0), $0}' | sort -n | cut -d' ' -f2- > "$DIRS_TO_CREATE"

# Create directories in batches via SSH for efficiency
ssh "$REMOTE_USER@$REMOTE_HOST" "$(
    echo 'while IFS= read -r dir; do'
    echo '  mkdir "$dir" 2>/dev/null || true'
    echo 'done <<'"'EOF'"
    cat "$DIRS_TO_CREATE"
    echo 'EOF'
)"


# Add file transfer commands
while IFS= read -r file_path; do
    # Calculate relative path from ROOT and normalize it
    rel_path="${file_path#$ROOT_PATH/}"
    rel_path=$(normalize_path "$rel_path")
    remote_file_path="$REMOTE_DIR/$rel_path"
    echo "put \"$file_path\" \"$remote_file_path\"" >> "$SFTP_BATCH"
done < "$TEMP_FILE"

# Execute SFTP batch transfer
sftp -b "$SFTP_BATCH" "$REMOTE_USER@$REMOTE_HOST"

if [ $? -eq 0 ]; then
    echo "SFTP transfer completed successfully"
    count=$file_count
else
    echo "SFTP transfer encountered errors"
fi

# Create modified flist.tcl for remote server with normalized paths
{
    # Process the flist.tcl file line by line
    while IFS= read -r line; do
        if [[ "$line" =~ set\ ROOT ]]; then
            # Replace the ROOT path
            echo "set ROOT \"~/$REMOTE_DIR\""
        elif [[ "$line" =~ \$ROOT.*\.\. ]]; then
            # This line contains a path with ".." - normalize it
            normalized_line=$(echo "$line" | sed 's|\$ROOT/\.\./|\$ROOT/|g; s|\$ROOT\.\./|\$ROOT/|g; s|/\.\./|/|g; s|^\.\./||g; s|/\.\.$||g; s|\.\.$||g')
            echo "$normalized_line"
        else
            # Regular line, output as-is
            echo "$line"
        fi
    done < "$FLIST_PATH"
} > "$TEMP_FLIST"

# Transfer the modified flist.tcl
echo "put \"$TEMP_FLIST\" \"$REMOTE_DIR/flist.tcl\"" | sftp -b - "$REMOTE_USER@$REMOTE_HOST"

echo "Successfully transferred $count files to $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR. Remote flist.tcl ROOT set to: /home/$REMOTE_USER/$REMOTE_DIR"

# Clean up
rm -f "$TEMP_FILE" "$SFTP_BATCH" "$TEMP_FLIST" "$DIRS_CREATED" "$ALL_DIRS" "$DIRS_TO_CREATE" "$INCLUDE_FILES" "$HW_SVH_FILES"
