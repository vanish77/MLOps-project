#!/bin/bash
# Script to setup Google Drive as DVC remote

set -e

echo "========================================="
echo "Setting up Google Drive as DVC Remote"
echo "========================================="

# Check if Google Drive folder exists (try different possible paths)
DVC_FOLDER="MLOps-DVC-Storage"

# Try different possible Google Drive paths
if [ -d "$HOME/GoogleDrive" ]; then
    GDRIVE_PATH="$HOME/GoogleDrive"
elif [ -d "$HOME/Google Drive" ]; then
    GDRIVE_PATH="$HOME/Google Drive"
elif [ -d "/Users/ivanevgenyevich/GoogleDrive" ]; then
    GDRIVE_PATH="/Users/ivanevgenyevich/GoogleDrive"
else
    echo "? Google Drive folder not found"
    echo ""
    echo "Please install Google Drive Desktop first:"
    echo "  https://www.google.com/drive/download/"
    echo ""
    echo "Or specify the path manually:"
    echo "  export GDRIVE_PATH=/path/to/your/google/drive"
    echo "  bash setup_gdrive_remote.sh"
    exit 1
fi

echo "? Google Drive folder found: $GDRIVE_PATH"

# Create DVC storage folder in Google Drive
DVC_STORAGE_PATH="$GDRIVE_PATH/$DVC_FOLDER"

# Try to create folder
echo "Creating folder: $DVC_STORAGE_PATH"
if mkdir -p "$DVC_STORAGE_PATH" 2>/dev/null; then
    echo "? Created folder: $DVC_STORAGE_PATH"
else
    echo "??  Cannot create folder automatically (permission issue)"
    echo ""
    echo "Please create the folder manually:"
    echo "  1. Open Google Drive in Finder"
    echo "  2. Create a new folder named: $DVC_FOLDER"
    echo "  3. Or run: mkdir -p \"$DVC_STORAGE_PATH\""
    echo ""
    read -p "Press Enter after creating the folder, or Ctrl+C to cancel..."
    
    # Check if folder was created
    if [ ! -d "$DVC_STORAGE_PATH" ]; then
        echo "? Folder still not found. Please create it manually and run the script again."
        exit 1
    fi
    echo "? Folder found: $DVC_STORAGE_PATH"
fi

echo "? Created folder: $DVC_STORAGE_PATH"

# Check if local storage exists
LOCAL_STORAGE="dvc_storage"
if [ -d "$LOCAL_STORAGE" ] && [ "$(ls -A $LOCAL_STORAGE 2>/dev/null)" ]; then
    echo ""
    echo "?? Copying existing data to Google Drive..."
    echo "   This may take a while (current size: $(du -sh $LOCAL_STORAGE | cut -f1))"
    
    # Copy data
    cp -r "$LOCAL_STORAGE"/* "$DVC_STORAGE_PATH/" 2>/dev/null || true
    
    echo "? Data copied to Google Drive"
else
    echo "??  No existing local storage found, skipping copy"
fi

# Setup DVC remote
echo ""
echo "?? Configuring DVC remote..."

# Remove existing gdrive remote if exists
dvc remote remove gdrive 2>/dev/null || true

# Add new remote
dvc remote add gdrive "$DVC_STORAGE_PATH"

# Set as default
dvc remote default gdrive

echo "? DVC remote configured"

# Show configuration
echo ""
echo "========================================="
echo "DVC Remote Configuration:"
echo "========================================="
dvc remote list

echo ""
echo "========================================="
echo "Next steps:"
echo "========================================="
echo "1. Wait for Google Drive Desktop to sync files"
echo "2. Verify sync in Google Drive web interface"
echo "3. Test DVC push:"
echo "   dvc push"
echo "4. Test DVC pull (from another location):"
echo "   dvc pull"
echo ""
echo "?? Google Drive folder: $DVC_STORAGE_PATH"
echo "?? View in browser: https://drive.google.com"
echo "========================================="
