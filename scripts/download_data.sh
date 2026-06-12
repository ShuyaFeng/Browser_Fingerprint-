#!/bin/bash
# Download browser fingerprint datasets
# Run from the project root: bash scripts/download_data.sh

set -e
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJ_ROOT/data/raw"
mkdir -p "$DATA_DIR/li_cao_imc2020"

echo "=== Downloading Li & Cao IMC 2020 dataset ==="
echo "File size: ~3.7 GB compressed. This will take a while."
echo ""

URL="https://zenodo.org/api/records/7743719/files/final_with_header.csv.zip/content"
DEST="$DATA_DIR/li_cao_imc2020/final_with_header.csv.zip"

if [ -f "$DEST" ]; then
    echo "Already downloaded: $DEST"
else
    curl -L --progress-bar -o "$DEST" "$URL"
    echo "Download complete."
fi

echo ""
echo "=== Extracting ==="
cd "$DATA_DIR/li_cao_imc2020"
if [ -f "final_with_header.csv" ]; then
    echo "Already extracted."
else
    unzip -o final_with_header.csv.zip
    echo "Extraction complete."
fi

echo ""
echo "=== Quick schema check ==="
python3 - <<'PYEOF'
import os, sys
fpath = os.path.join(os.environ.get('DATA_DIR', 'data/raw/li_cao_imc2020'), 'final_with_header.csv')
if not os.path.exists(fpath):
    print(f"File not found: {fpath}")
    sys.exit(1)
with open(fpath) as f:
    header = f.readline().strip()
    row1 = f.readline().strip()
cols = header.split('\t')
print(f"Columns ({len(cols)}): {cols[:10]} ...")
print(f"First row sample: {row1[:200]}")
PYEOF

echo ""
echo "Done. Dataset ready at: $DATA_DIR/li_cao_imc2020/final_with_header.csv"
