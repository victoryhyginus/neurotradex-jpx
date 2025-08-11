#!/usr/bin/env bash
# Tidy notebooks/: move HTML reports to reports/ and big PNGs to images/
# Usage: ./cleanup_notebooks.sh [PNG_SIZE_KB]   (default 200)

set -euo pipefail

PNG_LIMIT_KB="${1:-200}"

NB_DIR="notebooks"
REPORTS_DIR="reports"
IMAGES_DIR="images"

echo "📁 Ensuring target folders exist…"
mkdir -p "$REPORTS_DIR" "$IMAGES_DIR"

if [ ! -d "$NB_DIR" ]; then
  echo "❌ '$NB_DIR' not found. Run from repo root."; exit 1
fi

echo "🧹 Moving HTML reports from $NB_DIR → $REPORTS_DIR"
# Works on macOS' old bash too
found_html=0
find "$NB_DIR" -maxdepth 1 -type f -name "*.html" -print0 | while IFS= read -r -d '' f; do
  found_html=1
  echo "  • $(basename "$f")"
  git mv -f "$f" "$REPORTS_DIR/" 2>/dev/null || mv -f "$f" "$REPORTS_DIR/"
done
[ "$found_html" -eq 0 ] && echo "  (none)"

echo "🖼  Moving large PNGs (> ${PNG_LIMIT_KB} KB) from $NB_DIR → $IMAGES_DIR"
found_png=0
# BSD find on macOS supports -size +N[k]
find "$NB_DIR" -maxdepth 1 -type f -name "*.png" -size +"${PNG_LIMIT_KB}"k -print0 | while IFS= read -r -d '' f; do
  found_png=1
  # stat is different on macOS vs linux; try mac first, then linux
  sz=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo "?")
  printf "  • %s  (%s bytes)\n" "$(basename "$f")" "$sz"
  git mv -f "$f" "$IMAGES_DIR/" 2>/dev/null || mv -f "$f" "$IMAGES_DIR/"
done
[ "$found_png" -eq 0 ] && echo "  (none)"

echo "✅ Done."
echo "   Kept: *.ipynb and small assets in $NB_DIR"
echo "   Moved: *.html → $REPORTS_DIR, large PNGs → $IMAGES_DIR"
echo
echo "🔎 git status (review before commit):"
git status --porcelain || true

