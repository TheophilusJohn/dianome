#!/usr/bin/env bash
# PDF of docs/writeup.md with pandoc (Theo runs it; pandoc + a LaTeX engine are not installed on the dev Mac).
#   brew install pandoc basictex      # or tectonic: brew install tectonic, then PDF_ENGINE=tectonic
#   bash docs/writeup-pdf.sh          # -> docs/writeup.pdf
# Runs scripts/writeup-check.mjs first, so a write-up with an uncited number never becomes a PDF.
set -euo pipefail
cd "$(dirname "$0")/.."
node scripts/writeup-check.mjs docs/writeup.md
ENGINE="${PDF_ENGINE:-xelatex}"
pandoc docs/writeup.md \
  --from gfm+footnotes \
  --to pdf --pdf-engine="$ENGINE" \
  --lua-filter docs/pandoc/repo-links.lua \
  --metadata title="Dianome: a model CDN and split inference for the browser" \
  --metadata author="Theophilus John" \
  --metadata date="$(date -u +%Y-%m-%d) · $(git rev-parse --short HEAD)" \
  --variable geometry:margin=2.2cm --variable fontsize=10pt --variable colorlinks=true --variable linkcolor=blue --variable urlcolor=blue \
  --toc --toc-depth=2 \
  --output docs/writeup.pdf
echo "wrote docs/writeup.pdf"
