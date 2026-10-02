#!/usr/bin/env bash
# Build <team>_submission.zip in the layout required by the challenge:
#   output/{matching_results,candidate_pairs}.tsv
#   code/business_entity_resolution/{src,tests,README.md,requirements.txt}
#   Documentation_template.md
# The output files are validated with the official validator before zipping.
#
# usage: scripts/package_submission.sh <team_name> [output_dir=output] [dataset_dir=student_resource/dataset]
set -euo pipefail

TEAM=${1:?usage: $0 <team_name> [output_dir] [dataset_dir]}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${2:-$ROOT/output}
DATA=${3:-$ROOT/student_resource/dataset}
VALIDATOR=$(dirname "$DATA")/utils/validate_submission.py

python3 "$VALIDATOR" --matching "$OUT/matching_results.tsv" --candidate "$OUT/candidate_pairs.tsv" \
  --test-dir "$DATA/test" --check-ids

STAGE=$(mktemp -d)
trap 'rm -r "$STAGE"' EXIT
PKG="$STAGE/code/business_entity_resolution"
mkdir -p "$STAGE/output" "$PKG"

cp "$OUT/matching_results.tsv" "$OUT/candidate_pairs.tsv" "$STAGE/output/"
rsync -a --exclude "__pycache__" --exclude ".pytest_cache" --exclude ".DS_Store" --exclude "*.egg-info" \
  "$ROOT/src" "$ROOT/tests" "$PKG/"
cp "$ROOT/requirements.txt" "$PKG/"
cp "$ROOT/docs/submission_readme.md" "$PKG/README.md"
cp "$ROOT/docs/methodology.md" "$STAGE/Documentation_template.md"

ZIP="$ROOT/${TEAM}_submission.zip"
rm -f "$ZIP"
(cd "$STAGE" && zip -qr "$ZIP" output code Documentation_template.md)
echo "wrote $ZIP"
