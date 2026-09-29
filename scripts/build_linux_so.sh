#!/usr/bin/env bash
# Compiles the MATLAB Coder sources (matlab/s6_generate_linux_sources.m) into
# artifacts/linux64/evaluatePolicy.so with g++. No MATLAB needed on this machine.
#
#   bash scripts/build_linux_so.sh            # e.g. inside WSL
#
# Usage: build_linux_so.sh [source_dir] [output_dir]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="${1:-$root/build/linux_src}"
out="${2:-$root/artifacts/linux64}"

[ -d "$src" ] || { echo "Source directory $src not found. Run matlab/s6_generate_linux_sources.m first." >&2; exit 1; }
command -v g++ >/dev/null || { echo "g++ not found (Ubuntu: sudo apt install g++)." >&2; exit 1; }
mkdir -p "$out"

# Export only what Windows exports through evaluatePolicy.def: the entry point
# and the *_initialize / *_terminate lifecycle functions. Everything else stays
# local to the library.
map="$(mktemp)"
trap 'rm -f "$map"' EXIT
cat > "$map" <<'EOF'
{ global: *evaluatePolicy*; local: *; };
EOF

g++ -std=c++11 -O2 -fPIC -shared \
    -DBUILDING_EVALUATEPOLICY \
    -I"$src" \
    "$src"/*.cpp \
    -Wl,--version-script="$map" \
    -o "$out/evaluatePolicy.so"

echo "Built $out/evaluatePolicy.so"
echo "Dynamic dependencies:"
ldd "$out/evaluatePolicy.so" | sed 's/^/  /'
echo "Exported symbols:"
nm -D --defined-only "$out/evaluatePolicy.so" | sed 's/^/  /'
