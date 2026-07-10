#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
  printf 'usage: %s MODEL_DIR OUTPUT_DIR\n' "$0" >&2
  exit 2
fi

MODEL_DIR="$1"
OUTPUT_DIR="$2"
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
BIN="${GEMMA4_MAC_SANDBOX_BIN:-}"

if [ -z "$BIN" ]; then
  BIN="$(swift build --package-path "$SCRIPT_DIR" -c release --show-bin-path)/Gemma4MacSandbox"
fi

mkdir -p "$OUTPUT_DIR"

for mode in resident paged; do
  for color in red blue; do
    for run in 1 2 3; do
      RECEIPT="$OUTPUT_DIR/${mode}-${color}-${run}.json"
      rm -f "$RECEIPT"
      "$BIN" run \
        --mode "$mode" \
        --color "$color" \
        --model-dir "$MODEL_DIR" \
        --output "$RECEIPT"
    done
  done
done
