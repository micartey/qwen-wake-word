#!/usr/bin/env bash
set -euo pipefail

devices=()

while IFS= read -r line; do
    card=$(echo "$line" | sed -n 's/^card \([0-9]*\):.*/\1/p')
    device=$(echo "$line" | sed -n 's/.*device \([0-9]*\):.*/\1/p')
    description=$(echo "$line" | sed -n 's/^card [0-9]*: [^ ]* \[\(.*\)\], device [0-9]*: .*/\1/p')

    if [[ -n "$card" && -n "$device" ]]; then
        devices+=("hw:${card},${device}|${description}")
    fi
done < <(arecord -l 2>/dev/null | grep '^card')

if [[ ${#devices[@]} -eq 0 ]]; then
    echo "No capture devices found." >&2
    exit 1
fi

echo "Available capture devices:"
echo ""

for i in "${!devices[@]}"; do
    hw="${devices[$i]%%|*}"
    desc="${devices[$i]#*|}"
    printf "  [%d] %-10s %s\n" "$i" "$hw" "$desc"
done

echo ""
read -rp "Select device [0-$((${#devices[@]} - 1))]: " choice

if ! [[ "$choice" =~ ^[0-9]+$ ]] || [[ "$choice" -ge ${#devices[@]} ]]; then
    echo "Invalid selection." >&2
    exit 1
fi

selected="${devices[$choice]%%|*}"

cat << EOF > ~/.asoundrc
pcm.!default {
    type asym
    capture.pcm "mic_plug"
}

pcm.mic_plug {
    type plug
    slave.pcm "${selected}"
}
EOF

echo "Wrote ~/.asoundrc with capture device: ${selected}"
