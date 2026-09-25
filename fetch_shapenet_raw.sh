#!/bin/bash
# Download the raw Umetani ShapeNet-Car release (2.03 GB) to
# external/shapenet_car/mlcfd_data.zip. Unpacking: REPRODUCING.md step 1.
#
# The server returns no data on an unranged GET and refuses parallel
# connections, so the file is fetched serially in 50 MB ranged chunks.
# Resumable: complete chunks are skipped on re-run.
#
#   ./fetch_shapenet_raw.sh [dest_dir]
set -u
URL="http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip"
TOTAL=2029047879                      # verified against the origin's Content-Length
CHUNK=$((50 * 1024 * 1024))
DEST="${1:-external/shapenet_car}"

mkdir -p "$DEST/parts" || exit 1
cd "$DEST" || exit 1

i=0; off=0
while [ $off -lt $TOTAL ]; do
  end=$((off + CHUNK - 1)); [ $end -ge $TOTAL ] && end=$((TOTAL - 1))
  want=$((end - off + 1))
  f=parts/p$(printf %04d $i)
  for a in 1 2 3 4 5 6 7 8; do
    [ -f "$f" ] && [ "$(stat -c%s "$f")" = "$want" ] && break
    curl -s -A "Mozilla/5.0" --max-time 900 -r "${off}-${end}" -o "$f" "$URL"
    [ -f "$f" ] && [ "$(stat -c%s "$f")" = "$want" ] && break
    echo "chunk $i attempt $a got $(stat -c%s "$f" 2>/dev/null) want $want" >> dl.log
    sleep 5
  done
  if [ "$(stat -c%s "$f" 2>/dev/null)" != "$want" ]; then
    echo "FAILED chunk $i" >> dl.log; exit 1
  fi
  echo "chunk $i ok ($((end + 1))/$TOTAL)" >> dl.log
  off=$((end + 1)); i=$((i + 1))
done

cat parts/p* > mlcfd_data.zip
got=$(stat -c%s mlcfd_data.zip)
if [ "$got" != "$TOTAL" ]; then
  echo "ASSEMBLY SIZE MISMATCH: $got != $TOTAL" >> dl.log; exit 1
fi
rm -rf parts
echo "ASSEMBLED size=$got" >> dl.log
echo DONE >> dl.log
