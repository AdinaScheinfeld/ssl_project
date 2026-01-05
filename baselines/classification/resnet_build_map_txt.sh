OUT_ROOT=/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_resnet
JSON_DIR=$OUT_ROOT
MAP_FILE=$OUT_ROOT/fold_map.txt
rm -f "$MAP_FILE"
idx=0
for JSON in $(ls -1 "$JSON_DIR"/cls_folds_*.json | sort -V); do
  NF=$(python - <<PY
import json
j=json.load(open("$JSON"))
print(len(j["folds"]))
PY
)
  for ((f=0; f<NF; f++)); do
    echo "$idx|$JSON|$f" >> "$MAP_FILE"
    idx=$((idx+1))
  done
done
echo "Total=$idx"
