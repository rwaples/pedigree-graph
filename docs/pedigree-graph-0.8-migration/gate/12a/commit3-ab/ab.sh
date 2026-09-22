set -e
OLD=/tmp/claude-1000/-data-Documents-simACE/929d0db5-ba2d-4ea4-9bee-8ec7c889a87e/scratchpad/old-target/release/pgr-bench-pairs
NEW=/data/Documents/simACE/external/pedigree-graph/target/release/pgr-bench-pairs
TSV=/tmp/claude-1000/-data-Documents-simACE/929d0db5-ba2d-4ea4-9bee-8ec7c889a87e/scratchpad/qual/random_300k.tsv
OUT=/tmp/claude-1000/-data-Documents-simACE/929d0db5-ba2d-4ea4-9bee-8ec7c889a87e/scratchpad/ab.jsonl
: > $OUT
for rep in 0 1 2; do for t in 1 6; do
  for arm in "old buffered" "new speed" "old two_pass" "new memory"; do
    set -- $arm
    if [ $1 = old ]; then B=$OLD; F=--emitter; else B=$NEW; F=--execution; fi
    line=$($B $TSV $F $2 --threads $t --max-degree 5 2>/dev/null)
    echo "{\"rep\": $rep, \"arm\": \"$1 $2\", \"threads\": $t, \"record\": $line}" >> $OUT
    echo "[$rep] t$t $1 $2 done"
  done
done; done
echo AB_DONE
