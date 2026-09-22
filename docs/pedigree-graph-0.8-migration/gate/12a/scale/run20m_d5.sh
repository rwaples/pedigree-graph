cd /data/Documents/simACE/external/pedigree-graph
/usr/bin/time -v target/release/pgr-bench-pairs /tmp/claude-1000/-data-Documents-simACE/929d0db5-ba2d-4ea4-9bee-8ec7c889a87e/scratchpad/scale/pedsum_20M.tsv --emitter two_pass --threads 6 --max-degree 5 > /tmp/claude-1000/-data-Documents-simACE/929d0db5-ba2d-4ea4-9bee-8ec7c889a87e/scratchpad/scale/20m.two_pass.d5.json 2> /tmp/claude-1000/-data-Documents-simACE/929d0db5-ba2d-4ea4-9bee-8ec7c889a87e/scratchpad/scale/20m.two_pass.d5.err
echo "exit $?"
grep -E "Elapsed|Maximum resident|panicked|could not allocate|Killed" /tmp/claude-1000/-data-Documents-simACE/929d0db5-ba2d-4ea4-9bee-8ec7c889a87e/scratchpad/scale/20m.two_pass.d5.err
echo D5_DONE
