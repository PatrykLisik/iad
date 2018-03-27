#!/usr/bin/env bash

n=0
START=$(date +%s.%N)
for i in $( ls in*.txt ); do
    ./program.py $i out$n.txt >/dev/null &
    n=$((n+1))
done
wait
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo "Ended after $DIFF seconds"
