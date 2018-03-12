#!/usr/bin/env bash

n=0
for i in $( ls in*.txt ); do
    ./program.py $i > out$n.txt &
    n=$((n+1))
done

wait
