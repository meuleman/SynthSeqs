#!/bin/bash

seqs=$1
motif=$2
fimo --bgfile --uniform-- $motif $seqs
rm out.txt
touch out.txt
mv fimo_out/fimo.txt .
tail -n +2 fimo.txt >> out.txt
rm fimo.txt
rm -r fimo_out 
