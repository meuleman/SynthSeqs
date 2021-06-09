#!/bin/bash

motif=$1
rm hits.txt 
touch hits.txt
for iteration in *.fasta;
do
    fimo --bgfile --uniform-- --thresh 1e-5 /home/pbromley/projects/synth_seqs/memes/vascular_endothelial/ERG_ETS_2.meme $iteration
    hits="$(awk -F '\t' '{print $3}' fimo_out/fimo.txt | sort | uniq -c | tail -n +2 | wc -l)"
    echo -e "$iteration\t$hits" >> hits.txt
done
