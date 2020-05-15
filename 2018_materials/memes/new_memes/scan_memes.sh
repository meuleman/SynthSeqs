#!/bin/bash

seqs=$1
rm out.txt
touch out.txt
for motif in *.meme;
do
	pattern=$motif
	fimo --bgfile --uniform-- $pattern $seqs
	mv fimo_out/fimo.txt .
	tail -n +2 fimo.txt >> out.txt
	rm fimo.txt
	rm -r fimo_out
done	

