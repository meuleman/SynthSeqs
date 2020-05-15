#!/bin/bash

seqs=$1
rm out.txt
touch out.txt
for motif in *.meme;
do
        pattern=$motif
        fimo --bgfile --uniform-- $pattern $seqs
        mv fimo_out/fimo.txt .
        tail -n +2 fimo.txt > fimo_tmp.txt
        awk '{ arr[$2]+=1; arrr[$2]+=$6 } END { for (key in arr) printf("%s\t%s\t%s\t'$pattern'\n", key, arr[key], arrr[key]/arr[key]) }' fimo_tmp.txt | sort +0n -1 >> out.txt
        rm fimo.txt
        rm -r fimo_out
        rm fimo_tmp.txt
done
