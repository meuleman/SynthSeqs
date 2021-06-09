#!/bin/bash

rm out.txt
rm tomtom_out/*
for file in ../../memes/*/*.meme
do 
    tomtom -thresh 0.05 -eps $file filters.meme
    cat tomtom_out/tomtom.txt | tail -n +2 >> out.txt
done
