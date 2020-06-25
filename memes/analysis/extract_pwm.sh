#!/bin/bash

for memefile in */*.meme;
do
	tmp=${memefile%.*}
	awk 'FNR >= 13 {print $1,$2,$3,$4}' $memefile > "$tmp.txt"
done	

