#!/bin/bash
mkdir experiments &> /dev/null
cd experiments
for size in 5 10 15 20 25
do
    mkdir "${size}_4"
    cd "${size}_4"
    python ../../ReAI.py $size 4 > record.txt
    cd ..
done

for clusters in 10 25 50 100
do
    echo $clusters
    mkdir "25_${clusters}"
    cd "25_${clusters}"
    python ../../ReAI.py 25 $clusters  > record.txt
    cd ..
done
