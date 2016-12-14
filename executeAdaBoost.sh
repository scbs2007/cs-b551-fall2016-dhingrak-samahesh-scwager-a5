#!/bin/sh
stumpCount=( 1 2 3 4 5 6 7 8 9 10 11 25 50 75 100 )
adatrain='../cv_set4_750/adatrain_750_'
count=0
while [ $count -lt 15 ]
do
    echo "${stumpCount[$count]}"
    file=$adatrain
    file+=${stumpCount[$count]}
    python -m cProfile -s time orient.py ../validationSet4/trainFile4 ../validationSet4/testFile4 adaboost ${stumpCount[$count]}  > $file & 
    count=`expr $count + 1` 
done
