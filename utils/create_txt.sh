#!/bin/bash

dataset_path="$1"

if [  -n $dataset_path ]; then
    cd $dataset_path
fi

if [ -d image ] && [ -d label ]; then
    if [ -f train_list.txt ]; then
        echo "File train_list.txt has existed."
    else
        find image -type f | sort > train.kidn.tmp
        find label -type f | sort > train.lab.kidn.tmp
        paste -d " " train.kidn.tmp train.lab.kidn.tmp > all.kidn.tmp

        awk '{if (NR % 50 != 0) print $0}' all.kidn.tmp > train_list.txt
        awk '{if (NR % 50 == 0) print $0}' all.kidn.tmp > val_list.txt

        rm *.kidn.tmp
        echo "Create train_list.txt and val_list.txt"
    fi
fi

