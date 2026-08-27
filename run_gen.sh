#!/bin/bash

index_dir="data/product_indexes"
products_file="data/products.jsonl"
preferences_dir="data/preferences"
longmemeval_file="data/longmemeval_s_cleaned.json"
output_file="data/shopping_companion_s_cleaned.jsonl"
max_workers=100

mkdir -p logs ${preferences_dir}
rm -rf ${preferences_dir}/*

# 1. generate dialogue
python src/dialogue_gen.py --index_dir ${index_dir} --products_file ${products_file} --preferences_dir ${preferences_dir} --max_workers ${max_workers}
if [ $? -eq 0 ]; then
    echo "generate dialogue success"
else
    exit 1
fi

# 2. generate question
python src/question_gen.py --preferences_dir ${preferences_dir} --longmemeval_file ${longmemeval_file} --output_file ${output_file} --max_workers ${max_workers}
if [ $? -eq 0 ]; then
    echo "generate question success"
else
    exit 1
fi
