#!/bin/bash

conversation_file="data/shopping_companion_s_cleaned.jsonl"
index_dir="data/mem_indexes"
sentence_model_name="all-MiniLM-L6-v2"

mkdir -p logs ${index_dir}
rm -rf ${index_dir}/*

# 1. build mem indexes
python src/build_mem_indexes.py --conversation_file ${conversation_file} --index_dir ${index_dir} --sentence_model_name ${sentence_model_name}
if [ $? -eq 0 ]; then
    echo "build memory indexes success"
else
    exit 1
fi

# 2. serve mem
ps aux | grep serve_mem | grep -v grep | awk '{print $2}' | xargs kill -9
PORT=5632
nohup python src/serve_mem.py --index_dir ${index_dir} --sentence_model_name ${sentence_model_name} --port ${PORT} > logs/serve_mem 2>&1 &

while true; do
  # 检查端口是否被占用（使用 ss 或 lsof）
  if ss -tuln | grep -q ":$PORT "; then
    echo "Port $PORT is now in use. Exiting wait loop."
    break
  fi
  echo "Port $PORT is free. Waiting..."
  sleep 2
done

# 3. split train and test
python src/split_train_test.py --ref_file ${conversation_file} --test_size 100 --valid_question_types single_product,add_on_deals

# 4. test tools
python src/test_tools.py mem
