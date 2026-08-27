#!/bin/bash

resources_dir="data/product_resources"
indexes_dir="data/product_indexes"
odps_table="${ODPS_TABLE:?Set ODPS_TABLE to the product source table and partition}"
products_file="data/products.jsonl"
docs_file="${resources_dir}/docs.jsonl"

mkdir -p logs ${resources_dir} ${indexes_dir}
rm -rf ${resources_dir}/*
rm -rf ${indexes_dir}/*

# 1. build product docs
python src/build_product_docs.py --tasks download,convert --odps_table ${odps_table} --products_file ${products_file} --docs_file ${docs_file}
if [ $? -eq 0 ]; then
    echo "build product docs success"
else
    exit 1
fi

# 2. build product indexes
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ${resources_dir} \
  --index ${indexes_dir} \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storePositions --storeDocvectors --storeRaw
if [ $? -eq 0 ]; then
    echo "build product indexes success"
else
    exit 1
fi

# 3. serve product
ps aux | grep serve_product | grep -v grep | awk '{print $2}' | xargs kill -9
PORT=5631
nohup python src/serve_product.py --index_dir ${indexes_dir} --port ${PORT} > logs/serve_product 2>&1 &

while true; do
  # 检查端口是否被占用（使用 ss 或 lsof）
  if ss -tuln | grep -q ":$PORT "; then
    echo "Port $PORT is now in use. Exiting wait loop."
    break
  fi
  echo "Port $PORT is free. Waiting..."
  sleep 2
done

# 4. test tools
python src/test_tools.py product
