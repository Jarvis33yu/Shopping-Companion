method=two_stage
model=qwen3-4b-thinking-2507_lora64
base_model=qwen3-4b-thinking-2507
hyp_file=data/rollout_${method}_${model}.jsonl
ref_file=data/shopping_companion_s_cleaned_test.jsonl
index_dir=data/product_indexes/
threads=5

# 1. run agent
python src/two_stage_agent.py \
    --hyp_file ${hyp_file} \
    --ref_file ${ref_file} \
    --model ${base_model} \
    --threads ${threads} \
    --base_url http://127.0.0.1:8000/v1 \
    --api_key "None" \
> logs/rollout_${method}_${model} 2>&1

if [ $? -eq 0 ]; then
    echo "build memory indexes success"
else
    exit 1
fi

# 2. evaluate
python src/evaluate.py --hyp_file ${hyp_file} --ref_file ${ref_file} --index_dir ${index_dir} --max_workers 100 --debug True