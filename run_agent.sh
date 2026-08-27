#!/bin/bash

ref_file="data/shopping_companion_s_cleaned_test.jsonl"
methods=(two_stage one_stage oracle missing_or_wrong feature_names)
models=(gpt-5-2025-08-07-GlobalStandard gpt-4.1-2025-04-14-GlobalStandard gpt-4o-2024-11-20 qwen3-max qwen3-next-80b-a3b-thinking qwen3-30b-a3b-thinking-2507)
threads=5
index_dir="data/product_indexes"

# 1. run agent
for method in ${methods[@]}; do
    for model in ${models[@]}; do

        if [[ ${model} != "gpt-5-2025-08-07-GlobalStandard" && ${method} != "two_stage" ]]; then
            continue
        fi

        hyp_file="data/rollout_${method}_${model}.jsonl"

        echo "Running agent for ${method} ${model}:"
        if [ ${method} == "two_stage" ]; then
            python src/two_stage_agent.py --hyp_file ${hyp_file} --ref_file ${ref_file} --model ${model} --threads ${threads} > logs/rollout_${method}_${model} 2>&1 &
        elif [ ${method} == "one_stage" ]; then
            python src/one_stage_agent.py --hyp_file ${hyp_file} --ref_file ${ref_file} --model ${model} --threads ${threads} > logs/rollout_${method}_${model} 2>&1 &
        elif [ ${method} == "oracle" ]; then
            python src/oracle_agent.py --hyp_file ${hyp_file} --ref_file ${ref_file} --model ${model} --threads ${threads} > logs/rollout_${method}_${model} 2>&1 &
        elif [ ${method} == "missing_or_wrong" ]; then
            python src/user_sim_agent.py --hyp_file ${hyp_file} --ref_file ${ref_file} --user_mode missing_or_wrong --model ${model} --threads ${threads} > logs/rollout_${method}_${model} 2>&1 &
        elif [ ${method} == "feature_names" ]; then
            python src/user_sim_agent.py --hyp_file ${hyp_file} --ref_file ${ref_file} --user_mode feature_names --model ${model} --threads ${threads} > logs/rollout_${method}_${model} 2>&1 &
        else
            echo "Invalid method: ${method}"
            exit 1
        fi
    done
done

wait

# 2. evaluate
for method in ${methods[@]}; do
    for model in ${models[@]}; do
        hyp_file="data/rollout_${method}_${model}.jsonl"
        echo "Evaluating ${method} ${model}:"
        python src/evaluate.py --hyp_file ${hyp_file} --ref_file ${ref_file} --index_dir ${index_dir} --max_workers 100 --debug True
    done
done
