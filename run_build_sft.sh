#!/bin/bash

rs_files="data/reject_sample_two_stage_gpt-4.1-2025-04-14-GlobalStandard_train.jsonl,data/reject_sample_two_stage_gpt-4o-2024-11-20_train.jsonl"
sft_file="data/sft_data.json"
model_name="Qwen/Qwen3-4B-Instruct-2507"
debug=True

python src/build_sft.py --rs_files ${rs_files} --sft_file ${sft_file} --model_name ${model_name} --debug ${debug} > logs/build_sft 2>&1 &
