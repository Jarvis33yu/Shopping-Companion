set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"

: "${MODEL_PATH:?Set MODEL_PATH to the base or SFT model directory}"
: "${TRAIN_FILE:=$PROJECT_DIR/preprocessed_data/train.parquet}"
: "${VAL_FILE:=$PROJECT_DIR/preprocessed_data/test.parquet}"
: "${OUTPUT_DIR:=$PROJECT_DIR/checkpoints}"
: "${NUM_GPUS:=8}"

function now() {
    date '+%Y-%m-%d-%H-%M'
}

EXPERIMENT_NAME="qwen3-4b-sft-lora_prm_orm_n8_$(now)"


export MLFLOW_TRACKING_URI=sqlite:////tmp/mlruns.db
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR" \
    --config-name='agentic_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    custom_reward_function.path=$PROJECT_DIR/reward_async.py \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.trace.backend=mlflow \
    actor_rollout_ref.rollout.trace.token2text=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb","mlflow"]' \
    trainer.project_name=shopping_companion \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.default_local_dir="$OUTPUT_DIR/$EXPERIMENT_NAME" \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/agentic_tools.yaml" \
    trainer.total_epochs=20 $@
