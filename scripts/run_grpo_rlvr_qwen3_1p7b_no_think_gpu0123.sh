#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/seankopylov/projects/solve-hard-problems"
VERL_DIR="$ROOT/verl-main"
VENV="/home/seankopylov/projects/GRPO/verl/.venv"

source "$VENV/bin/activate"

DATA_SRC="$ROOT/data/run_2000_pass4_pipeline_t1_v2_fast_withhint_only_gpu0123_highutil_v2/final_tasks_with_hints_avg4_mid.jsonl"
DATA_DIR="$ROOT/data/rlvr_hints_onpolicy_grpo_v1"

python "$ROOT/scripts/build_grpo_rlvr_dataset.py" \
  --input-jsonl "$DATA_SRC" \
  --out-dir "$DATA_DIR"

EXP="grpo_qwen3_1p7b_rlvr_hints_no_think_is_4gpu_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/$EXP"
mkdir -p "$RUN_DIR"/{logs,checkpoints,config,tensorboard,val_generations,train_generations}

export HF_HOME="$ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME"
export XDG_CACHE_HOME="$ROOT/.cache"
export RAY_TMPDIR="/tmp/ray_shp"
export TMPDIR="/tmp/tmp_shp"
export TENSORBOARD_DIR="$RUN_DIR/tensorboard"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$RAY_TMPDIR" "$TMPDIR"

TRAIN_BS="${TRAIN_BS:-32}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
SAVE_FREQ="${SAVE_FREQ:-100}"
LR="${LR:-2e-6}"
ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL:-0.55}"
ROLLOUT_MAX_SEQS="${ROLLOUT_MAX_SEQS:-96}"

cat > "$RUN_DIR/config/launch_cmd.sh" <<CMD
cd $VERL_DIR
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_USE_V1=1 PYTHONUNBUFFERED=1 \\
HF_HOME=$HF_HOME \\
HF_DATASETS_CACHE=$HF_DATASETS_CACHE \\
TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE \\
XDG_CACHE_HOME=$XDG_CACHE_HOME \\
RAY_TMPDIR=$RAY_TMPDIR \\
TMPDIR=$TMPDIR \\
TENSORBOARD_DIR=$TENSORBOARD_DIR \\
python -m verl.trainer.main_ppo \\
  algorithm.adv_estimator=grpo \\
  data.train_files=$DATA_DIR/train.parquet \\
  data.val_files=$DATA_DIR/val.parquet \\
  data.return_raw_chat=True \\
  data.train_batch_size=$TRAIN_BS \\
  data.val_batch_size=$TRAIN_BS \\
  data.max_prompt_length=4096 \\
  data.max_response_length=4096 \\
  data.filter_overlong_prompts=True \\
  data.truncation=error \\
  +data.apply_chat_template_kwargs.enable_thinking=False \\
  actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \\
  +actor_rollout_ref.model.override_config.attn_implementation=eager \\
  actor_rollout_ref.model.use_remove_padding=False \\
  actor_rollout_ref.model.enable_gradient_checkpointing=True \\
  actor_rollout_ref.actor.optim.lr=$LR \\
  actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BS \\
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \\
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \\
  actor_rollout_ref.actor.use_kl_loss=True \\
  actor_rollout_ref.actor.kl_loss_coef=0.003 \\
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \\
  actor_rollout_ref.rollout.name=vllm \\
  actor_rollout_ref.rollout.mode=async \\
  actor_rollout_ref.rollout.agent.num_workers=4 \\
  actor_rollout_ref.rollout.n=$ROLLOUT_N \\
  actor_rollout_ref.rollout.temperature=1.0 \\
  actor_rollout_ref.rollout.top_p=1.0 \\
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
  actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_UTIL \\
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \\
  actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_SEQS \\
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \\
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \\
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \\
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \\
  algorithm.use_kl_in_reward=False \\
  algorithm.rollout_correction.rollout_is=sequence \\
  algorithm.rollout_correction.rollout_is_threshold=2.0 \\
  algorithm.rollout_correction.rollout_is_batch_normalize=False \\
  algorithm.rollout_correction.rollout_rs=null \\
  algorithm.rollout_correction.bypass_mode=False \\
  reward.custom_reward_function.path=$ROOT/scripts/reward_verl_wrapper.py \\
  reward.custom_reward_function.name=compute_score \\
  trainer.critic_warmup=0 \\
  trainer.val_before_train=True \\
  trainer.log_val_generations=8 \\
  trainer.rollout_data_dir=$RUN_DIR/train_generations \\
  trainer.validation_data_dir=$RUN_DIR/val_generations \\
  trainer.logger='["console","tensorboard"]' \\
  trainer.default_local_dir=$RUN_DIR/checkpoints \\
  trainer.project_name=solve_hard_problems_rlvr \\
  trainer.experiment_name=$EXP \\
  trainer.n_gpus_per_node=4 \\
  trainer.nnodes=1 \\
  trainer.save_freq=$SAVE_FREQ \\
  trainer.test_freq=5 \\
  trainer.total_epochs=$TOTAL_EPOCHS
CMD
chmod +x "$RUN_DIR/config/launch_cmd.sh"

LOG_FILE="$RUN_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"

echo "EXP=$EXP"
echo "RUN_DIR=$RUN_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "TRAIN_BS=$TRAIN_BS ROLLOUT_N=$ROLLOUT_N TOTAL_EPOCHS=$TOTAL_EPOCHS SAVE_FREQ=$SAVE_FREQ"

time bash "$RUN_DIR/config/launch_cmd.sh" 2>&1 | tee "$LOG_FILE"

# tensorboard --logdir /home/seankopylov/projects/solve-hard-problems/runs/grpo_qwen3_1p7b_rlvr_hints_no_think_is_4gpu_20260305_195516/tensorboard --host 0.0.0.0 --port 12521
