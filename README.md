# From unsolvable to solvable rlvr

This repo contains my compact setup for training on hard OpenMathReasoning tasks and checking whether hint-based training transfers to no-hint solving.

## 1) Dataset source and initial filtering
- Initial data source: https://huggingface.co/datasets/nvidia/OpenMathReasoning.
- First filter: keep only rows with `pass_rate_72b_tir < 0.3` and `problem_type = has_answer_extracted`.
- During manual checks I found two issues in this split:
  - many wrong `ground_truth` answers;
  - some SFT trajectories looked like answer-guessing.

Because of that, I re-validated problems with `Qwen3-8B` in thinking mode and kept only tasks where `pass@8 > 0`. For these tasks I keep one correct `Qwen3-8B` solution (used later for hint generation).

## 2) Hard subset construction (time-constrained run)
From this cleaned pool I took 2000 tasks and ran `Qwen3-1.7B` with thinking **disabled**.

- Baseline stage (no hint): 4 rollouts per task, `temperature=1`, `max_response_length=4096`.
- Kept only tasks with `pass@4 = 0`.
- Result: **1359** tasks.

I used 4 rollouts (not 128) because of runtime constraints, but for this subset the tasks are hard enough that `pass@4` is still a useful hardness filter.

## 3) Hint generation and filtering
For each of the 1359 tasks:
- `Qwen3-8B` (thinking enabled) got: problem + one correct solution;
- it was prompted to generate a hint (explicit, but without revealing final answer).

Then I evaluated `Qwen3-1.7B` (thinking disabled) on `problem + hint`, again with 4 rollouts.

- Kept only tasks where `0 < avg@4 < 1`.
- Final result: **297** tasks.

So these 297 are exactly the "mid" zone: unsolved without hint in baseline, but partially solvable with hint.

## 4) GRPO training runs
Then I launched two on-policy GRPO trainings (`len(mini_batches)=1`) with `Qwen3-1.7B` (thinking disabled), `temperature=1`, `max_response_length=4096`, and IS correction between `p_vllm` and `p_fsdp`.

1. Main run:
- train: hard tasks + same tasks with hints (594 rows total);
- val: only hard tasks without hints (297 rows).

2. Control run:
- train = val = the same 297 hard tasks without hints.

Goal of this pair: compare transfer from hint-augmented RLVR vs pure no-hint training on the same hard core set.

## Research goal
The main question in this project is whether adding hint-augmented tasks during RLVR can help the model solve harder tasks **without** hints, i.e. whether this training creates useful generalization instead of only overfitting to hint format.

## Data layout
- `data/run_2000_pass4_pipeline_t1_v2_fast_withhint_only_gpu0123_highutil_v2/summary.json`
  - final pipeline counters (`1359` baseline-hard, `297` final hint-helped).
- `data/run_2000_pass4_pipeline_t1_v2_fast_withhint_only_gpu0123_highutil_v2/with_hint_eval.jsonl`
  - per-task eval results for `problem + hint` stage.
- `data/run_2000_pass4_pipeline_t1_v2_fast_withhint_only_gpu0123_highutil_v2/final_tasks_with_hints_avg4_mid.jsonl`
  - final 297 tasks used for RLVR data build.
- `data/rlvr_hints_onpolicy_grpo_v1/{train.parquet,val.parquet,stats.json}`
  - dataset for mixed (hint + no-hint) training.
- `data/rlvr_nohint_onpolicy_grpo_v1/{train.parquet,val.parquet,stats.json,stats_nohint_same.json}`
  - dataset for no-hint control training.

## How to run trainings
1. Prepare pinned `verl` clone:
```bash
bash scripts/setup_verl.sh
```

2. Start main RLVR run (GPUs 0,1,2,3):
```bash
bash scripts/run_grpo_rlvr_qwen3_1p7b_no_think_gpu0123.sh
```

3. Start no-hint control RLVR run (GPUs 4,5,6,7):
```bash
bash scripts/run_grpo_rlvr_qwen3_1p7b_no_think_gpu4567_nohint.sh
```

Both launchers use the same reward interface (`scripts/reward_verl_wrapper.py` -> `scripts/reward_utils.py`) and write TensorBoard events under `runs/*/tensorboard`.

## Repro dependency (verl)
`verl-main` is intentionally not tracked in git. Use:

```bash
bash scripts/setup_verl.sh
```

Pinned version is stored in `VERL_PIN.txt`.

## Privacy / repository scope
I intentionally did not publish raw logs, worker configs, temporary caches, full generation dumps, and other infra-specific artifacts. This is for privacy/safety reasons (machine/user/internal path details) and to keep this repo lightweight.

Also, part of the large-scale inference and hint generation orchestration was executed in a separate working project/environment. Here I attach the core final artifacts needed to reproduce the RLVR stage and understand the experiment decisions.
