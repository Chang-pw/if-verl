set -x
export CUDA_VISIBLE_DEVICES=4,5,6,7

ray stop
sleep 5

MODEL_PATH=/data2/dcy/downloads/model/Qwen/Qwen2-7B-Instruct
# COLD_START_MODEL_PATH=/mnt/remote-data/zengjie/ckps/qwen_cold_start_0506/checkpoint-168

NODES=4

ray start --head --num-gpus ${NODES} 

# First run
echo "Starting first training run..."
python3 -m verl.trainer.main_ppo \
    data.train_files=/data1/bowei/if-RL/fork/if-verl/data/train.parquet\
    data.val_files=/data1/bowei/if-RL/fork/if-verl/data/val_set.parquet\
    algorithm.adv_estimator=grpo \
    custom_reward_function.name=instruction \
    reward_model.reward_manager=instruction_hard \
    reward_model.max_length=1500 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    data.train_batch_size=8 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    algorithm.use_kl_in_reward=False \
    trainer.logger=['swanlab'] \
    trainer.log_val_generations=20 \
    trainer.project_name='if-verl_202505' \
    trainer.experiment_name='qwen2_7b_grpo_distill' \
    trainer.n_gpus_per_node=${NODES} \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.val_before_train=True \
    trainer.total_epochs=6 "$@"
