set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ray stop
sleep 5

MODEL_PATH=/mnt/remote-data/zengjie/models/DeepSeek-R1-Distill-Qwen-7B
COLD_START_MODEL_PATH=/mnt/remote-data/zengjie/ckps/qwen_cold_start_0506/checkpoint-168

NODES=8

ray start --head --num-gpus ${NODES} 

# First run
echo "Starting first training run..."
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    custom_reward_function.name=instruction \
    reward_model.reward_manager=instruction_hard \
    config.data.custom_cls.name =instruction \
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
