set -x


# ray stop
# sleep 5

MODEL_PATH=/nas/shared/kilab/hf-hub/DeepSeek-R1-Distill-Qwen-7B
# COLD_START_MODEL_PATH=/mnt/remote-data/zengjie/ckps/qwen_cold_start_0506/checkpoint-168


ray job submit --address="http://127.0.0.1:8265" \
    -- python3 -m verl.trainer.main_ppo \
        data.train_files=/oss/rqy/qingyu/if-verl-main/data/train_all_scaler_processed_2w_dp.parquet\
        data.val_files=/oss/rqy/qingyu/if-verl-main/data/test.parquet\
        data.train_batch_size=768 \
        data.max_prompt_length=8192 \
        data.max_response_length=8192 \
        algorithm.adv_estimator=grpo \
        custom_reward_function.name=instruction \
        reward_model.reward_manager=instruction_hard \
        reward_model.max_length=15000 \
        actor_rollout_ref.model.path=${MODEL_PATH} \
        actor_rollout_ref.rollout.n=5 \
        actor_rollout_ref.actor.ppo_mini_batch_size=96 \
        actor_rollout_ref.actor.use_kl_loss=False \
        algorithm.use_kl_in_reward=False \
        trainer.logger=['swanlab'] \
        trainer.log_val_generations=20 \
        trainer.project_name='verl_grpo_example_gsm8k' \
        trainer.experiment_name='deepseek_llm_7b_function_rm' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=3 \
        trainer.save_freq=52 \
        trainer.test_freq=52 \
        trainer.total_epochs=15 $@
