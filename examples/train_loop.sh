for gamma in 0.95 0.99; do
  for ppo_epoch in 3 5 7; do
    for recurrent in False True; do
      python train.py \
        --env hemac \
        --algo mappo \
        --load_config ./../tuned_configs/hemac/ippo/config.json \
        --exp_name "runB_g${gamma}_ppo${ppo_epoch}_rec${recurrent}" \
        --gamma $gamma \
        --ppo_epoch $ppo_epoch \
        --use_recurrent_policy $recurrent \
        --use_naive_recurrent_policy False \
        --action_aggregation mean \
        --share_param False
    done
  done
done
