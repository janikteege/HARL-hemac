for entropy in 0.05 0.08 0.12; do
  for gamma in 0.95 0.99; do
    for lr in 1e-4 2e-5; do
      python train.py \
	--env hemac \
	--algo mappo \
        --load_config ./../tuned_configs/hemac/ippo/config.json \
        --exp_name "e${entropy}_g${gamma}_lr${lr}" \
	--share_param False \
        --entropy_coef $entropy \
        --gamma $gamma \
	--lr $lr
    done
  done
done
