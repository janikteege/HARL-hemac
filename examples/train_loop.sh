for entropy in 0.05 0.08 0.12 0.16; do
  for gamma in 0.95 0.97 0.99; do
    for lr in 5e-5 1e-4; do
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
