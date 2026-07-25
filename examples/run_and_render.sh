python train.py \
  --algo mappo \
  --env hemac \
  --exp_name test_render \
  --use_render True \
  --load_config "./../tuned_configs/hemac/ippo/config.json" \
  --model_dir "./results/hemac/hemac/mappo/new_state_recn1/seed-06636-2026-07-25-13-17-47/models" \
  --render_ratio 1.2
