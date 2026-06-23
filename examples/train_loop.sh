for agg in mean prod; do
  for recn in 1 2; do
      python train.py \
        --env hemac \
        --algo mappo \
        --load_config ./../tuned_configs/hemac/ippo/config.json \
        --exp_name "runC_agg${agg}_recn${recn}" \
        --action_aggregation $agg \
	--recurrent_n $recn
    done
  done
done
