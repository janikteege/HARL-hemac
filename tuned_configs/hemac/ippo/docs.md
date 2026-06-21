# MAPPO HeMAC Configuration Parameters

## Algorithm Parameters (`algo_args.algo`)

- **`action_aggregation`**: How per-dimension action probability ratios are combined into one training weight (prod to multiply them, mean to average them).
- **`actor_num_mini_batch`**: Number of chunks the actor training data is split into per epoch (e.g., 1 = full batch, 2 = two mini-batches).
- **`clip_param`**: Clipping threshold that limits how far the new policy probability ratio can move from the old policy in one update step (e.g., 0.2 means ratios are clipped to roughly [0.8, 1.2]).
- **`critic_epoch`**: Number of training passes (epochs) over the critic/value-network data per update cycle—higher means the critic is optimized more times before moving on.
- **`critic_num_mini_batch`**: Number of mini-batches used to split critic training data each critic epoch (e.g., 1 = full-batch critic update, higher values = smaller batches and more update steps per epoch).
- **`entropy_coef`**: Weight of the policy entropy bonus in the actor loss, controlling exploration strength (higher = more exploration, lower = more exploitation).
- **`fixed_order`**: Whether agents are updated in a constant sequence every training step (true) or in a varying/randomized order (false).
- **`gae_lambda`**: Generalized Advantage Estimation smoothing factor that trades bias vs. variance in advantage estimates (0 = low variance/high bias, 1 = low bias/higher variance, commonly around 0.95).
- **`gamma`**: Discount factor for future rewards (higher values make the agent care more about long-term returns).
- **`huber_delta`**: Threshold where Huber loss switches from quadratic to linear error penalty in value loss.
- **`max_grad_norm`**: Maximum gradient norm used for gradient clipping to stabilize training.
- **`ppo_epoch`**: Number of optimization passes over actor data per PPO update cycle.
- **`share_param`**: Whether all agents share the same actor parameters.
- **`use_clipped_value_loss`**: Whether to apply PPO-style clipping to value function updates for stability.
- **`use_gae`**: Whether to compute advantages using Generalized Advantage Estimation instead of plain returns.
- **`use_huber_loss`**: Whether to use Huber loss (instead of MSE) for critic/value regression.
- **`use_max_grad_norm`**: Whether to enable gradient clipping using max_grad_norm.
- **`use_policy_active_masks`**: Whether to mask out inactive/dead agents when computing policy loss/entropy.
- **`value_loss_coef`**: Weight multiplier for critic/value loss in the total optimization objective.

## Device Parameters (`algo_args.device`)

- **`cuda`**: Whether to run training/inference on GPU via CUDA (true) instead of CPU.
- **`cuda_deterministic`**: Enables more deterministic CUDA behavior for reproducibility (often with some speed tradeoff).
- **`torch_threads`**: Number of CPU threads PyTorch uses for parallel ops.

## Evaluation Parameters (`algo_args.eval`)

- **`eval_episodes`**: How many episodes to run evaluation for.
- **`n_eval_rollout_threads`**: Number of parallel environment threads used during evaluation.
- **`use_eval`**: Whether to run evaluation during training.

## Model Parameters (`algo_args.model`)

- **`activation_func`**: Activation function used in neural networks (tanh).
- **`critic_lr`**: Learning rate for the critic optimizer.
- **`data_chunk_length`**: Length of data chunks for recurrent policies.
- **`gain`**: Gain of the output layer of the network.
- **`hidden_sizes`**: Sizes of hidden layers in the neural network.
- **`initialization_method`**: Method for initializing network parameters (orthogonal_).
- **`lr`**: Learning rate for the actor optimizer.
- **`opti_eps`**: Epsilon parameter for Adam optimizer.
- **`recurrent_n`**: Number of recurrent layers.
- **`std_x_coef`**: Standard deviation coefficient for policy.
- **`std_y_coef`**: Standard deviation coefficient for policy.
- **`use_feature_normalization`**: Whether to normalize input features.
- **`use_naive_recurrent_policy`**: Whether to use naive recurrent policy.
- **`use_recurrent_policy`**: Whether to use recurrent policy.
- **`weight_decay`**: Weight decay for optimizer regularization.

## Training Parameters (`algo_args.train`)

- **`episode_length`**: Number of environment steps collected per rollout before an update.
- **`eval_interval`**: How often to run evaluation.
- **`log_interval`**: How often to print/write training logs.
- **`model_dir`**: Path to load a pretrained/checkpointed model from (null = train from scratch).
- **`n_rollout_threads`**: Number of parallel environment instances for training data collection.
- **`num_env_steps`**: Total training budget in environment steps before stopping.
- **`use_linear_lr_decay`**: Whether to linearly decay learning rate over training.
- **`use_proper_time_limits`**: Whether to handle time-limit truncations separately from true terminal states in return computation.
- **`use_valuenorm`**: Whether to normalize value targets/returns for critic training stability.

## Environment Parameters (`env_args`)

- **`area_size`**: Size of the simulation area [width, height].
- **`drone_config.discrete_action_space`**: Whether drones use discrete action space.
- **`drone_config.drone_max_charge`**: Maximum charge capacity for drones.
- **`drone_config.drone_max_speed`**: Maximum speed for drones.
- **`drone_config.drone_ui_dimension`**: UI dimension for drone representation.
- **`drone_config.drones_starting_pos`**: Starting positions for drones (empty = random).
- **`drone_sensor.model`**: Sensor model for drones (RoundCamera).
- **`drone_sensor.params.sensing_range`**: Sensing range for drone sensor.
- **`max_cycles`**: Maximum number of cycles per episode.
- **`max_obstacles`**: Maximum number of obstacles in environment.
- **`min_obstacles`**: Minimum number of obstacles in environment.
- **`n_drones`**: Number of drones in the environment.
- **`n_observers`**: Number of observers in the environment.
- **`n_provisioners`**: Number of provisioners in the environment.
- **`observer_comm_range`**: Communication range for observers.
- **`observer_sensor.model`**: Sensor model for observers (ForwardFacingCamera).
- **`observer_sensor.params.hfov`**: Horizontal field of view for observer sensor.
- **`observer_sensor.params.sensing_range`**: Sensing range for observer sensor.
- **`patrol_config.area`**: Patrol area waypoints.
- **`patrol_config.benchmark`**: Whether to use benchmark mode.
- **`poi_config`**: Configuration for points of interest (dimension, spawn mode, speed).
- **`provisioner_sensor.model`**: Sensor model for provisioners.
- **`provisioner_sensor.params.hfov`**: Horizontal field of view for provisioner sensor.
- **`provisioner_sensor.params.sensing_range`**: Sensing range for provisioner sensor.
- **`render_ratio`**: Render ratio for visualization.
- **`rescuing_targets`**: Whether rescuing targets is enabled.
- **`time_factor`**: Time factor for simulation speed.

## Main Arguments (`main_args`)

- **`algo`**: Algorithm to use (mappo).
- **`env`**: Environment to use (hemac).
- **`exp_name`**: Experiment name.
- **`load_config`**: Path to load configuration from (empty = use current config).

