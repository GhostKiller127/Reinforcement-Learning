configs = {
    "CartPole-v1": {
        "load_run": None,
        "wandb_id": None,
        "lr_finder": False,
        "metrics": True,
        "bandits": True,
        "jax_seed": 69,
        "played_frames": 0,
        "train_frames": 5000000,
        "per_buffer_size": 5000000,
        "per_min_frames": 1000000,
        "per_priority_exponent": 0.9,
        "sample_reuse": 2,
        "update_frequency": 1,
        "num_envs": 256,
        "val_envs": 14,
        "batch_size": 16,
        "observation_length": 16,
        "sequence_length": 100,
        "bootstrap_length": 100,
        "d_push": 4000,
        "d_pull": 50,
        "d_target": 500,
        "discount": 0.995,
        "reward_scaling_1": 0.35,
        "reward_scaling_2": 0.3,
        "v_loss_scaling": 1.0,
        "q_loss_scaling": 1.0,
        "p_loss_scaling": 1.0,
        "c_clip": 1.05,
        "rho_clip": 1.05,
        "learning_rate": 1e-3,
        "warmup_steps": 4000,
        "weight_decay": 0.05,
        "adamw_epsilon": 1e-4,
        "adamw_clip_norm": 1e10,
        "mixed_precision": False,
        "architecture": 'S5',
        "parameters": {
            "dense": {
                "input_dim": 4,
                "hidden_dim": 128,
                "action_dim": 2,
                "dropout": 0.1},
            "dense_jax": {
                "input_shape": (1, 4),
                "input_dim": 4,
                "hidden_dim": 128,
                "action_dim": 2,
                "dropout": 0.1},
            "S5": {
                "input_dim": 4,
                "input_shape": (1, 1, 4),
                "n_layers": 4,
                "d_model": 32,
                "ssm_size": 32,
                "blocks": 4,
                "decoder_dim": 64,
                "action_dim": 2,
                "activation": 'half_glu2',
                "prenorm": False,
                "batchnorm": False,
                "dropout": 0.1}},
        "bandit_params": {
            "mode": ["argmax", "random"],
            "tau1": [0.0, 50.0],
            "tau2": [0.0, 50.0],
            "epsilon": [0.0, 1.0],
            "acc": [50, 50, 10],
            "acc2": [2, 3, 4],
            "width": [2, 3, 4],
            "lr": [0.05, 0.1, 0.2],
            "d": 3}
    },
    "LunarLander-v2": {
        "load_run": None,
        "wandb_id": None,
        "lr_finder": False,
        "metrics": True,
        "bandits": True,
        "jax_seed": 69,
        "played_frames": 0,
        "train_frames": 10000000,
        "sample_reuse": 2,
        "per_buffer_size": 5000000,
        "per_min_frames": 1000000,
        "per_priority_exponent": 0.9,
        "update_frequency": 1,
        "num_envs": 256,
        "val_envs": 14,
        "batch_size": 16,
        "observation_length": 16,
        "sequence_length": 100,
        "bootstrap_length": 100,
        "d_push": 4000,
        "d_pull": 50,
        "d_target": 2000,
        "discount": 0.997,
        "reward_scaling_1": 0.15,
        "reward_scaling_2": 0.2,
        "v_loss_scaling": 1.0,
        "q_loss_scaling": 2.0,
        "p_loss_scaling": 3.0,
        "c_clip": 1.05,
        "rho_clip": 1.05,
        "learning_rate": 1e-3,
        "warmup_steps": 4000,
        "weight_decay": 0.05,
        "adamw_epsilon": 1e-4,
        "adamw_clip_norm": 1e10,
        "mixed_precision": False,
        "architecture": 'S5',
        "parameters": {
            "dense": {
                "input_dim": 8,
                "hidden_dim": 256,
                "action_dim": 4,
                "dropout": 0.1},
            "dense_jax": {
                "input_shape": (1, 8),
                "input_dim": 8,
                "hidden_dim": 256,
                "action_dim": 4,
                "dropout": 0.1},
            "S5": {
                "input_dim": 8,
                "input_shape": (1, 1, 8),
                "n_layers": 4,
                "d_model": 32,
                "ssm_size": 32,
                "blocks": 4,
                "decoder_dim": 32,
                "action_dim": 4,
                "activation": 'half_glu2',
                "prenorm": False,
                "batchnorm": False,
                "dropout": 0.1}},
        "bandit_params": {
            "mode": ["argmax", "random"],
            "tau1": [0.0, 50.0],
            "tau2": [0.0, 50.0],
            "epsilon": [0.0, 1.0],
            "acc": [50, 50, 10],
            "acc2": [2, 3, 4],
            "width": [2, 3, 4],
            "lr": [0.05, 0.1, 0.2],
            "d": 3}
    },
    "LaserHockey-v0": {
        "load_run": None,
        "wandb_id": None,
        "lr_finder": False,
        "metrics": True,
        "bandits": True,
        "jax_seed": 69,
        "played_frames": 0,
        "train_frames": 100000000,
        "sample_reuse": 2,
        "per_buffer_size": 5000000,
        "per_min_frames": 1000000,
        "per_priority_exponent": 0.9,
        "update_frequency": 1,
        "num_envs": 256,
        "val_envs": 14,
        "batch_size": 16,
        "observation_length": 4,
        "sequence_length": 100,
        "bootstrap_length": 100,
        "d_push": 4000,
        "d_pull": 50,
        "d_target": 1000,
        "discount": 0.997,
        "reward_scaling_1": 2,
        "reward_scaling_2": 2.4,
        "v_loss_scaling": 8.0,
        "q_loss_scaling": 0.25,
        "p_loss_scaling": 16.0,
        "c_clip": 1.05,
        "rho_clip": 1.05,
        "learning_rate": 1e-3,
        "warmup_steps": 4000,
        "weight_decay": 0.05,
        "adamw_epsilon": 1e-4,
        "adamw_clip_norm": 1e10,
        "mixed_precision": False,
        "architecture": 'S5',
        "parameters": {
            "dense": {
                "input_dim": 16,
                "hidden_dim": 256,
                "action_dim": 27,
                "dropout": 0.1},
            "dense_jax": {
                "input_shape": (1, 16),
                "input_dim": 16,
                "hidden_dim": 256,
                "action_dim": 27,
                "dropout": 0.1},
            "S5": {
                "input_dim": 16,
                "input_shape": (1, 1, 16),
                "n_layers": 4,
                "d_model": 64,
                "ssm_size": 64,
                "blocks": 4,
                "decoder_dim": 64,
                "action_dim": 27,
                "activation": 'half_glu2',
                "prenorm": False,
                "batchnorm": False,
                "dropout": 0.1}},
        "bandit_params": {
            "mode": ["argmax", "random"],
            "tau1": [0.0, 50.0],
            "tau2": [0.0, 50.0],
            "epsilon": [0.0, 1.0],
            "acc": [50, 50, 10],
            "acc2": [2, 3, 4],
            "width": [2, 3, 4],
            "lr": [0.05, 0.1, 0.2],
            "d": 3}
    },
    "Crypto-v0": {
        # Add hyperparameters for Crypto-v0 environment here
    }
}