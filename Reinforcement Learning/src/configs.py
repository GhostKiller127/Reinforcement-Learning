configs = {
    "CartPole-v1": {
        "num_frames": 1000000,
        "replay": 1,
        "num_envs": 16,
        "batch_size": 4,
        "sequence_length": 20,
        "bootstrap_length": 1,
        "d_push": 4,
        "d_pull": 10,
        "discount": 0.995,
        "reward_scaling_1": 5,
        "reward_scaling_2": 1,
        "v_loss_scaling": 1.0,
        "q_loss_scaling": 1.0,
        "p_loss_scaling": 1.0,
        "c_clip": 1.05,
        "rho_clip": 1.05,
        "optimizer": "Adam Weight Decay",
        "weight_decay_rate": 0.01,
        "learning_rate": 3e-4,
        "lr_finder": False,
        "warmup_steps": 1000,
        "adamw_beta1": 0.9,
        "adamw_beta2": 0.98,
        "adamw_epsilon": 1e-6,
        "adamw_clip_norm": 50.0,
        "architecture_params": {
            "architecture": "dense",
            "input_dim": 4,
            "hidden_dim": 128,
            "action_dim": 2},
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
        "num_frames": 3000000,
        "replay": 1,
        "num_envs": 16,
        "batch_size": 4,
        "sequence_length": 20,
        "bootstrap_length": 1,
        "d_push": 4,
        "d_pull": 10,
        "discount": 0.997,
        "reward_scaling_1": 1.25,
        "reward_scaling_2": 0.25,
        "v_loss_scaling": 1.0,
        "q_loss_scaling": 2.0,
        "p_loss_scaling": 2.0,
        "c_clip": 1.05,
        "rho_clip": 1.05,
        "optimizer": "Adam Weight Decay",
        "weight_decay_rate": 0.01,
        "learning_rate": 1e-4,
        "lr_finder": False,
        "warmup_steps": 1000,
        "adamw_beta1": 0.9,
        "adamw_beta2": 0.98,
        "adamw_epsilon": 1e-6,
        "adamw_clip_norm": 50.0,
        "architecture_params": {
            "architecture": "dense",
            "input_dim": 8,
            "hidden_dim": 128,
            "action_dim": 4},
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
        "architecture_params": {
            "architecture": "dense",
            "input_dim": 16,
            "hidden_dim": 128,
            "action_dim": 4},
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