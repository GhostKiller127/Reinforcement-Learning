configs = {
    "CartPole-v1": {
        "num_steps": 1000,
        "replay": 2,
        "num_envs": 2,
        "batch_size": 64,
        "sequence_length": 5,
        "bootstrap_length": 5,
        "discount": 0.997,
        "v_loss_scaling": 1.0,
        "q_loss_scaling": 10.0,
        "pi_loss_scaling": 10.0,
        "importance_sampling_clip_c": 1.05,
        "importance_sampling_clip_rho": 1.05,
        "optimizer": "Adam Weight Decay",
        "weight_decay_rate": 0.01,
        "learning_rate": 5e-4,
        "warmup_steps": 4000,
        "adamw_beta1": 0.9,
        "adamw_beta2": 0.98,
        "adamw_epsilon": 1e-6,
        "adamw_clip_norm": 50.0,
        "d_push": 25,
        "d_pull": 64,
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
        "num_steps": 1000,
        "replay": 2,
        "num_envs": 2,
        "batch_size": 64,
        "sequence_length": 5,
        "bootstrap_length": 5,
        "discount": 0.997,
        "v_loss_scaling": 1.0,
        "q_loss_scaling": 10.0,
        "pi_loss_scaling": 10.0,
        "importance_sampling_clip_c": 1.05,
        "importance_sampling_clip_rho": 1.05,
        "optimizer": "Adam Weight Decay",
        "weight_decay_rate": 0.01,
        "learning_rate": 5e-4,
        "warmup_steps": 4000,
        "adamw_beta1": 0.9,
        "adamw_beta2": 0.98,
        "adamw_epsilon": 1e-6,
        "adamw_clip_norm": 50.0,
        "d_push": 25,
        "d_pull": 64,
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