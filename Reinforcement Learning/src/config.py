config_params = {
    "Acrobot-v1": {
        "num_frames": 200000000,
        "replay": 2,
        "num_envs": 160,
        "seq_length": 80,
        "bootstrap": 5,
        "batch_size": 64,
        "discount": 0.997,
        "v_loss_scaling": 1.0,
        "q_loss_scaling": 10.0,
        "pi_loss_scaling": 10.0,
        "importance_sampling_clip_c": 1.05,
        "importance_sampling_clip_rho": 1.05,
        "backbone": "IMPALA,deep",
        "lstm_units": 256,
        "optimizer": "Adam Weight Decay",
        "weight_decay_rate": 0.01,
        "learning_rate": 5e-4,
        "warmup_steps": 4000,
        "adamw_beta1": 0.9,
        "adamw_beta2": 0.98,
        "adamw_epsilon": 1e-6,
        "adamw_clip_norm": 50.0,
        "learner_push_model_every_n_steps": 25,
        "actor_pull_model_every_n_steps": 64,
        "bandit_params": {
            "mode": ["argmax", "random"],
            "tau1": [0.0, 50.0],
            "tau2": [0.0, 50.0],
            "epsilon": [0.0, 1.0],
            "acc": [50, 50, 10],
            "acc2": [2, 3, 4],
            "width": [2, 3, 4],
            "lr": [0.05, 0.1, 0.2],
            "d": 3
        }
    },
    "CartPole-v1": {
        # Add hyperparameters for CartPole-v1 environment here
    },
    "MountainCar-v0": {
        # Add hyperparameters for MountainCar-v0 environment here
    },
    "Pendulum-v1": {
        # Add hyperparameters for Pendulum-v1 environment here
    },
    "BipedalWalker-v2": {
        # Add hyperparameters for BipedalWalker-v2 environment here
    },
    "BipedalWalkerHardcore-v2": {
        # Add hyperparameters for BipedalWalkerHardcore-v2 environment here
    },
    "CarRacing-v0": {
        # Add hyperparameters for CarRacing-v0 environment here
    },
    "LunarLander-v2": {
        # Add hyperparameters for LunarLander-v2 environment here
    },
    "LaserHockey-v0": {
        # Add hyperparameters for LaserHockey-v0 environment here
    },
    "Crypto-v0": {
        # Add hyperparameters for Crypto-v0 environment here
    }
}