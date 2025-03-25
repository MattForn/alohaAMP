import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from simple_1d_gym import SimpleGymEnv

def optimize_sac(trial):
    """Objective function for hyperparameter optimization"""
    env = SimpleGymEnv()

    # Define search space for hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.9999)
    tau = trial.suggest_uniform('tau', 0.005, 0.05)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    #buffer_size = trial.suggest_categorical('buffer_size', [10000, 50000, 100000])
    ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.1, 0.01, 0.001])

    # Create the SAC model with the suggested hyperparameters
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        batch_size=batch_size,
        buffer_size=1000, #buffer_size,
        ent_coef=ent_coef,
        verbose=0  # Suppress training logs for faster optimization
    )

    # Train the model for a short period (to keep optimization fast)
    model.learn(total_timesteps=1000)

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)

    # Return the mean reward as the optimization objective
    return mean_reward

# Run Optuna optimization
study = optuna.create_study(direction="maximize")  # Maximize reward
study.optimize(optimize_sac, n_trials=32, show_progress_bar=True, n_jobs=16)  # Run 20 optimization trials

# Get the best hyperparameters
best_hyperparams = study.best_params
print("Best hyperparameters found:", best_hyperparams)

# Train final SAC model with the best hyperparameters
train_final_mdl = False
if train_final_mdl:
    env = SimpleGymEnv()
    best_model = SAC("MlpPolicy", env, **best_hyperparams, verbose=1)
    best_model.learn(total_timesteps=50000)  # Train longer with the best settings
