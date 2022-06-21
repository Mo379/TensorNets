import os
from datetime import timedelta
from time import time

class trainer:
    """
    Trainer.
    """
    def __init__(
        self,
        env,
        env_test,
        algo,
        log_dir,
        seed,
        action_repeat,
        num_agent_steps,
        eval_interval,
        num_eval_episodes,
        save_params,
        wandb_run,
    ):
        assert num_agent_steps % action_repeat == 0
        assert eval_interval % action_repeat == 0

        # Envs.
        self.env = env
        self.env_test = env_test

        # Set seeds.
        self.env.seed(seed)
        self.env_test.seed(2 ** 31 - seed)

        # Algorithm.
        self.algo = algo

        # Log setting.
        self.log = {"step": [], "return": []}
        self.csv_path = os.path.join(log_dir, "log.csv")
        self.param_dir = os.path.join(log_dir, "param")

        # Other parameters.
        self.action_repeat = action_repeat
        self.num_agent_steps = num_agent_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.save_params = save_params
        self.wandb_run = wandb_run

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Initialize the environment.
        state = self.env.reset()

        for step in range(1, self.num_agent_steps + 1):
            state = self.algo.step(self.env, state)

            if self.algo.is_update(state.shape[0]):
                print('learning')
                self.algo.update(self.writer)

            if step % self.eval_interval == 0:
                #self.evaluate(step)
                
                if self.save_params:
                    print('saving_params')
                    self.algo.save_params(os.path.join(self.param_dir, f"step{step}"))


    def evaluate(self, step):
        total_return = 0.0
        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            done = False
            while not done:
                action = self.algo.select_action(state)
                state, reward, done, _ = self.env_test.step(action)
                total_return += reward

        # Log mean return.
        mean_return = total_return / self.num_eval_episodes
        # To CSV.
        self.log["step"].append(step * self.action_repeat)
        self.log["return"].append(mean_return)
        pd.DataFrame(self.log).to_csv(self.csv_path, index=False)


    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
