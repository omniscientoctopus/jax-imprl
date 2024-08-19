import jax


class LinearScheduler:

    def __init__(self, final, num_episodes=None, rate=None, initial=1.0) -> None:

        self.initial = initial * 1.0
        self.final = final * 1.0

        if rate is not None and num_episodes is None:
            self.rate = rate
        elif num_episodes is not None and rate is None:
            self.rate = (self.initial - self.final) / num_episodes
        elif rate is None and num_episodes is None:
            print("Neither rate nor num_episodes provided!")
        else:
            print("Only rate or num_episodes must be provided not both!")

    def get(self, i):
        x = self.initial - self.rate * i
        return jax.lax.clamp(self.final, x, self.initial)
