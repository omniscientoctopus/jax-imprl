import jax


class LinearScheduler:

    def __init__(self, final, steps=None, rate=None, initial=1.0) -> None:

        self.initial = initial * 1.0
        self.final = final * 1.0

        if rate is not None and steps is None:
            self.rate = rate
        elif steps is not None and rate is None:
            self.rate = (self.initial - self.final) / steps
        elif rate is None and steps is None:
            print("Neither rate nor steps provided!")
        else:
            print("Only rate or steps must be provided not both!")

    def get(self, i):
        x = self.initial - self.rate * i
        return jax.lax.clamp(self.final, x, self.initial)
