from typing import Optional


class EMA:
    """
    Exponential moving average tracker.
    """

    def __init__(self, val: Optional[float] = None, decay: float = 0.99):
        self.val = val
        self.decay = decay

    def __call__(self, x):
        """Update the moving average for the variable."""
        prev_average = self.val
        decay = self.decay
        if prev_average is None:
            new_average = x
        else:
            new_average = (1. - decay) * x + decay * prev_average
        self.val = new_average
        return new_average
