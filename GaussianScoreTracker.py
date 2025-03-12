import torch


class GaussianScoreTracker:
    def __init__(self, ema_alpha=0.1):
        """
        Tracks Gaussian scores over time using an Exponential Moving Average (EMA).

        Args:
            ema_alpha (float): Smoothing factor (higher = more responsive, lower = smoother).
        """
        self.ema_alpha = ema_alpha
        self.gaussian_scores = None  # Moving average scores
        self.gaussian_count = 0  # Track number of Gaussians

    def update(self, new_scores):
        """
        Updates the moving average of Gaussian scores.

        Args:
            new_scores (torch.Tensor): Tensor of current frame's Gaussian scores.
        """
        with torch.no_grad():  # Ensure no gradients are tracked
            new_scores = new_scores.detach()  # Detach incoming scores
            num_gaussians = new_scores.shape[0]

            # If Gaussians count changes, reset tracker
            if self.gaussian_scores is None or num_gaussians != self.gaussian_count:
                self.reset(num_gaussians)

            # Apply EMA Update: new_average = (1 - alpha) * old_average + alpha * new_value
            self.gaussian_scores = (1 - self.ema_alpha) * self.gaussian_scores + self.ema_alpha * new_scores
            self.gaussian_scores.requires_grad_(False)  # Ensure no gradients

    def reset(self, num_gaussians):
        """
        Resets the Gaussian score tracker when Gaussian count changes.

        Args:
            num_gaussians (int): The new number of Gaussians after densification.
        """
        with torch.no_grad():
            print("Resetting Gaussian Score Tracker")
            self.gaussian_scores = torch.zeros(num_gaussians, device="cuda", dtype=torch.float32)
            self.gaussian_scores.requires_grad_(False)  # Ensure it never tracks gradients
            self.gaussian_count = num_gaussians

    def get_scores(self):
        """
        Retrieves the current moving average of Gaussian scores.

        Returns:
            torch.Tensor: Tensor of smoothed Gaussian scores.
        """
        return self.gaussian_scores if self.gaussian_scores is not None else None
