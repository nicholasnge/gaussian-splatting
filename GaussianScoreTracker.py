import torch

class GaussianScoreTracker:
    def __init__(self):
        """
        Tracks Gaussian scores using a simple running average.
        """
        self.running_sum = None
        self.count = 0
        self.gaussian_count = 0

    def update(self, new_scores):
        """
        Updates the running average of Gaussian scores.

        Args:
            new_scores (torch.Tensor): Tensor of current frame's Gaussian scores.
        """
        with torch.no_grad():
            new_scores = new_scores.detach()  # Detach incoming scores
            new_scores = torch.clamp(new_scores, max=100)
            num_gaussians = new_scores.shape[0]

            if self.running_sum is None or num_gaussians != self.gaussian_count:
                self.reset(num_gaussians)

            self.running_sum += new_scores
            self.count += 1

    def reset(self, num_gaussians):
        """
        Resets the tracker if the number of Gaussians changes.

        Args:
            num_gaussians (int): The new number of Gaussians.
        """
        print("Resetting Gaussian Score Tracker")
        self.running_sum = torch.zeros(num_gaussians, device="cuda", dtype=torch.float32)
        self.count = 0
        self.gaussian_count = num_gaussians

    def get_scores(self):
        """
        Returns the current average scores.

        Returns:
            torch.Tensor or None: Averaged Gaussian scores.
        """
        if self.running_sum is None or self.count == 0:
            return None
        return self.running_sum / self.count
