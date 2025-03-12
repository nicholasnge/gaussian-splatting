import torch


class GaussianScoreScaler:
    @staticmethod
    def linear(grad_threshold, gaussian_scores, n_init_points, alpha=1.0):
        return GaussianScoreScaler._scale(grad_threshold, gaussian_scores, n_init_points, alpha, "linear")

    @staticmethod
    def sqrt(grad_threshold, gaussian_scores, n_init_points, alpha=1.0):
        return GaussianScoreScaler._scale(grad_threshold, gaussian_scores, n_init_points, alpha, "sqrt")

    @staticmethod
    def log(grad_threshold, gaussian_scores, n_init_points, alpha=1.0):
        return GaussianScoreScaler._scale(grad_threshold, gaussian_scores, n_init_points, alpha, "log")

    @staticmethod
    def _scale(grad_threshold, gaussian_scores, n_init_points, alpha, method):
        """Applies the selected scaling method."""
        with torch.no_grad():  # Ensures no gradient tracking
            # Ensure `gaussian_scores` is detached and has `requires_grad=False`
            gaussian_scores = gaussian_scores.detach()

            # Resize `gaussian_scores` if needed
            if gaussian_scores.shape[0] < n_init_points:
                gaussian_scores = torch.cat([
                    gaussian_scores, 
                    torch.zeros(n_init_points - gaussian_scores.shape[0], device="cuda", dtype=gaussian_scores.dtype)
                ])

            # Apply different scaling methods
            if method == "linear":
                scaled_scores = gaussian_scores / gaussian_scores.max()
            elif method == "sqrt":
                scaled_scores = torch.sqrt(gaussian_scores) / torch.sqrt(gaussian_scores.max())
            elif method == "log":
                scaled_scores = torch.log1p(gaussian_scores) / torch.log1p(gaussian_scores.max())
            else:
                raise ValueError(f"Unknown scaling method: {method}")

            # Adjust the gradient threshold using scaled scores
            scaled_scores = 1.0 + alpha * (1.0 - scaled_scores)

            return grad_threshold * scaled_scores
