import torch 

class GaussianGrowthTracker:
    def __init__(self, object_memory_MB, max_objectid=None, densify_until=15000):
        if isinstance(object_memory_MB, (list, tuple)):
            object_memory_MB = torch.tensor(object_memory_MB, dtype=torch.float32)

        self.densify_until = densify_until
        self.num_objects = object_memory_MB.shape[0] if max_objectid is None else max_objectid + 1
        self.target_memory_bytes = object_memory_MB * 1e6  # MB → bytes

        self.history = {i: [] for i in range(self.num_objects)}
        self.latest_thresholds = torch.ones(self.num_objects, device='cuda')

        # SH degree → number of floats
        self.sh_float_counts = {0: 14, 1: 23, 2: 38, 3: 59}

    def _estimate_object_memory(self, degrees: torch.Tensor) -> float:
        """Estimate memory in bytes for a set of SH degrees."""
        mem = 0.0
        for sh_deg, count in zip(*torch.unique(degrees, return_counts=True)):
            mem += self.sh_float_counts[int(sh_deg.item())] * count.item()
        return mem * 4  # bytes

    def update(self, iteration: int, gaussians) -> torch.Tensor:
        objectid = gaussians.objectid.detach()
        sh_degrees = gaussians._sh_degree.detach()
        thresholds = torch.ones(self.num_objects, device='cuda')

        for oid in range(self.num_objects):
            mask = (objectid == oid)
            if not mask.any():
                continue

            degrees_for_oid = sh_degrees[mask]
            mem_used = self._estimate_object_memory(degrees_for_oid)
            self.history[oid].append((iteration, mem_used, None))

            hist = self.history[oid]
            if len(hist) < 2:
                thresholds[oid] = 1.0
                continue

            N = min(10, len(hist))
            recent = hist[-N:]
            iters, mems, _ = zip(*recent)

            dt = iters[-1] - iters[0]
            if dt == 0:
                thresh = 1.0
            else:
                dmem = mems[-1] - mems[0]
                rate = dmem / dt
                remaining_iters = max(self.densify_until - iters[-1], 1)
                projected_total = mems[-1] + rate * remaining_iters
                target = self.target_memory_bytes[oid].item() * 1.05 # we aim 5% higher since it usually lowers for no reason

                deviation = (projected_total - target) / target
                progress = iters[-1] / self.densify_until
                tolerance = max(2.0 * (1 - progress), 0.05)

                if abs(deviation) < tolerance:
                    thresh = 1.0
                else:
                    factor = min(max(abs(deviation) / tolerance, 1.0), 3.0)
                    thresh = factor if deviation > 0 else 1.0 / factor

            self.history[oid][-1] = (iteration, mem_used, thresh)
            self.latest_thresholds[oid] = thresh
            thresholds[oid] = thresh

        return thresholds
    
    def updateOnly(self, iteration: int, gaussians):
        objectid = gaussians.objectid.detach()
        sh_degrees = gaussians._sh_degree.detach()

        for oid in range(self.num_objects):
            mask = (objectid == oid)
            if not mask.any():
                continue

            degrees_for_oid = sh_degrees[mask]
            mem_used = self._estimate_object_memory(degrees_for_oid)
            self.history[oid].append((iteration, mem_used, None))

    def print_history(self):
        for oid in range(self.num_objects):
            print(f"Object ID {oid} history:")
            for it, mem, thresh in self.history[oid]:
                mem_mb = mem / 1e6
                if thresh is not None:
                    print(f"  Iter {it}, Mem {mem_mb:.2f}MB, Threshold {thresh:.3f}")
                else:
                    print(f"  Iter {it}, Mem {mem_mb:.2f}MB, Threshold not computed yet")

    def getThresholds(self):
        return self.latest_thresholds.clone()

class RatioScalingGaussianGrowthTracker:
    def __init__(self, ratios, total_memory_MB, max_objectid=None, densify_until=15000):
        if isinstance(ratios, (list, tuple)):
            ratios = torch.tensor(ratios, dtype=torch.float32)
        self.ratios = ratios / ratios.min()
        self.total_memory_bytes = total_memory_MB * 1e6
        self.densify_until = densify_until
        self.num_objects = len(ratios) if max_objectid is None else max_objectid + 1

        self.history = {i: [] for i in range(self.num_objects)}
        self.latest_thresholds = self.ratios.to('cuda')

        self.sh_float_counts = {0: 14, 1: 23, 2: 38, 3: 59}

        print(f"[GrowthTracker] Initialized with total_mem={total_memory_MB:.2f}MB, "
            f"ratios={self.ratios.tolist()}, densify_until={self.densify_until}, "
            f"num_objects={self.num_objects}")

    def _estimate_object_memory(self, degrees: torch.Tensor) -> float:
        mem = 0.0
        for sh_deg, count in zip(*torch.unique(degrees, return_counts=True)):
            mem += self.sh_float_counts[int(sh_deg.item())] * count.item()
        return mem * 4  # bytes

    def update(self, iteration: int, gaussians) -> torch.Tensor:
        objectid = gaussians.objectid.detach()
        sh_degrees = gaussians._sh_degree.detach()

        total_mem_used = 0.0
        for oid in range(self.num_objects):
            mask = (objectid == oid)
            if not mask.any():
                continue
            degrees_for_oid = sh_degrees[mask]
            mem_used = self._estimate_object_memory(degrees_for_oid)
            self.history[oid].append((iteration, mem_used, None))
            total_mem_used += mem_used

        # Collect recent memory stats
        valid_histories = [
            (self.history[oid][-2], self.history[oid][-1])
            for oid in range(self.num_objects)
            if len(self.history[oid]) >= 2
        ]

        if len(valid_histories) < 1:
            scale = 1.0
        elif len(self.history[0]) < 5:  # assuming all histories are the same length
            scale = 1.0
        else:
            dt = max(h1[0] for _, h1 in valid_histories) - min(h0[0] for h0, _ in valid_histories)
            dmem = sum(h1[1] - h0[1] for h0, h1 in valid_histories)
            rate = dmem / max(dt, 1)
            latest_iter = max(h1[0] for _, h1 in valid_histories)
            remaining_iters = max(self.densify_until - latest_iter, 1)
            projected_total = total_mem_used + rate * remaining_iters
            target = self.total_memory_bytes * (2 - (iteration/self.densify_until))  # aim for double?
            print(f"target is: {target}")
            
            deviation = projected_total / target
            scale = max(min(deviation, 3.0), 0.7)
            scale = scale if deviation > 0 else 1.0 / scale

        # Apply EMA to update thresholds
        new_thresholds = self.ratios * scale
        self.latest_thresholds = 0.9 * self.latest_thresholds + 0.1 * new_thresholds.to('cuda')
        return self.latest_thresholds.clone()

    def updateOnly(self, iteration: int, gaussians):
        objectid = gaussians.objectid.detach()
        sh_degrees = gaussians._sh_degree.detach()

        for oid in range(self.num_objects):
            mask = (objectid == oid)
            if not mask.any():
                continue
            degrees_for_oid = sh_degrees[mask]
            mem_used = self._estimate_object_memory(degrees_for_oid)
            self.history[oid].append((iteration, mem_used, None))

    def print_history(self):
        for oid in range(self.num_objects):
            print(f"Object ID {oid} history:")
            for it, mem, _ in self.history[oid]:
                print(f"  Iter {it}, Mem {mem / 1e6:.2f}MB")

    def getThresholds(self):
        return self.latest_thresholds.clone()
    

def count_low_sh_components(gaussians, threshold=1e-4):
    """
    For SH degree 3 Gaussians, count how many have degree 3, 2+3, or 1+2+3 coeffs close to zero.
    """
    # Mapping: SH degree → number of coeffs (excluding DC)
    degree_to_coeffs = {1: 3, 2: 5, 3: 7}
    sh3_mask = (gaussians._sh_degree == 3)
    f_rest = gaussians._features_rest[sh3_mask]  # [N, 3, 45] for SH3

    # Counts
    deg3_zero = 0
    deg2_3_zero = 0
    deg1_2_3_zero = 0

    # Indices for degrees (assuming packed in order: deg1, deg2, deg3)
    deg3_start = 3 + 5  # deg1+deg2
    deg3_end = deg3_start + 7

    deg2_start = 3
    deg2_end = deg2_start + 5

    deg1_start = 0
    deg1_end = 3

    # Compute norms over relevant slices
    deg3_norm = torch.norm(f_rest[:, :, deg3_start:deg3_end], dim=(1, 2))
    deg2_norm = torch.norm(f_rest[:, :, deg2_start:deg3_end], dim=(1, 2))  # deg2 + deg3
    deg1_norm = torch.norm(f_rest[:, :, deg1_start:deg3_end], dim=(1, 2))  # deg1 + deg2 + deg3

    deg3_zero = (deg3_norm < threshold).sum().item()
    deg2_3_zero = (deg2_norm < threshold).sum().item()
    deg1_2_3_zero = (deg1_norm < threshold).sum().item()

    print(f"SH3 Gaussians: {sh3_mask.sum().item()}")
    print(f"  Degree 3 close to 0:     {deg3_zero}")
    print(f"  Degree 2+3 close to 0:  {deg2_3_zero}")
    print(f"  Degree 1+2+3 close to 0:{deg1_2_3_zero}")
