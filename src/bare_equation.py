import numpy as np
import matplotlib.pyplot as plt
import os

class MycelialNetwork:
    def __init__(self):
        # Parameters from paper
        self.α = 0.04     # Branching rate (/hr)
        self.β = 0.000023 # Anastomosis coefficient (μm/hr)
        self.v_avg = 2.4   # Average tip speed (μm/hr)
        self.v_p = 2.8     # Puller tip speed (μm/hr)
        self.puller_frac = 0.3  # 30% puller tips
        self.ρ_sat = 1700.0 # Saturation density (μm⁻¹)

        # Simulation parameters
        self.Δt = 0.01    # Smaller timestep (hr)
        self.total_time = 100
        self.plot_interval = 5  # Plot every 5 hours

        # Initial state
        self.tips = 1.0
        self.puller_tips = self.tips * self.puller_frac
        self.regular_tips = self.tips - self.puller_tips
        self.ρ = 0.0
        self.time = 0.0

        # Initialize tip positions
        self.tip_positions = np.array([[0.0, 0.0]])  # Start at the origin

        # Lists to store data for plotting
        self.time_series = [self.time]
        self.tips_series = [self.tips]
        self.ρ_series = [self.ρ]

        # Create a directory to save plots
        self.output_dir = '../simulations'
        os.makedirs(self.output_dir, exist_ok=True)

    def update(self):
        # Initialize merged_positions to avoid reference errors
        merged_positions = []

        # Ensure we always have at least 1 tip
        if len(self.tip_positions) == 0:
            self.tip_positions = np.array([[0.0, 0.0]])

        # Branching - create new tips
        branching_prob = self.α * self.Δt
        new_tips = []
        for tip in self.tip_positions:
            if np.random.rand() < branching_prob:
                # Create new tip near parent with small random offset
                new_tip = tip + np.random.normal(0, 0.2, 2)
                new_tips.append(new_tip)

        if new_tips:
            self.tip_positions = np.vstack([self.tip_positions, new_tips])

        # Anastomosis with improved merging
        if len(self.tip_positions) > 1:
            dist_matrix = np.linalg.norm(
                self.tip_positions[:, None] - self.tip_positions[None, :],
                axis=-1
            )
            np.fill_diagonal(dist_matrix, np.inf)

            # Find merge candidates
            r_merge = 0.5  # Merge radius
            merge_prob = min(0.5, self.β * self.ρ * 1e-3 * self.Δt * 1000)  # Increased probability

            merged_indices = set()

            for i in range(len(self.tip_positions)):
                for j in range(i+1, len(self.tip_positions)):
                    if dist_matrix[i, j] < r_merge and np.random.rand() < merge_prob:
                        midpoint = (self.tip_positions[i] + self.tip_positions[j]) / 2
                        merged_positions.append(midpoint)
                        merged_indices.update([i, j])
                        self.ρ = min(self.ρ + 10, self.ρ_sat)  # Larger density increase

            # Apply merges
            if merged_indices:
                keep_mask = np.ones(len(self.tip_positions), dtype=bool)
                keep_mask[list(merged_indices)] = False
                self.tip_positions = np.vstack([
                    self.tip_positions[keep_mask],
                    np.array(merged_positions)
                ])

        # Tip movement
        density_gradient = -self.tip_positions / (np.linalg.norm(self.tip_positions, axis=1, keepdims=True) + 1e-5)
        movement = density_gradient * (self.v_avg * self.Δt)
        self.tip_positions += movement

        # Hyphal growth (∂ρ/∂t = v * n)
        Δρ = (self.v_p * self.puller_tips + self.v_avg * self.regular_tips) * self.Δt
        self.ρ = min(self.ρ + Δρ, self.ρ_sat)

        # Update time and tracking
        self.time += self.Δt
        self.tips = len(self.tip_positions)  # Use actual tip count
        self.time_series.append(self.time)
        self.tips_series.append(self.tips)
        self.ρ_series.append(self.ρ)

        # Diagnostic output
        if self.time % 1 < self.Δt:  # Print once per hour
            print(f"t={self.time:.1f}hr: tips={self.tips}, mergers={len(merged_positions)//2}, ρ={self.ρ:.1f}")
            if len(self.tip_positions) > 1:
                print(f"Distance matrix: {dist_matrix}")
                print(f"Merged indices: {merged_indices}")
                print(f"Merged positions: {merged_positions}")

    def run(self):
        print(f"t={self.time:.1f}hr: tips={self.tips:.2f} (pullers={self.puller_tips:.2f}), ρ={self.ρ:.4f} μm")
        self.plot_network(0)  # Plot initial state

        while self.time < self.total_time:
            self.update()
            if abs(self.time % self.plot_interval) < self.Δt:  # Plot every plot_interval hours
                print(f"t={self.time:.1f}hr: tips={self.tips:.2f} (pullers={self.puller_tips:.2f}), ρ={self.ρ:.4f} μm")
                self.plot_network(int(self.time))

    def plot_network(self, time_index):
        plt.figure(figsize=(6, 6))

        # Plot tips at their actual positions
        plt.scatter(self.tip_positions[:, 0], self.tip_positions[:, 1], color='blue', label='Tips', s=5)

        # Plot hyphal density as a background color
        plt.imshow(np.full((100, 100), self.ρ / self.ρ_sat), extent=(-10, 10, -10, 10), cmap='Greens', alpha=0.5)

        plt.title(f'Mycelial Network at t={self.time:.1f} hr')
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()

        # Save the plot as an image file
        plot_filename = os.path.join(self.output_dir, f'network_t={time_index:03d}hr.png')
        plt.savefig(plot_filename)
        plt.close()

# Run simulation
model = MycelialNetwork()
model.run()
