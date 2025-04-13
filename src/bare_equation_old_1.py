import numpy as np
import matplotlib.pyplot as plt
import os

class MycelialNetwork:
    def __init__(self):
        # Parameters from paper
        self.α = 0.15     # Increased branching rate (/hr)
        self.β = 0.00001 # Decreased anastomosis coefficient (μm hr)
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
        self.α_min = 0.01  # Minimum branching rate at high density

        # Initialize tip positions
        self.tip_positions = np.array([[0.0, 0.0]])  # Start at the origin

        # Lists to store data for plotting
        self.time_series = [self.time]
        self.tips_series = [self.tips]
        self.ρ_series = [self.ρ]

        self.prev_positions = self.tip_positions.copy()
        self.last_merger_count = 0
        self.merge_distance = 5.0  # μm

        self.target_growth_rate = 1.0  # Tips per hour
        self.last_branch_time = 0.0
        self.merger_interval = 1.0     # Check for mergers every hour
        self.last_merger_check = 0.0

        # Create a directory to save plots
        self.output_dir = '../simulations'
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize total merger count
        self.total_merger_count = 0

    def update(self):
        # Initialize tracking
        merger_count = 0

        # Current number of tips
        n_tips = len(self.tip_positions)

        # Calculate puller tips (30% of total, minimum 1)
        n_pullers = max(1, int(n_tips * self.puller_frac))
        n_pullers = min(n_pullers, n_tips) if n_tips > 0 else 0

        # Initialize puller indices - NEW IMPROVED VERSION
        puller_indices = np.array([], dtype=int)
        
        if n_tips > 0:
            # Method 1: Random selection (strict 30% enforcement)
            if n_pullers > 0:
                puller_indices = np.random.choice(n_tips, size=n_pullers, replace=False)

            self.prev_positions = self.tip_positions.copy()

        if n_tips > 0:
            # Calculate radial vectors
            norms = np.linalg.norm(self.tip_positions, axis=1, keepdims=True)
            radial = self.tip_positions / (norms + 1e-5)

            # CONTROLLED BRANCHING with proper puller inheritance
            if self.time - self.last_branch_time >= 1.0/self.target_growth_rate:
                if np.random.rand() < 0.7:  # 70% chance to branch
                    new_pos = self.tip_positions[-1] + radial[-1] * 0.5 + np.random.normal(0, 0.3, 2)
                    is_puller = np.random.rand() < self.puller_frac  # 30% chance to be puller
                    self.tip_positions = np.vstack([self.tip_positions, new_pos])

                    # Update puller indices if new tip is puller
                    if is_puller:
                        puller_indices = np.append(puller_indices, len(self.tip_positions)-1)
                    self.last_branch_time = self.time

            # CONTROLLED ANASTOMOSIS with puller ratio protection
            if self.time - self.last_merger_check >= self.merger_interval:
                if n_tips > 1 and np.random.rand() < 0.5:  # 50% chance per interval
                    non_pullers = [i for i in range(n_tips) if i not in puller_indices]
                    current_puller_ratio = len(puller_indices)/n_tips if n_tips > 0 else 0

                    # Only merge if we have excess non-pullers (protect 30% ratio)
                    if len(non_pullers) > n_tips * (1 - self.puller_frac):
                        remove_idx = np.random.choice(non_pullers)
                        self.tip_positions = np.delete(self.tip_positions, remove_idx, axis=0)
                        merger_count += 1

                        # Update puller indices after merge
                        puller_indices = puller_indices[puller_indices != remove_idx]
                        puller_indices[puller_indices > remove_idx] -= 1
                self.last_merger_check = self.time

            # MOVEMENT with puller speed differentiation
            movement = np.zeros_like(self.tip_positions)
            if len(self.tip_positions) > 0:
                # Recalculate radial vectors after changes
                norms = np.linalg.norm(self.tip_positions, axis=1, keepdims=True)
                radial = self.tip_positions / (norms + 1e-5)

                # Movement calculation
                random_component = np.random.normal(0, 0.1, size=self.tip_positions.shape)
                movement = 0.8 * radial + 0.2 * random_component
                movement = movement / (np.linalg.norm(movement, axis=1, keepdims=True) + 1e-5)

                # Apply speeds (puller tips move faster)
                speeds = np.full(len(self.tip_positions), self.v_avg * self.Δt)
                if len(puller_indices) > 0:
                    valid_pullers = puller_indices[puller_indices < len(speeds)]
                    speeds[valid_pullers.astype(int)] = self.v_p * self.Δt

                self.tip_positions += movement * speeds[:, None]

        # LINEAR DENSITY UPDATE
        self.ρ = min(5.6 * len(self.tip_positions), self.ρ_sat)  # 5.6 μm per tip

        # Update tracking variables
        self.time += self.Δt
        self.tips = len(self.tip_positions)
        self.puller_tips = len(puller_indices)
        self.time_series.append(self.time)
        self.tips_series.append(self.tips)
        self.ρ_series.append(self.ρ)
        self.total_merger_count += merger_count
        
    def run(self):
        print("Time(hr) | Tips | Pullers | Density(μm) | % Saturation | Mergers")
        print("--------------------------------------------------------")
        self.print_status(0)  # Initial state

        while self.time < self.total_time:
            self.update()
            # Print every hour instead of every 5 hours
            if abs(self.time % 1) < self.Δt:  
                self.print_status(int(self.time))
                self.plot_network(int(self.time))  # Still plot every hour


    # Update print_status method to use total_merger_count
    def print_status(self, time_index):
        print(f"{self.time:6.1f} | {self.tips:4} | {self.puller_tips:7} | "
              f"{self.ρ:10.1f} | {self.ρ/self.ρ_sat*100:5.1f}% | "
              f"{self.total_merger_count:6}")

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

        # Draw connections between merged tips
        if len(self.tip_positions) > 1:
            dist_matrix = np.linalg.norm(
                self.tip_positions[:, None] - self.tip_positions[None, :],
                axis=-1
            )
            # Show connections within merge distance
            for i in range(len(self.tip_positions)):
                for j in range(i+1, len(self.tip_positions)):
                    if dist_matrix[i,j] < 1.5:  # Slightly larger than merge distance for visibility
                        plt.plot([self.tip_positions[i,0], self.tip_positions[j,0]],
                                [self.tip_positions[i,1], self.tip_positions[j,1]],
                                'g-', alpha=0.3, linewidth=0.5)


# Run simulation
model = MycelialNetwork()
model.run()
