import numpy as np
import matplotlib.pyplot as plt
import os

class MycelialNetwork:
    def __init__(self):
        # Parameters from paper
        self.α = 0.15     # Branching rate (/hr)
        self.β = 0.00001  # Anastomosis coefficient (μm hr)
        self.v_avg = 2.4   # Average tip speed (μm/hr)
        self.v_p = 2.8     # Puller tip speed (μm/hr)
        self.puller_frac = 0.3  # 30% puller tips
        self.ρ_sat = 1700.0 # Saturation density (μm⁻¹)

        # Simulation parameters
        self.Δt = 0.01    # Small time step (hr)
        self.total_time = 1000
        self.plot_interval = 100  # Plot every 5 hours

        # Initial state
        self.tips = 1.0
        self.puller_tips = self.tips * self.puller_frac
        self.regular_tips = self.tips - self.puller_tips
        self.ρ = 0.0
        self.time = 0.0
        self.α_min = 0.01  # Minimum branching rate at high density

        # Initialize tip positions
        self.tip_positions = np.array([[0.0, 0.0]])  # Start at the origin
        self.tip_directions = np.array([[1.0, 0.0]])
        self.branch_angle_std = 0.3

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

        self.puller_indices = np.array([], dtype=int)  # Track puller indices

    def update(self):
        # Initialize tracking
        merger_count = 0

        # Current number of tips
        n_tips = len(self.tip_positions)

        # Calculate puller tips (30% of total, minimum 1)
        n_pullers = max(1, int(n_tips * self.puller_frac))
        n_pullers = min(n_pullers, n_tips) if n_tips > 0 else 0

        # Initialize puller indices
        puller_indices = np.array([], dtype=int)

        if n_tips > 0:
            # Random selection of puller tips
            if n_pullers > 0:
                puller_indices = np.random.choice(n_tips, size=n_pullers, replace=False)

            # Calculate radial vectors
            norms = np.linalg.norm(self.tip_positions, axis=1, keepdims=True)
            radial = self.tip_positions / (norms + 1e-5)

            # Controlled Branching with proper puller inheritance
            if self.time - self.last_branch_time >= 1.0 / self.target_growth_rate:
                if np.random.rand() < 0.7:  # 70% chance to branch
                    parent_idx = np.random.randint(n_tips)
                    parent_pos = self.tip_positions[parent_idx]
                    parent_dir = radial[parent_idx]

                    # Create new branch direction with full circle coverage
                    branch_angle = np.random.uniform(0, 2 * np.pi)  # Full circle
                    rot_matrix = np.array([
                        [np.cos(branch_angle), -np.sin(branch_angle)],
                        [np.sin(branch_angle), np.cos(branch_angle)]
                    ])
                    new_dir = np.dot(rot_matrix, parent_dir)

                    new_pos = parent_pos + new_dir * 0.5 + np.random.normal(0, 0.3, 2)
                    is_puller = np.random.rand() < self.puller_frac  # 30% chance to be puller
                    self.tip_positions = np.vstack([self.tip_positions, new_pos])

                    # Update puller indices if new tip is puller
                    if is_puller:
                        puller_indices = np.append(puller_indices, len(self.tip_positions) - 1)
                    self.last_branch_time = self.time

            # Controlled Anastomosis with puller ratio protection
            if self.time - self.last_merger_check >= self.merger_interval:
                if n_tips > 1 and np.random.rand() < 0.5:  # 50% chance per interval
                    non_pullers = [i for i in range(n_tips) if i not in puller_indices]
                    current_puller_ratio = len(puller_indices) / n_tips if n_tips > 0 else 0

                    # Only merge if we have excess non-pullers (protect 30% ratio)
                    if len(non_pullers) > n_tips * (1 - self.puller_frac):
                        remove_idx = np.random.choice(non_pullers)
                        self.tip_positions = np.delete(self.tip_positions, remove_idx, axis=0)
                        merger_count += 1

                        # Update puller indices after merge
                        puller_indices = puller_indices[puller_indices != remove_idx]
                        puller_indices[puller_indices > remove_idx] -= 1
                self.last_merger_check = self.time

            # Movement with puller speed differentiation
            movement = np.zeros_like(self.tip_positions)
            if len(self.tip_positions) > 0:
                # Recalculate radial vectors after changes
                norms = np.linalg.norm(self.tip_positions, axis=1, keepdims=True)
                radial = self.tip_positions / (norms + 1e-5)

                # Movement calculation with more randomness
                random_component = np.random.normal(0, 0.3, size=self.tip_positions.shape)
                movement = 0.6 * radial + 0.4 * random_component
                movement = movement / (np.linalg.norm(movement, axis=1, keepdims=True) + 1e-5)

                # Apply speeds (puller tips move faster)
                speeds = np.full(len(self.tip_positions), self.v_avg * self.Δt)
                if len(puller_indices) > 0:
                    valid_pullers = puller_indices[puller_indices < len(speeds)]
                    speeds[valid_pullers.astype(int)] = self.v_p * self.Δt

                self.tip_positions += movement * speeds[:, None]

        # Linear density update
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
            # if abs(self.time % self.plot_interval) < self.Δt:
            if abs(self.time % 1) < self.Δt:  # Print every hour
                self.print_status(int(self.time))
                self.plot_network(int(self.time))
                # self.plot_radial_distribution()  # Optional
        
        self.plot_time_series()        

    def print_status(self, time_index):
        # Calculate merger count (you'll need to track this in update())
        # merger_count = getattr(self, 'last_merger_count', 0)

        # print(f"{self.time:6.1f} | {self.tips:4} | {self.puller_tips:7} | "
        #       f"{self.ρ:10.1f} | {self.ρ / self.ρ_sat * 100:5.1f}% | "
        #       f"{self.total_merger_count:6}")

        summary_line = (f"{self.time:6.1f} | {len(self.tip_positions):4} | {self.puller_tips:7} | "
                   f"{self.ρ:10.1f} | {self.ρ / self.ρ_sat * 100:5.1f}% | "
                   f"{self.total_merger_count:6}")
    
        # Detailed coordinates output
        coord_header = "\nTip Positions:"
        coord_lines = []
        for i, pos in enumerate(self.tip_positions):
            is_puller = "P" if i in self.puller_indices else " "
            coord_lines.append(f"Tip {i:2}{is_puller}: ({pos[0]:6.1f}, {pos[1]:6.1f})")

        # Recent merger events
        merger_info = ""
        if hasattr(self, 'last_merger_count') and self.last_merger_count > 0:
            merger_info = f"\nRecent mergers: {self.last_merger_count} tips merged this step"

        # Full debug output
        debug_output = (f"\n=== Network Status at t={self.time:.1f} hr ===\n"
                       f"{summary_line}\n"
                       f"{coord_header}\n" + "\n".join(coord_lines) + 
                       f"\nWavefront radius: {np.max(np.linalg.norm(self.tip_positions, axis=1)) if len(self.tip_positions) > 0 else 0:.1f} μm" +
                       f"{merger_info}\n" +
                       "="*40)

        print(debug_output)

    def plot_time_series(self):
        plt.figure(figsize=(12, 8))

        # Tip count over time
        plt.subplot(2, 2, 1)
        plt.plot(self.time_series, self.tips_series, 'b-')
        plt.xlabel('Time (hr)')
        plt.ylabel('Number of Tips')
        plt.title('Tip Population Dynamics')
        plt.grid(True, alpha=0.3)

        # Density over time
        plt.subplot(2, 2, 2)
        plt.plot(self.time_series, self.ρ_series, 'g-')
        plt.axhline(self.ρ_sat, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Time (hr)')
        plt.ylabel('Density (μm⁻¹)')
        plt.title('Hyphal Density')
        plt.grid(True, alpha=0.3)

        # Puller tip ratio
        puller_ratio = [self.puller_frac] * len(self.time_series)  # Simplified
        plt.subplot(2, 2, 3)
        plt.plot(self.time_series, puller_ratio, 'm-')
        plt.xlabel('Time (hr)')
        plt.ylabel('Puller Tip Fraction')
        plt.title('Puller Tip Maintenance')
        plt.grid(True, alpha=0.3)

        # Wavefront progression
        if hasattr(self, 'max_dist_series'):
            plt.subplot(2, 2, 4)
            plt.plot(self.time_series, self.max_dist_series, 'c-')
            plt.xlabel('Time (hr)')
            plt.ylabel('Wavefront Radius (μm)')
            plt.title('Wavefront Progression')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        ts_filename = os.path.join(self.output_dir, 'time_series.png')
        plt.savefig(ts_filename, dpi=150)
        plt.close()

    def plot_radial_distribution(self):
        if len(self.tip_positions) == 0:
            return

        # Calculate distances from origin
        distances = np.linalg.norm(self.tip_positions, axis=1)

        # Create histogram
        bins = np.linspace(0, np.max(distances)+1, 20)
        counts, bin_edges = np.histogram(distances, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        plt.figure(figsize=(8, 6))
        plt.bar(bin_centers, counts, width=bins[1]-bins[0], alpha=0.7)
        plt.xlabel('Distance from Origin (μm)')
        plt.ylabel('Number of Tips')
        plt.title('Radial Distribution of Tips')
        plt.grid(True, alpha=0.3)

        rad_filename = os.path.join(self.output_dir, f'radial_dist_t={self.time:.1f}.png')
        plt.savefig(rad_filename, dpi=150)
        plt.close()

    def plot_network(self, time_index):
        plt.figure(figsize=(10, 10))

        # Create a colormap for tip ages (if tracking)
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.tip_positions)))

        # Plot connections between nearby tips (simulating hyphae)
        if len(self.tip_positions) > 1:
            # Calculate pairwise distances
            dists = np.linalg.norm(self.tip_positions[:, None] - self.tip_positions[None, :], axis=-1)

            # Draw connections between nearby tips
            for i in range(len(self.tip_positions)):
                for j in range(i+1, len(self.tip_positions)):
                    if dists[i,j] < 2.0:  # Connection threshold
                        plt.plot([self.tip_positions[i,0], self.tip_positions[j,0]],
                                [self.tip_positions[i,1], self.tip_positions[j,1]],
                                'g-', alpha=0.4, linewidth=0.7)

        # Plot tips with size according to their activity
        plt.scatter(self.tip_positions[:,0], self.tip_positions[:,1], 
                    c=colors, s=50, edgecolors='k', linewidths=0.5)

        # Add a circle showing wavefront progression
        max_dist = np.max(np.linalg.norm(self.tip_positions, axis=1)) if len(self.tip_positions) > 0 else 0
        wavefront_circle = plt.Circle((0,0), max_dist, color='r', fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_patch(wavefront_circle)

        # Formatting
        plt.title(f"Mycelial Network at t={self.time:.1f} hr\n"
                  f"Tips: {len(self.tip_positions)} | Density: {self.ρ:.1f} μm⁻¹ ({self.ρ/self.ρ_sat*100:.1f}% saturation)")
        plt.xlabel("X Position (μm)")
        plt.ylabel("Y Position (μm)")
        plt.grid(True, alpha=0.3)

        # Dynamic axis limits based on tip spread
        if len(self.tip_positions) > 0:
            buffer = 5
            x_min, x_max = np.min(self.tip_positions[:,0])-buffer, np.max(self.tip_positions[:,0])+buffer
            y_min, y_max = np.min(self.tip_positions[:,1])-buffer, np.max(self.tip_positions[:,1])+buffer
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

        # Save plot
        plot_filename = os.path.join(self.output_dir, f'network_{time_index:03d}hr.png')
        plt.savefig(plot_filename, dpi=150)
        plt.close()

# Run simulation
model = MycelialNetwork()
model.run()
