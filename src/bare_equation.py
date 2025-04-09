import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg


class MycelialNetwork:
    def __init__(self):
        # Parameters from paper
        self.α = 0.04     # Branching rate (/hr)
        self.β = 0.000023 # Anastomosis coefficient (μm/hr)
        self.v_avg = 2.4   # Average tip speed (μm/hr)
        self.v_p = 2.8     # Puller tip speed (μm/hr)
        self.puller_frac = 0.3  # 30% puller tips
        self.ρ_sat = 1700.0 # Saturation density (μm⁻¹)

        self.grid_size = 50  # For density fields
        self.n_grid = np.zeros((self.grid_size, self.grid_size))
        self.ρ_grid = np.zeros((self.grid_size, self.grid_size))
        self.puller_mask = None

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
        # Reset merge tracking
        self.merged_positions = np.empty((0, 2))

        # Convert current tip positions to density fields
        self.update_density_fields_from_tips()

        # Detect wavefront (puller tips)
        self.detect_wavefront()

        # Update densities using PDE logic
        self.update_densities_hybrid()

        # Convert densities back to tip counts (ensure minimum 1 tip)
        self.tips = max(1, int(round(self.tips)))
        self.puller_tips = int(self.tips * self.puller_frac)
        self.regular_tips = self.tips - self.puller_tips

        # Tip branching now creates actual new tips
        new_tip_count = int(self.α * self.tips * self.Δt)
        if new_tip_count > 0:
            parent_indices = np.random.choice(len(self.tip_positions), new_tip_count)
            new_positions = self.tip_positions[parent_indices] + np.random.normal(0, 0.1, (new_tip_count, 2))
            self.tip_positions = np.vstack([self.tip_positions, new_positions])

        # Anastomosis with improved merging
        if len(self.tip_positions) > 1:
            # Calculate pairwise distances
            dists = np.linalg.norm(self.tip_positions[:, None] - self.tip_positions[None, :], axis=-1)
            np.fill_diagonal(dists, np.inf)

            # Find merge candidates
            r_merge = 0.5  # Increased merge radius
            i, j = np.where(dists < r_merge)
            pairs = set((min(i,k), max(i,k)) for i,k in zip(i,j) if i < k)

            # Process merges
            merged_indices = set()
            merged_positions = []

            for i, j in pairs:
                if i not in merged_indices and j not in merged_indices:
                    if np.random.rand() < self.β * self.Δt * 100:  # Scaled probability
                        midpoint = (self.tip_positions[i] + self.tip_positions[j]) / 2
                        merged_positions.append(midpoint)
                        merged_indices.update([i, j])
                        self.ρ = min(self.ρ + 20, self.ρ_sat)  # Significant density increase

            # Apply merges
            if merged_indices:
                self.merged_positions = np.array(merged_positions)
                keep_mask = np.ones(len(self.tip_positions), dtype=bool)
                keep_mask[list(merged_indices)] = False
                self.tip_positions = np.vstack([self.tip_positions[keep_mask], self.merged_positions])

        # Tip movement
        self.update_tip_positions()

        # Store data
        self.time += self.Δt
        self.time_series.append(self.time)
        self.tips_series.append(len(self.tip_positions))  # Use actual count
        self.ρ_series.append(self.ρ)

        # Diagnostic output
        if self.time % 1 < self.Δt:  # Print once per hour
            print(f"t={self.time:.1f}hr: tips={len(self.tip_positions)}, mergers={len(self.merged_positions)}, ρ={self.ρ:.1f}")

    def update_density_fields_from_tips(self):
        """Convert discrete tips to continuous density fields"""
        self.n_grid = np.zeros((self.grid_size, self.grid_size))
        self.ρ_grid = np.zeros((self.grid_size, self.grid_size))

        # Convert tip positions to grid coordinates
        grid_coords = ((self.tip_positions + 10) * (self.grid_size-1)/20).astype(int)

        # Spread tip influence over nearby grid points
        for (x,y) in grid_coords:
            for i in range(max(0,x-1), min(self.grid_size,x+2)):
                for j in range(max(0,y-1), min(self.grid_size,y+2)):
                    dist = np.linalg.norm(self.tip_positions - [x,y])
                    weight = np.exp(-dist**2/(2*0.5**2))  # Gaussian kernel
                    self.n_grid[i,j] += weight * self.tips/len(self.tip_positions)

        # Scale hyphal density to match your existing ρ
        self.ρ_grid[:,:] = self.ρ * (self.n_grid/np.sum(self.n_grid) if np.sum(self.n_grid) > 0 else 0)

    def detect_wavefront(self):
        """Identify puller tips at the expanding edge"""
        if len(self.tip_positions) == 0:
            return

        # Find tips farthest from center
        distances = np.linalg.norm(self.tip_positions, axis=1)
        front_threshold = np.percentile(distances, 70)
        self.puller_mask = distances >= front_threshold

    def update_densities_hybrid(self):
        """PDE-inspired updates scaled to your population counts"""
        # Calculate effective branching and anastomosis
        effective_b = self.α * self.tips
        effective_a = self.β * self.tips * self.ρ * 1e-3

        # Apply to tip populations (preserving your tuned behavior)
        new_tips = effective_b * self.Δt
        lost_tips = effective_a * self.Δt

        # Update populations
        self.puller_tips += new_tips * self.puller_frac - lost_tips * (self.puller_tips/max(1,self.tips))
        self.regular_tips += new_tips * (1-self.puller_frac) - lost_tips * (self.regular_tips/max(1,self.tips))
        self.tips = self.puller_tips + self.regular_tips

        # Hyphal growth - pullers contribute more
        puller_ratio = self.puller_tips/max(1,self.tips)
        effective_v = self.v_p * puller_ratio + self.v_avg * (1-puller_ratio)
        Δρ = effective_v * self.tips * self.Δt
        self.ρ = min(self.ρ + Δρ, self.ρ_sat)

    def update_tip_counts_from_densities(self):
        """Ensure conservation of your carefully tuned totals"""
        # No conversion needed - we maintained tip counts directly
        pass

    def update_tip_positions(self):
        """Your existing position update logic"""
        num_new_tips = int(self.α * self.tips * self.Δt)
        if num_new_tips > 0:
            parent_indices = np.random.choice(len(self.tip_positions), num_new_tips)
            new_positions = self.tip_positions[parent_indices] + np.random.randn(num_new_tips, 2) * 0.1
            self.tip_positions = np.vstack([self.tip_positions, new_positions])

        density_gradient = -self.tip_positions / (np.linalg.norm(self.tip_positions, axis=1, keepdims=True) + 1e-5)
        movement = density_gradient * (self.v_avg * self.Δt)
        self.tip_positions += movement

    def run(self):
        # Set matplotlib to non-interactive backend
        import matplotlib
        matplotlib.use('Agg')  # Set before importing pyplot

        while self.time < self.total_time:
            self.update()

            # Plot at specified intervals
            if abs(self.time % self.plot_interval) < self.Δt:
                self.plot_network(self.time)
    
        # Generate final summary plot
        self.plot_summary()

    def plot_network(self, time_index):
        plt.figure(figsize=(8, 8))

        # Plot hyphal density background
        plt.imshow(
            np.full((100, 100), self.ρ / self.ρ_sat),
            extent=(-10, 10, -10, 10),
            cmap='Greens',
            alpha=0.3,
            vmin=0, vmax=1
        )

        # Plot all tips
        plt.scatter(
            self.tip_positions[:, 0], 
            self.tip_positions[:, 1],
            c='blue',
            s=10,
            label=f'Tips ({len(self.tip_positions)})'
        )

        # Highlight merged positions if they exist
        if hasattr(self, 'merged_positions') and len(self.merged_positions) > 0:
            plt.scatter(
                self.merged_positions[:, 0],
                self.merged_positions[:, 1],
                c='red',
                marker='x',
                s=50,
                label=f'Mergers ({len(self.merged_positions)})'
            )

        plt.title(f'Mycelial Network at t={self.time:.1f} hr')
        plt.colorbar(label='Normalized Hyphal Density')
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

        # Save to file without displaying
        plot_filename = os.path.join(self.output_dir, f'network_t={int(time_index):03d}hr.png')
        plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
        plt.close()  # Critical - prevents plot display

# Run simulation
model = MycelialNetwork()
model.run()
