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
        # Calculate number of puller tips (must be whole number)
        n_tips = len(self.tip_positions)
        n_pullers = max(1, int(n_tips * self.puller_frac))  # At least 1 puller
        
        # SAFEGUARD: Ensure we don't request more pullers than existing tips
        n_pullers = min(n_pullers, n_tips) if n_tips > 0 else 0
    
        # Identify puller tips as the ones farthest from origin (wavefront)
        puller_indices = np.array([], dtype=int)
        if n_tips > 0 and n_pullers > 0:
            distances = np.linalg.norm(self.tip_positions, axis=1)
            # Get indices of the n_pullers farthest tips
            puller_indices = np.argpartition(distances, -n_pullers)[-n_pullers:]
            # Convert to integer and ensure they're within bounds
            puller_indices = np.unique(puller_indices.astype(int))
            puller_indices = puller_indices[puller_indices < n_tips]  # Ensure valid indices
    
        # Branching - create new tips
        new_tips = []
        for i, tip in enumerate(self.tip_positions):
            if np.random.rand() < self.α * self.Δt:
                # Create new tip near parent with small random offset
                new_tip = tip + np.random.normal(0, 0.1, 2)
                # New tip has 50% chance to inherit puller status if parent is puller
                is_puller = (i in puller_indices) and (np.random.rand() < 0.5)
                new_tips.append((new_tip, is_puller))
    
        # Add new tips
        if new_tips:
            new_positions = [tip[0] for tip in new_tips]
            new_puller_indices = [n_tips + i for i, (_, is_puller) in enumerate(new_tips) if is_puller]
            self.tip_positions = np.vstack([self.tip_positions, new_positions])
            # Combine old and new puller indices, ensuring they're valid
            puller_indices = np.unique(np.concatenate([
                puller_indices,
                np.array(new_puller_indices, dtype=int)
            ]))
            # Ensure all indices are within current bounds
            puller_indices = puller_indices[puller_indices < len(self.tip_positions)]
    
        # Anastomosis
        if len(self.tip_positions) > 1:
            dist_matrix = np.linalg.norm(
                self.tip_positions[:, None] - self.tip_positions[None, :],
                axis=-1
            )
            np.fill_diagonal(dist_matrix, np.inf)
            
            # Find pairs within merge distance
            r_merge = 0.5
            pairs = np.argwhere(dist_matrix < r_merge)
            np.random.shuffle(pairs)  # Randomize merge order
            
            merged_positions = []
            merged_indices = set()
            
            for i, j in pairs:
                if i not in merged_indices and j not in merged_indices:
                    merge_prob = self.β * self.ρ * self.Δt * 1e-3
                    if np.random.rand() < merge_prob:
                        midpoint = (self.tip_positions[i] + self.tip_positions[j]) / 2
                        merged_positions.append(midpoint)
                        merged_indices.update([i, j])
            
            # Apply merges
            if merged_indices:
                keep_mask = np.ones(len(self.tip_positions), dtype=bool)
                keep_mask[list(merged_indices)] = False
                self.tip_positions = np.vstack([
                    self.tip_positions[keep_mask],
                    np.array(merged_positions)
                ])
                # Update puller indices after merge
                puller_indices = np.array([
                    idx for idx in puller_indices 
                    if idx not in merged_indices and idx < len(self.tip_positions)
                ], dtype=int)
    
        # Movement - pullers move faster
        if len(self.tip_positions) > 0:
            # Random walk component + outward bias
            movement = np.random.normal(0, 0.1, self.tip_positions.shape)
            radial = self.tip_positions / (np.linalg.norm(self.tip_positions, axis=1, keepdims=True) + 1e-5)
            movement += 0.3 * radial
            
            # Normalize and scale by speed
            movement = movement / (np.linalg.norm(movement, axis=1, keepdims=True) + 1e-5)
            speeds = np.full(len(self.tip_positions), self.v_avg * self.Δt)
            
            # SAFE INDEXING: Only apply if we have valid puller indices
            if len(puller_indices) > 0:
                valid_pullers = puller_indices[puller_indices < len(speeds)]
                speeds[valid_pullers] = self.v_p * self.Δt
            
            self.tip_positions += movement * speeds[:, None]
    
        # Update hyphal density
        n_pullers = len(puller_indices)
        n_regular = len(self.tip_positions) - n_pullers
        Δρ = (self.v_p * n_pullers + self.v_avg * n_regular) * self.Δt
        self.ρ = min(self.ρ + Δρ, self.ρ_sat)
        
        # Update time and tracking
        self.time += self.Δt
        self.tips = len(self.tip_positions)
        self.puller_tips = n_pullers  # Store actual number of pullers
        self.time_series.append(self.time)
        self.tips_series.append(self.tips)
        self.ρ_series.append(self.ρ)

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
