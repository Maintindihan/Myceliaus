import numpy as np
import os
from scipy_fix import KDTree
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class MycelialNetwork:
    def __init__(self, total_time=1000, plot_interval=50):
        # Your exact parameters
        self.α = 0.15       # Branching rate (/hr)
        self.β = 0.00001    # Anastomosis coefficient (μm hr)
        self.v_avg = 2.4     # Average tip speed (μm/hr)
        self.v_p = 2.8       # Puller tip speed (μm/hr)
        self.puller_frac = 0.3  # Fraction of puller tips
        self.ρ_sat = 1700.0  # Saturation density (μm⁻¹)
        self.α_min = 0.01    # Minimum branching rate
        
        # Simulation parameters
        self.Δt = 0.01       # Timestep (hr)
        self.total_time = total_time
        self.plot_interval = plot_interval
        self.time = 0.0
        self.last_branch_time = 0.0  # Initialize branching timer
        self.merge_distance = 5.0
        
        # Initial state (tracking individual tips)
        self.tip_positions = np.array([[0.0, 0.0]])  # Starting at origin
        self.tip_directions = np.array([[1.0, 0.0]]) # Initial direction
        self.tip_types = np.array([True])  # First tip is puller
        self.hyphae = []     # Stores all segments [start, end]
        self.mergers = 0     # Total merger count

        # Add these if missing:
        self.last_merger_check = 0.0
        self.merger_interval = 1.0  # Check for mergers every hour
        
        # Growth tracking
        self.time_series = []
        self.tips_series = []
        self.ρ_series = []
        self.puller_series = []
        self.merger_series = []
        
        # Output directory
        self.output_dir = '../simulations'
        os.makedirs(self.output_dir, exist_ok=True)

    def update(self):
        n_tips = len(self.tip_positions)
        
        # Movement (puller tips move faster)
        speeds = np.where(self.tip_types, self.v_p, self.v_avg) * self.Δt
        displacements = self.tip_directions * speeds[:, np.newaxis]
        
        # Store previous positions for hyphae tracking
        prev_positions = self.tip_positions.copy()
        self.tip_positions += displacements
        
        # Record new hyphal segments
        for i in range(n_tips):
            self.hyphae.append([prev_positions[i], self.tip_positions[i]])
        
        # Branching (maintains puller fraction)
        if self.time - self.last_branch_time >= 1.0/self.α:
            branch_mask = np.random.rand(n_tips) < self.α * self.Δt
            if np.any(branch_mask):
                # Get branching tips' data
                parent_pos = self.tip_positions[branch_mask]
                parent_dir = self.tip_directions[branch_mask]
                parent_type = self.tip_types[branch_mask]
                
                # Corrected rotation matrix construction
                angles = np.random.normal(0, 0.2, size=len(parent_pos))
                rot_matrices = np.array([[[np.cos(a), -np.sin(a)], 
                                         [np.sin(a), np.cos(a)]] for a in angles])
                new_dirs = np.einsum('ijk,ik->ij', rot_matrices, parent_dir)
                
                # Determine new tip types (30% pullers)
                new_types = np.random.rand(len(parent_pos)) < self.puller_frac
                
                # Add new tips
                self.tip_positions = np.vstack([self.tip_positions, parent_pos])
                self.tip_directions = np.vstack([self.tip_directions, new_dirs])
                self.tip_types = np.concatenate([self.tip_types, new_types])
                
                self.last_branch_time = self.time
        
        # Anastomosis (controlled merging)
        if len(self.tip_positions) > 1:
            kd_tree = KDTree(self.tip_positions)
            pairs = list(kd_tree.query_pairs(self.merge_distance))
            
            # Prefer merging non-pullers to maintain ratio
            non_puller_indices = np.where(~self.tip_types)[0]
            if len(non_puller_indices) > len(self.tip_positions) * (1 - self.puller_frac):
                # Randomly select non-pullers to merge
                merge_count = min(len(pairs), len(non_puller_indices) - int(len(self.tip_positions) * (1 - self.puller_frac)))
                if merge_count > 0:
                    to_merge = np.random.choice(non_puller_indices, size=merge_count, replace=False)
                    self.tip_positions = np.delete(self.tip_positions, to_merge, axis=0)
                    self.tip_directions = np.delete(self.tip_directions, to_merge, axis=0)
                    self.tip_types = np.delete(self.tip_types, to_merge)
                    self.mergers += len(to_merge)
        
        # Update density (linear with tip count up to saturation)
        self.ρ = min(5.6 * len(self.tip_positions), self.ρ_sat)
        
        # Record history
        self.time += self.Δt
        self.time_series.append(self.time)
        self.tips_series.append(len(self.tip_positions))
        self.puller_series.append(np.sum(self.tip_types))
        self.ρ_series.append(self.ρ)
        self.merger_series.append(self.mergers)

         # Visualization update
        if self.time % self.plot_interval == 0:
            self.visualize_growth()

    def print_status(self):
        print(f"{self.time:6.1f} | {len(self.tip_positions):4} | "
              f"{np.sum(self.tip_types):7} | "
              f"{self.ρ:10.1f} | "
              f"{self.ρ/self.ρ_sat*100:5.1f}% | "
              f"{self.mergers:6}")

    def plot_network(self, filename=None):
        plt.figure(figsize=(10, 10))
        
        # Plot hyphae
        if self.hyphae:
            segments = np.array(self.hyphae)
            lc = LineCollection(segments, linewidths=0.5, colors='#8B4513', alpha=0.6)
            plt.gca().add_collection(lc)
        
        # Plot tips
        if len(self.tip_positions) > 0:
            puller_mask = self.tip_types
            plt.scatter(self.tip_positions[puller_mask, 0], self.tip_positions[puller_mask, 1], 
                       color='red', s=15, label='Puller tips')
            plt.scatter(self.tip_positions[~puller_mask, 0], self.tip_positions[~puller_mask, 1],
                       color='blue', s=8, label='Regular tips')
        
        plt.title(f"Mycelial Network (t={self.time:.1f} hr)\n"
                 f"Tips: {len(self.tip_positions)} | Pullers: {np.sum(self.tip_types)}\n"
                 f"Density: {self.ρ:.1f} μm ({self.ρ/self.ρ_sat*100:.1f}% sat)\n"
                 f"Mergers: {self.mergers}")
        plt.xlabel("X position (μm)")
        plt.ylabel("Y position (μm)")
        plt.axis('equal')
        plt.legend()
        
        if filename:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=150)
            plt.close()
        else:
            plt.show()

    def visualize_growth(self):
        """Show actual mycelial growth pattern"""
        self.ax.clear()
        
        # Plot hyphae
        if self.hyphae:
            segments = np.array(self.hyphae)
            lc = LineCollection(segments, linewidths=0.8, 
                               colors=plt.cm.copper(np.linspace(0.2, 0.8, len(self.hyphae))))
            self.ax.add_collection(lc)
        
        # Plot tips with direction indicators
        if len(self.tip_positions) > 0:
            # Color puller tips differently
            puller_mask = self.tip_types
            self.ax.scatter(
                self.tip_positions[puller_mask, 0], 
                self.tip_positions[puller_mask, 1],
                c='red', s=30, label='Puller tips'
            )
            self.ax.scatter(
                self.tip_positions[~puller_mask, 0],
                self.tip_positions[~puller_mask, 1],
                c='blue', s=15, label='Regular tips'
            )
            
            # Add direction vectors
            arrow_scale = 10.0
            for pos, dir in zip(self.tip_positions, self.tip_directions):
                self.ax.arrow(
                    pos[0], pos[1],
                    dir[0]*arrow_scale, dir[1]*arrow_scale,
                    head_width=3, head_length=5, fc='green', ec='green'
                )
        
        # Formatting
        self.ax.set_title(f"Mycelial Growth - Time: {self.time:.1f} hours\n"
                         f"Tips: {len(self.tip_positions)} | "
                         f"Density: {self.ρ:.1f} μm ({self.ρ/self.ρ_sat*100:.1f}% saturation)")
        self.ax.legend()
        plt.draw()
        plt.pause(0.01)  # Brief pause to update display
        
        # Save frame
        os.makedirs('growth_frames', exist_ok=True)
        plt.savefig(f'growth_frames/frame_{int(self.time):04d}.png', dpi=120)

# Run simulation with visualization
sim = MycelialNetwork(total_time=200, plot_interval=5)  # Shorter runtime for demo

print("Simulating mycelial growth...")
while sim.time <= sim.total_time:
    sim.update()
    
print("Simulation complete!")
plt.show()