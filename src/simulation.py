# plotting.py
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_mycelial_network(time_series, tips_series, rho_series, tip_positions, output_dir, timestep):
    """
    Plot the mycelial network growth over time.

    Parameters:
    - time_series: List of time points.
    - tips_series: List of the number of tips at each time point.
    - rho_series: List of densities at each time point.
    - tip_positions: Array of tip positions at the current timestep.
    - output_dir: Directory to save the plots.
    - timestep: Current timestep for naming the plot file.
    """
    plt.figure(figsize=(10, 5))

    # Plot the number of tips over time
    plt.subplot(1, 2, 1)
    plt.plot(time_series, tips_series, label='Number of Tips')
    plt.xlabel('Time (hr)')
    plt.ylabel('Number of Tips')
    plt.title('Number of Tips Over Time')
    plt.legend()

    # Plot the density over time
    plt.subplot(1, 2, 2)
    plt.plot(time_series, rho_series, label='Density (μm)', color='orange')
    plt.xlabel('Time (hr)')
    plt.ylabel('Density (μm)')
    plt.title('Density Over Time')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/growth_plot_{timestep}.png')
    plt.close()

    # Plot the tip positions
    plt.figure(figsize=(6, 6))
    plt.scatter(tip_positions[:, 0], tip_positions[:, 1], s=5, color='blue')
    plt.xlabel('X Position (μm)')
    plt.ylabel('Y Position (μm)')
    plt.title('Tip Positions')
    plt.axis('equal')
    plt.savefig(f'{output_dir}/tip_positions_{timestep}.png')
    plt.close()

def plot_network(time_series, tips_series, rho_series, tip_positions, output_dir, timestep, time, ρ_sat):
    """
    Plot the mycelial network at a specific timestep.

    Parameters:
    - time_series: List of time points.
    - tips_series: List of the number of tips at each time point.
    - rho_series: List of densities at each time point.
    - tip_positions: Array of tip positions at the current timestep.
    - output_dir: Directory to save the plots.
    - timestep: Current timestep for naming the plot file.
    - time: Current simulation time.
    - ρ_sat: Saturation density.
    """
    plt.figure(figsize=(6, 6))

    # Plot tips at their actual positions
    plt.scatter(tip_positions[:, 0], tip_positions[:, 1], color='blue', label='Tips', s=5)

    # Plot hyphal density as a background color
    density_background = np.full((100, 100), rho_series[-1] / ρ_sat)
    plt.imshow(np.full((100, 100), rho_series[-1] / ρ_sat), extent=(-10, 10, -10, 10), cmap='Greens', alpha=0.5)

    # Draw connections between merged tips
    if len(tip_positions) > 1:
        dist_matrix = np.linalg.norm(
            tip_positions[:, None] - tip_positions[None, :],
            axis=-1
        )
        # Show connections within merge distance
        for i in range(len(tip_positions)):
            for j in range(i + 1, len(tip_positions)):
                if dist_matrix[i, j] < 1.5:  # Slightly larger than merge distance for visibility
                    plt.plot([tip_positions[i, 0], tip_positions[j, 0]],
                             [tip_positions[i, 1], tip_positions[j, 1]],
                             'g-', alpha=0.3, linewidth=0.5)

    plt.title(f'Mycelial Network at t={time:.1f} hr')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()

    # Save the plot as an image file
    plot_filename = os.path.join(output_dir, f'network_t={timestep:03d}hr.png')
    plt.savefig(plot_filename)
    plt.close()
