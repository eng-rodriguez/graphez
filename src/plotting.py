import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.patches import Circle, Arc, FancyArrowPatch
from src.connectivity import calculate_connectivity_threshold, prepare_connectivity_matrix


def plot_eegdata(raw, duration=15, scalings=10e-5, time_format="clock"):
    """Plot continous eegdata recording"""
    raw.plot(duration=duration, scalings=scalings, show_scrollbars=False, time_format=time_format)


def plot_epochs(epochs, n_epochs, scalings=10e-5):
    """Plot epochs eegdata recording"""
    epochs.plot(n_epochs=n_epochs, scalings=scalings, show_scrollbars=False)
    

def plot_sources(raw, ica):
    """Plot independent components sources"""
    ica.plot_sources(raw, show_scrollbars=False)


def plot_connectivity_maps_all_bands(con, ch_names, threshold=0.1, cmap="Reds", figsize=(12, 10)):
    """Plot connectivity maps for all frequency bands in a 2x2 grid"""
    bands = list(con.keys())
    
    _, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, band in enumerate(bands):
        if i >= len(axes):
            break
            
        ax = axes[i]
        _plot_connectivity_map(con[band], ch_names, threshold=threshold, cmap=cmap, ax=ax)
        ax.set_title(f"{band.capitalize()} Band (top {threshold:.0%} connections)", pad=10, fontsize=12, fontweight="bold")
    
    _hide_unused_subplots(axes, len(bands))
    plt.tight_layout()
    plt.show()


def plot_connectivity_matrices_all_bands(con, ch_names, cmap="Reds", figsize=(12, 10)):
    """Plot connectivity matrices for all frequency bands in a 2x2 grid"""
    bands = list(con.keys())
    
    _, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, band in enumerate(bands):
        if i >= len(axes):
            break
            
        conn_matrix = prepare_connectivity_matrix(con[band], len(ch_names))
        _plot_single_connectivity_matrix(conn_matrix, ch_names, band, axes[i], cmap)
    
    _hide_unused_subplots(axes, len(bands))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.show()


def _plot_connectivity_map(con, ch_names, threshold=0.1, cmap="Reds", ax=None, electrode_size=800):
    """Plot connectivity topomap with arrows between electrodes."""
    n_channels = len(ch_names)
    con_matrix = prepare_connectivity_matrix(con, n_channels)
    
    # Get electrode layout
    layout = _generate_circle_layout()
    
    # Filter to only include channels that exist in layout
    valid_channels = [channel for channel in ch_names if channel in layout]
    
    # Create position array for valid channels
    used_layout = {channel: layout[channel] for channel in valid_channels}
    position = np.array([used_layout[channel] for channel in valid_channels])
    
    # Filter connectivity matrix to match valid channels
    valid_indices = [i for i, channel in enumerate(ch_names) if channel in layout]
    filtered_con = con_matrix[np.ix_(valid_indices, valid_indices)]
    
    if ax is None:
        _, ax = plt.subplots(figsize=(10,10))
    
    # Draw head outlines
    _draw_head(ax, radius=0.5)
    
    # Plot electrode positions
    for i, channel in enumerate(valid_channels):
        # Electrode circles and labels
        ax.scatter(*position[i], s=electrode_size, facecolor="black", edgecolor="gray", lw=2, zorder=3, alpha=0.9)
        ax.text(*position[i], channel, ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=4)
    
    # Setup colormap and normalization
    if np.max(np.abs(filtered_con)) > 0:
        vmax = np.max(np.abs(filtered_con))
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Calculate threshold
    proportional_threshold = calculate_connectivity_threshold(filtered_con, threshold, valid_channels)
    
    # Draw connectivity arrows
    arrows_drawn = 0
    
    for chani in range(len(valid_channels)):
        for chanj in range(chani + 1, len(valid_channels)):  # Only upper triangle
            strength = abs(filtered_con[chani,chanj])
            if strength >= proportional_threshold:
                # Calculate curved line position with offset to avoid overlap with electrodes
                vec = position[chanj] - position[chani]   
                dist = np.linalg.norm(vec)
                if dist == 0:
                    continue
                
                # Normalize direction vector
                dir_vec = vec / dist
                
                # Offset from electrode centers
                electrode_radius = 0.02
                start = position[chani] + dir_vec * electrode_radius
                end = position[chanj] - dir_vec * electrode_radius
                
                # Line properties
                line_color = plt.cm.get_cmap(cmap)(norm(strength))
                line_width = 3.0  # Constant line width
                
                # Create curved line without arrowhead
                curved_line = FancyArrowPatch(start, end, arrowstyle="-", color=line_color, lw=line_width, zorder=2, connectionstyle="arc3,rad=0.2")
                ax.add_patch(curved_line)
                arrows_drawn += 1
    
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8, pad=0.1)
    cbar.set_label('Connectivity Strength', rotation=270, labelpad=20, fontsize=11)
    
    ax.set_title("Connectivity Map", pad=20, fontsize=14, fontweight="bold")
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_xlim(-0.7, 0.7)              
    ax.set_ylim(-0.7, 0.7)
    
    return ax

def _plot_single_connectivity_matrix(con_matrix, ch_names, band_name, ax, cmap):
    """Plot a single connectivity matrix on the given axis"""
    heatmap = sns.heatmap(
        con_matrix,
        xticklabels=ch_names,
        yticklabels=ch_names,
        linewidths=0.5,
        cmap=cmap,
        square=True,
        ax=ax,
        cbar_kws={"label": "Connectivity Strength"}
    )
    cbar = heatmap.collections[0].colorbar
    cbar.set_label("Connectivity Strength", rotation=270, labelpad=20)
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(5)
    ax.set_title(f"{band_name.capitalize()} Band")


def _hide_unused_subplots(axes, n_used):
    """Hide unused subplots in the grid"""
    for i in range(n_used, len(axes)):
        axes[i].set_visible(False)


def _generate_circle_layout(radius_outer=0.42, radius_inner=0.28):
    """Generate 10-20 layout on anatomically accurate concentric circles"""
    layout = {}
    
    # Outer ring positions (frontal, temporal, and occipital electrodes)
    # Angles are measured from positive x-axis, counterclockwise
    outer_ring_positions = {
        "Fp1": 110,
        "Fp2": 70,
         "F7": 145,
         "F8": 35,
         "T3": 180,
         "T4": 0,
         "T5": 215,
         "T6": 325,
         "O1": 250,
         "O2": 290
    }
    
    # Calculate outer ring positions
    for channel, angle in outer_ring_positions.items():
        rad = np.deg2rad(angle)
        layout[channel] = (radius_outer * np.cos(rad), radius_outer * np.sin(rad))
        
    # Inner ring positions (more centrally located electrodes)
    # These form a smaller circle and are positioned more systematically
    angle_offset = 45 # degrees from horizontal for P3/P4 and F3/F4
    
    # Parietal electrodes
    layout["P3"] = (-radius_inner * np.cos(np.deg2rad(angle_offset)), -radius_inner * np.sin(np.deg2rad(angle_offset)))
    layout["P4"] = ( radius_inner * np.cos(np.deg2rad(angle_offset)), -radius_inner * np.sin(np.deg2rad(angle_offset)))
    
    # Central electrodes (on horizontal line)
    layout["C3"] = (-radius_inner, 0.0)
    layout["C4"] = ( radius_inner, 0.0)
    
    # Frontal electrodes (mirror of parietal)
    layout['F3'] = (-radius_inner * np.cos(np.deg2rad(angle_offset)), radius_inner * np.sin(np.deg2rad(angle_offset)))
    layout['F4'] = ( radius_inner * np.cos(np.deg2rad(angle_offset)), radius_inner * np.sin(np.deg2rad(angle_offset)))
    
    # Midline electrodes
    layout['Fz'] = (0.0, radius_inner * np.sin(np.deg2rad(angle_offset)))
    layout['Cz'] = (0.0, 0.0)
    layout['Pz'] = (0.0, -radius_inner * np.sin(np.deg2rad(angle_offset)))
    
    # Additional common electrodes in 10-20 system
    layout['Fpz'] = (0.0, radius_outer * np.sin(np.deg2rad(90)))
    layout['Oz'] = (0.0, radius_outer * np.sin(np.deg2rad(270)))
    
    # Modern naming equivalents (optional, for backward compatibility)
    layout['T7'] = layout['T3']  
    layout['T8'] = layout['T4']  
    layout['P7'] = layout['T5']  
    layout['P8'] = layout['T6']  
    
    return layout


def _draw_head(ax, radius=0.5):
    """Draw anatomically correct head outline with nose and ears"""
    head_circle = Circle((0,0), radius, edgecolor="black", facecolor="none", lw=3)
    ax.add_patch(head_circle)
    
    # Inner reference circle (dashed) - represent electrode placement boundary
    inner_circle = Circle((0,0), radius * 0.85, edgecolor="gray", facecolor="none", lw=2.5, linestyle=":", alpha=0.7)
    ax.add_patch(inner_circle)
    
    # Nose 
    nose_width = 0.08
    nose_arc = Arc((0, radius), nose_width, 0.06, theta1=0, theta2=180, lw=2, color='black')
    ax.add_patch(nose_arc)
    
    # Ears
    ear_width = 0.08
    ear_height = 0.10
    ear_y_center = ear_height/8
    
    left_ear = Arc((-radius, ear_y_center), ear_width, ear_height, theta1=90, theta2=270, lw=2, color='black')
    ax.add_patch(left_ear)
    
    right_ear = Arc((radius, ear_y_center), ear_width, ear_height, theta1=270, theta2=90, lw=2, color='black')
    ax.add_patch(right_ear)
    
    # Reference Line: Nasion - Inion (anterior - posterior)
    ax.plot([0,0], [-radius*0.9, radius*0.9], linestyle=":", color="gray", lw=1, alpha=0.8)
    
    # Reference Line: Left-Rigth line (lateral)
    ax.plot([-radius*0.9, radius*0.9], [0, 0], linestyle=':', color='gray', lw=1, alpha=0.5)
    
    # Anatomical direction labels
    label_offset = radius + 0.08
    ax.text(0, label_offset, "Anterior\n(Nasion)", ha="center", va="bottom", fontsize=10, fontweight="bold", alpha=0.8)
    ax.text(0, -label_offset, "Posterior\n(Inion)", ha="center", va="top", fontsize=10, fontweight="bold", alpha=0.8)
    ax.text(-label_offset, 0, 'Left', ha='right', va='center', fontsize=10, fontweight='bold', alpha=0.8)
    ax.text(label_offset, 0, 'Right', ha='left', va='center', fontsize=10, fontweight='bold', alpha=0.8)

