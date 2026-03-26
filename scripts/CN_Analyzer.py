import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from scipy.spatial import KDTree
from scipy.signal import find_peaks
import warnings
import os

warnings.filterwarnings('ignore')

class CoordinationNumberAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Coordination Number Analyzer")
        self.root.geometry("1400x900")
        
        # Variables
        self.atoms = []
        self.coords = []
        self.atom_symbols = []
        self.coordination_numbers = []
        self.neighbor_indices = []  # For each atom store the indices of its neighbors
        
        # Analysis parameters
        self.min_neighbors = tk.IntVar(value=8)
        self.max_neighbors = tk.IntVar(value=16)
        self.target_coordination = tk.IntVar(value=12)
        
        # Plot settings for publication
        self.axis_label_fontsize = tk.IntVar(value=14)
        self.axis_tick_fontsize = tk.IntVar(value=12)
        self.title_fontsize = tk.IntVar(value=16)
        self.font_family = tk.StringVar(value='Arial')
        self.plot_dpi = tk.IntVar(value=600)
        self.hist_color = tk.StringVar(value='#1f77b4')
        self.central_atom_symbol = tk.StringVar(value='Fe')
        self.neighbor_atom_symbol = tk.StringVar(value='Li')
        
        # Available fonts
        self.available_fonts = ['Arial', 'Times New Roman', 'Courier New', 
                               'Helvetica', 'DejaVu Sans', 'serif', 'sans-serif']
        
        # Colors for histogram
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        """Create the user interface"""
        # Main container
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left control panel
        control_frame = ttk.Frame(main_paned, width=400)
        main_paned.add(control_frame, weight=1)
        
        # Right panel for plots
        self.main_frame = ttk.Frame(main_paned)
        main_paned.add(self.main_frame, weight=3)
        
        # Scrollable control panel
        canvas = tk.Canvas(control_frame)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Content of the control panel
        content_frame = ttk.Frame(scrollable_frame, padding="15")
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        row = 0
        
        # Title
        title_label = tk.Label(content_frame, text="Coordination Number Analyzer", 
                              font=('Arial', 16, 'bold'))
        title_label.grid(row=row, column=0, columnspan=3, pady=(0, 20))
        row += 1
        
        # Section: File loading
        ttk.Label(content_frame, text="File Operations", 
                 font=('Arial', 12, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        row += 1
        
        # Button to load file
        ttk.Button(content_frame, text="Load XYZ File", 
                  command=self.load_xyz_file, width=20).grid(
            row=row, column=0, columnspan=3, pady=10)
        row += 1
        
        # Section: Analysis parameters
        ttk.Label(content_frame, text="Analysis Parameters", 
                 font=('Arial', 12, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(10, 10), sticky=tk.W)
        row += 1
        
        # Minimum number of neighbors to show
        ttk.Label(content_frame, text="Min neighbors to show:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        ttk.Spinbox(content_frame, from_=0, to=100, 
                   textvariable=self.min_neighbors, width=10).grid(
            row=row, column=1, sticky=tk.W, pady=3, padx=5)
        row += 1
        
        # Maximum number of neighbors to show
        ttk.Label(content_frame, text="Max neighbors to show:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        ttk.Spinbox(content_frame, from_=1, to=100, 
                   textvariable=self.max_neighbors, width=10).grid(
            row=row, column=1, sticky=tk.W, pady=3, padx=5)
        row += 1
        
        # Button to calculate
        ttk.Button(content_frame, text="Calculate Coordination Numbers", 
                  command=self.calculate_coordination_numbers, width=25).grid(
            row=row, column=0, columnspan=3, pady=15)
        row += 1
        
        # Separator
        ttk.Separator(content_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=15)
        row += 1
        
        # Section: Save atoms with a specific CN
        ttk.Label(content_frame, text="Save Atoms with Specific CN", 
                 font=('Arial', 12, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        row += 1
        
        # Target coordination number
        ttk.Label(content_frame, text="Target coordination number:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        ttk.Spinbox(content_frame, from_=0, to=100, 
                   textvariable=self.target_coordination, width=10).grid(
            row=row, column=1, sticky=tk.W, pady=3, padx=5)
        row += 1
        
        # Atom symbols for saving
        ttk.Label(content_frame, text="Central atom symbol:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        ttk.Entry(content_frame, textvariable=self.central_atom_symbol, 
                 width=8).grid(row=row, column=1, sticky=tk.W, pady=3, padx=5)
        row += 1
        
        ttk.Label(content_frame, text="Neighbor atom symbol:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        ttk.Entry(content_frame, textvariable=self.neighbor_atom_symbol, 
                 width=8).grid(row=row, column=1, sticky=tk.W, pady=3, padx=5)
        row += 1
        
        # Button to save atoms
        ttk.Button(content_frame, text="Save Atoms with Target CN", 
                  command=self.save_atoms_with_target_cn, width=25).grid(
            row=row, column=0, columnspan=3, pady=10)
        row += 1
        
        # Button to save all environments
        ttk.Button(content_frame, text="Save All Environments", 
                  command=self.save_all_environments, width=25).grid(
            row=row, column=0, columnspan=3, pady=5)
        row += 1
        
        # Separator
        ttk.Separator(content_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=15)
        row += 1
        
        # Section: Plot settings for publication
        ttk.Label(content_frame, text="Plot Settings for Publication", 
                 font=('Arial', 12, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        row += 1
        
        # Font family
        ttk.Label(content_frame, text="Font Family:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        font_combo = ttk.Combobox(content_frame, textvariable=self.font_family,
                                 values=self.available_fonts, width=15, state='readonly')
        font_combo.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=3, padx=5)
        row += 1
        
        # Title font size
        ttk.Label(content_frame, text="Title Font Size:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        title_frame = ttk.Frame(content_frame)
        title_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=3, padx=5)
        ttk.Entry(title_frame, textvariable=self.title_fontsize, width=6).pack(side=tk.LEFT)
        ttk.Label(title_frame, text="pt").pack(side=tk.LEFT, padx=5)
        row += 1
        
        # Axis label font size
        ttk.Label(content_frame, text="Axis Label Font Size:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        label_frame = ttk.Frame(content_frame)
        label_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=3, padx=5)
        ttk.Entry(label_frame, textvariable=self.axis_label_fontsize, width=6).pack(side=tk.LEFT)
        ttk.Label(label_frame, text="pt").pack(side=tk.LEFT, padx=5)
        row += 1
        
        # Axis tick font size
        ttk.Label(content_frame, text="Axis Tick Font Size:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        tick_frame = ttk.Frame(content_frame)
        tick_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=3, padx=5)
        ttk.Entry(tick_frame, textvariable=self.axis_tick_fontsize, width=6).pack(side=tk.LEFT)
        ttk.Label(tick_frame, text="pt").pack(side=tk.LEFT, padx=5)
        row += 1
        
        # Histogram color
        ttk.Label(content_frame, text="Histogram Color:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        color_frame = ttk.Frame(content_frame)
        color_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=3, padx=5)
        
        color_combo = ttk.Combobox(color_frame, textvariable=self.hist_color,
                                  values=self.colors, width=10, state='readonly')
        color_combo.pack(side=tk.LEFT)
        self.color_preview = tk.Label(color_frame, width=3, bg=self.hist_color.get())
        self.color_preview.pack(side=tk.LEFT, padx=5)
        color_combo.bind('<<ComboboxSelected>>', self.update_color_preview)
        row += 1
        
        # DPI for saving
        ttk.Label(content_frame, text="DPI for Saving:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        dpi_frame = ttk.Frame(content_frame)
        dpi_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=3, padx=5)
        
        ttk.Entry(dpi_frame, textvariable=self.plot_dpi, width=6).pack(side=tk.LEFT)
        ttk.Label(dpi_frame, text="(300-1200)").pack(side=tk.LEFT, padx=5)
        row += 1
        
        # Button to update plot appearance
        ttk.Button(content_frame, text="Update Plot Appearance", 
                  command=self.update_plot_appearance, width=25).grid(
            row=row, column=0, columnspan=3, pady=10)
        row += 1
        
        # Button to save high-quality plot
        ttk.Button(content_frame, text="Save High-Quality Plot", 
                  command=self.save_high_quality_plot, width=25).grid(
            row=row, column=0, columnspan=3, pady=5)
        row += 1
        
        # Separator
        ttk.Separator(content_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=15)
        row += 1
        
        # Section: File information
        ttk.Label(content_frame, text="File Information", 
                 font=('Arial', 12, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        row += 1
        
        # Text area for information
        self.info_text = tk.Text(content_frame, height=15, width=45, wrap=tk.WORD,
                                font=('Courier New', 9))
        scrollbar_info = tk.Scrollbar(content_frame, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar_info.set)
        
        self.info_text.grid(row=row, column=0, columnspan=2, pady=5, sticky=tk.W+tk.E)
        scrollbar_info.grid(row=row, column=2, pady=5, sticky=tk.N+tk.S)
        
        # Create figure for histogram
        self.fig, self.ax = plt.subplots(figsize=(10, 6), dpi=100)
        
        # Apply initial font settings
        self.apply_font_settings()
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to work")
        tk.Label(self.root, textvariable=self.status_var, 
                bd=1, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_color_preview(self, event=None):
        """Update color preview"""
        self.color_preview.config(bg=self.hist_color.get())
    
    def apply_font_settings(self):
        """Apply font settings to the current plot"""
        if hasattr(self, 'ax') and self.ax:
            # Apply font family
            plt.rcParams['font.family'] = self.font_family.get()
            
            # Update title font size if it exists
            if self.ax.get_title():
                self.ax.set_title(self.ax.get_title(), 
                                 fontsize=self.title_fontsize.get(), 
                                 fontweight='bold')
            
            # Update axis label font size
            self.ax.set_xlabel(self.ax.get_xlabel(), 
                              fontsize=self.axis_label_fontsize.get())
            self.ax.set_ylabel(self.ax.get_ylabel(), 
                              fontsize=self.axis_label_fontsize.get())
            
            # Update axis tick font size
            self.ax.tick_params(axis='both', which='major', 
                               labelsize=self.axis_tick_fontsize.get())
    
    def update_plot_appearance(self):
        """Update plot appearance based on current settings"""
        if hasattr(self, 'ax') and self.ax and len(self.coordination_numbers) > 0:
            try:
                # Rebuild plot with current settings
                self.plot_histogram()
                messagebox.showinfo("Success", "Plot appearance updated successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update plot: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No plot to update. Please calculate coordination numbers first.")
    
    def load_xyz_file(self):
        """Load XYZ file"""
        filename = filedialog.askopenfilename(
            title="Select XYZ File",
            filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")]
        )
        
        if not filename:
            return
            
        try:
            self.atoms = []
            self.coords = []
            self.atom_symbols = []
            
            with open(filename, 'r') as f:
                lines = f.readlines()
                
            start_line = 0
            atom_count = 0
            
            # Find start of data
            for i, line in enumerate(lines):
                if line.strip() and line.strip()[0].isdigit():
                    try:
                        atom_count = int(line.strip())
                        start_line = i + 2
                        break
                    except:
                        pass
            
            # Read atoms
            for line in lines[start_line:start_line + atom_count]:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        self.atom_symbols.append(parts[0])
                        self.coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
            self.coords = np.array(self.coords)
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, 
                f"File: {os.path.basename(filename)}\n"
                f"Atoms: {len(self.atom_symbols)}\n"
                f"Atom types: {set(self.atom_symbols)}\n"
                f"Coordinates:\n"
                f"  X: {self.coords[:, 0].min():.2f} - {self.coords[:, 0].max():.2f}\n"
                f"  Y: {self.coords[:, 1].min():.2f} - {self.coords[:, 1].max():.2f}\n"
                f"  Z: {self.coords[:, 2].min():.2f} - {self.coords[:, 2].max():.2f}"
            )
            
            self.status_var.set(f"Loaded file: {os.path.basename(filename)}")
            
        except Exception as e:
            self.status_var.set(f"Error loading file: {str(e)}")
    
    def calculate_coordination_numbers(self):
        """Calculate coordination numbers with automatic radius determination for each atom"""
        if len(self.coords) == 0:
            self.status_var.set("First load an XYZ file")
            return
            
        try:
            self.status_var.set("Calculating coordination numbers...")
            self.root.update()
            
            # Create KDTree for efficient neighbor search
            kdtree = KDTree(self.coords)
            
            self.coordination_numbers = []
            self.neighbor_indices = []
            
            # For each atom, find its neighbors
            for i, center in enumerate(self.coords):
                # Find distance to the nearest neighbor (excluding itself)
                distances, indices = kdtree.query(center, k=2)  # k=2: itself + nearest neighbor
                
                if len(distances) > 1:
                    # Distance to the nearest neighbor (first element is itself)
                    r_min = distances[1]
                    
                    # Determine cutoff radius: 1.001 * distance to nearest neighbor
                    cutoff_radius = 1.001 * r_min
                    
                    # Find all neighbors within this radius
                    neighbor_indices = kdtree.query_ball_point(center, cutoff_radius)
                    
                    # Exclude the atom itself from the neighbor list
                    neighbor_indices = [idx for idx in neighbor_indices if idx != i]
                    
                    # Store coordination number and neighbor indices
                    self.coordination_numbers.append(len(neighbor_indices))
                    self.neighbor_indices.append(neighbor_indices)
                else:
                    # If no neighbors
                    self.coordination_numbers.append(0)
                    self.neighbor_indices.append([])
            
            # Convert to numpy array
            self.coordination_numbers = np.array(self.coordination_numbers)
            
            # Update information
            self.update_info_text()
            
            # Plot histogram
            self.plot_histogram()
            
            self.status_var.set(f"Calculated coordination numbers for {len(self.coords)} atoms")
            
        except Exception as e:
            self.status_var.set(f"Calculation error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_info_text(self):
        """Update text information"""
        if len(self.coordination_numbers) > 0:
            # Coordination number statistics
            unique_cn, counts = np.unique(self.coordination_numbers, return_counts=True)
            
            info = f"COORDINATION NUMBER STATISTICS:\n"
            info += f"Total atoms: {len(self.coords)}\n"
            info += f"Mean CN: {np.mean(self.coordination_numbers):.2f}\n"
            info += f"Median CN: {np.median(self.coordination_numbers):.2f}\n"
            info += f"Std deviation: {np.std(self.coordination_numbers):.2f}\n"
            info += f"Min CN: {np.min(self.coordination_numbers)}\n"
            info += f"Max CN: {np.max(self.coordination_numbers)}\n\n"
            
            info += f"DISTRIBUTION:\n"
            info += f"CN  Count  Percentage\n"
            info += "-" * 30 + "\n"
            
            for cn, count in zip(unique_cn, counts):
                percentage = (count / len(self.coords)) * 100
                info += f"{cn:2d}  {count:5d}  {percentage:6.2f}%\n"
            
            # Information about the most common coordination numbers
            if len(counts) > 0:
                max_count_idx = np.argmax(counts)
                most_common_cn = unique_cn[max_count_idx]
                most_common_count = counts[max_count_idx]
                most_common_percentage = (most_common_count / len(self.coords)) * 100
                
                info += f"\nMost common CN: {most_common_cn} ({most_common_percentage:.1f}% of atoms)\n"
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)
    
    def plot_histogram(self):
        """Plot histogram of coordination number distribution"""
        if len(self.coordination_numbers) == 0:
            return
        
        # Clear previous plot
        self.ax.clear()
        
        # Apply font settings
        self.apply_font_settings()
        
        # Determine histogram range
        min_cn = max(0, self.min_neighbors.get())
        max_cn = min(self.max_neighbors.get(), np.max(self.coordination_numbers) + 1)
        
        # Create bins
        bins = np.arange(min_cn - 0.5, max_cn + 0.5, 1)
        
        # Build histogram
        hist, bin_edges = np.histogram(self.coordination_numbers, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Visualize histogram
        bars = self.ax.bar(bin_centers, hist, width=0.8, 
                          color=self.hist_color.get(), alpha=0.7,
                          edgecolor='black', linewidth=1)
        
        # Add values above bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                self.ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom',
                            fontsize=9, fontweight='bold')
        
        # Set axes and title
        self.ax.set_xlabel('Coordination Number', fontsize=self.axis_label_fontsize.get())
        self.ax.set_ylabel('Number of Atoms', fontsize=self.axis_label_fontsize.get())
        self.ax.set_title('Distribution of Coordination Numbers', 
                         fontsize=self.title_fontsize.get(), fontweight='bold')
        
        # Set integer values on X-axis
        self.ax.set_xticks(np.arange(min_cn, max_cn, 1))
        
        # Add grid
        self.ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Auto-scale Y-axis
        self.ax.set_ylim(0, np.max(hist) * 1.2)
        
        # Update canvas
        self.fig.tight_layout()
        self.canvas.draw()
    
    def save_atoms_with_target_cn(self):
        """Save atoms with a given coordination number and their neighbors"""
        if len(self.coordination_numbers) == 0:
            messagebox.showwarning("Warning", "No coordination numbers calculated. Calculate first.")
            return
        
        target_cn = self.target_coordination.get()
        
        # Find atoms with the given coordination number
        target_indices = np.where(self.coordination_numbers == target_cn)[0]
        
        if len(target_indices) == 0:
            messagebox.showinfo("Info", f"No atoms found with coordination number {target_cn}")
            return
        
        # Ask for filename to save
        filename = filedialog.asksaveasfilename(
            title="Save Atoms with Target CN",
            defaultextension=".xyz",
            filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")],
            initialfile=f"atoms_CN_{target_cn}.xyz"
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'w') as f:

                Count_n = 0
                # For each atom with target CN, create a separate frame
                for idx in target_indices:
                    # Neighbor indices for this atom
                    neighbor_indices = self.neighbor_indices[idx]
                    
                    # Total number of atoms in the frame: central atom + all its neighbors
                    Count_n += 1 + len(neighbor_indices)
                
                f.write(f"{Count_n}\n")
                f.write("# (c) 2025 a-prokhoda (Alexander S. Prokhoda) # Part of Quasicrystals Research Project\n")


                # For each atom with target CN, create a separate frame
                for idx in target_indices:
                    # Neighbor indices for this atom
                    neighbor_indices = self.neighbor_indices[idx]
                    
                    # Total number of atoms in the frame: central atom + all its neighbors
                    total_atoms = 1 + len(neighbor_indices)
                 
                    # Write central atom
                    x, y, z = self.coords[idx]
                    f.write(f"{self.central_atom_symbol.get()} {x:.6f} {y:.6f} {z:.6f}\n")
                    
                    # Write neighbors
                    for neighbor_idx in neighbor_indices:
                        x, y, z = self.coords[neighbor_idx]
                        f.write(f"{self.neighbor_atom_symbol.get()} {x:.6f} {y:.6f} {z:.6f}\n")
            
            messagebox.showinfo("Success", 
                              f"Saved {len(target_indices)} atoms with CN={target_cn} to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {str(e)}")
    
    def save_all_environments(self):
        """Save all environments of atoms with a given coordination number into one file"""
        if len(self.coordination_numbers) == 0:
            messagebox.showwarning("Warning", "No coordination numbers calculated. Calculate first.")
            return
        
        target_cn = self.target_coordination.get()
        
        # Find atoms with the given coordination number
        target_indices = np.where(self.coordination_numbers == target_cn)[0]
        
        if len(target_indices) == 0:
            messagebox.showinfo("Info", f"No atoms found with coordination number {target_cn}")
            return
        
        # Ask for filename to save
        filename = filedialog.asksaveasfilename(
            title="Save All Environments",
            defaultextension=".xyz",
            filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")],
            initialfile=f"all_environments_CN_{target_cn}.xyz"
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'w') as f:
                # Write all environments in one frame
                total_atoms = 0
                all_atoms = []
                all_symbols = []
                
                for idx in target_indices:
                    # Central atom
                    all_atoms.append(self.coords[idx])
                    all_symbols.append(self.central_atom_symbol.get())
                    
                    # Neighbors
                    for neighbor_idx in self.neighbor_indices[idx]:
                        all_atoms.append(self.coords[neighbor_idx])
                        all_symbols.append(self.neighbor_atom_symbol.get())
                
                total_atoms = len(all_atoms)
                
                # Write number of atoms
                f.write(f"{total_atoms}\n")
                
                # Comment
                f.write(f"All environments with CN={target_cn}. Total: {len(target_indices)} central atoms\n")
                
                # Write all atoms
                for symbol, coords in zip(all_symbols, all_atoms):
                    x, y, z = coords
                    f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")
            
            messagebox.showinfo("Success", 
                              f"Saved {len(target_indices)} environments with CN={target_cn} to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {str(e)}")
    
    def save_high_quality_plot(self):
        """Save plot in high quality for publication"""
        if len(self.coordination_numbers) == 0:
            messagebox.showwarning("Warning", "No plot to save. Calculate coordination numbers first.")
            return
            
        # Ask for filename
        filename = filedialog.asksaveasfilename(
            title="Save High-Quality Plot",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("TIFF files", "*.tiff;*.tif"),
                ("EPS files", "*.eps"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ]
        )
        
        if not filename:
            return
        
        try:
            # Get DPI setting
            dpi = self.plot_dpi.get()
            if dpi < 300 or dpi > 1200:
                messagebox.showwarning("Warning", "DPI should be between 300 and 1200. Using 600.")
                dpi = 600
            
            # Get file extension
            ext = os.path.splitext(filename)[1].lower()
            
            # Save with high quality
            if ext in ['.png', '.tiff', '.tif']:
                # For raster formats, use high DPI
                self.fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                                facecolor='white', edgecolor='none',
                                transparent=False, pad_inches=0.1)
            elif ext in ['.pdf', '.eps', '.svg']:
                # For vector formats
                self.fig.savefig(filename, format=ext[1:], bbox_inches='tight',
                                facecolor='white', edgecolor='none')
            else:
                # Default save as PNG
                filename = filename + '.png'
                self.fig.savefig(filename, dpi=dpi, bbox_inches='tight',
                                facecolor='white', edgecolor='none')
            
            messagebox.showinfo("Success", 
                              f"High-quality plot saved to:\n{filename}\n"
                              f"Resolution: {dpi} DPI\n"
                              f"Font: {self.font_family.get()}, "
                              f"Title: {self.title_fontsize.get()}pt, "
                              f"Labels: {self.axis_label_fontsize.get()}pt, "
                              f"Ticks: {self.axis_tick_fontsize.get()}pt")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plot: {str(e)}")

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = CoordinationNumberAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
