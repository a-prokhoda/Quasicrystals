import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('TkAgg')

class PairRDFAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Pair Radial Distribution Function (g(r)) Analyzer")
        self.root.geometry("1400x900")
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        
        # RDF parameters
        self.r_max = tk.DoubleVar(value=10.0)
        self.bin_width = tk.DoubleVar(value=0.01)
        self.auto_r_max = tk.BooleanVar(value=False)
        
        # Atom type selection
        self.atom_type1 = tk.StringVar(value='All')
        self.atom_type2 = tk.StringVar(value='All')
        
        # Plot appearance
        self.font_size_labels = tk.IntVar(value=14)
        self.font_size_ticks = tk.IntVar(value=12)
        self.line_color = tk.StringVar(value='#1f77b4')
        self.line_width = tk.DoubleVar(value=2.0)
        self.plot_dpi = tk.IntVar(value=600)
        
        # Current data
        self.current_atoms = None
        self.current_symbols = None
        self.current_comment = ""
        self.r_values = None
        self.g_r_values = None
        
        # Figure for visualization
        self.fig = None
        self.canvas = None
        self.ax = None
        
        # Colors for lines
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Common atom symbols
        self.atom_symbols = ['All', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                           'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                           'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                           'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                           'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                           'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                           'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                           'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                           'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main container with left and right panels
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - control
        left_frame = ttk.Frame(main_paned, width=450)
        main_paned.add(left_frame, weight=1)
        
        # Right panel - visualization
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        
        # Configure left panel
        self.create_control_panel(left_frame)
        
        # Configure right panel
        self.create_visualization_panel(right_frame)
        
    def create_control_panel(self, parent):
        # Scrollable frame for left panel
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scroll elements
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        control_frame = ttk.Frame(scrollable_frame, padding="15")
        control_frame.pack(fill=tk.BOTH, expand=True)
        
        row = 0
        
        # Title
        title_label = ttk.Label(control_frame, text="Pair Radial Distribution Function g(r)", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=row, column=0, columnspan=3, pady=(0, 20))
        row += 1
        
        # Section: File Operations
        ttk.Label(control_frame, text="File Operations", 
                 font=('Arial', 12, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        row += 1
        
        # Input file
        ttk.Label(control_frame, text="Input File:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        ttk.Entry(control_frame, textvariable=self.input_file, width=35).grid(
            row=row, column=1, pady=3)
        ttk.Button(control_frame, text="Browse...", 
                  command=self.browse_input_file).grid(
            row=row, column=2, pady=3, padx=(5,0))
        row += 1
        
        # Load button
        load_btn = ttk.Button(control_frame, text="Load XYZ File", 
                             command=self.load_xyz_file, width=20)
        load_btn.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=15)
        row += 1
        
        # Section: Atom Type Selection
        ttk.Label(control_frame, text="Atom Type Selection", 
                 font=('Arial', 12, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        row += 1
        
        # Atom type 1
        ttk.Label(control_frame, text="Atom Type 1:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        atom1_combo = ttk.Combobox(control_frame, textvariable=self.atom_type1,
                                  values=self.atom_symbols, width=15, state='readonly')
        atom1_combo.grid(row=row, column=1, sticky=tk.W, pady=3)
        row += 1
        
        # Atom type 2
        ttk.Label(control_frame, text="Atom Type 2:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        atom2_combo = ttk.Combobox(control_frame, textvariable=self.atom_type2,
                                  values=self.atom_symbols, width=15, state='readonly')
        atom2_combo.grid(row=row, column=1, sticky=tk.W, pady=3)
        row += 1
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=15)
        row += 1
        
        # Section: RDF Parameters
        ttk.Label(control_frame, text="PRDF Parameters", 
                 font=('Arial', 12, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        row += 1
        
        # Max distance (r_max)
        ttk.Label(control_frame, text="Maximum Distance (r_max):").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        rmax_frame = ttk.Frame(control_frame)
        rmax_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=3)
        
        rmax_entry = ttk.Entry(rmax_frame, textvariable=self.r_max, width=8)
        rmax_entry.pack(side=tk.LEFT)
        ttk.Label(rmax_frame, text="Å").pack(side=tk.LEFT, padx=5)
        row += 1
        
        # Auto r_max
        auto_rmax_check = ttk.Checkbutton(control_frame, text="Auto-calculate r_max from data",
                                         variable=self.auto_r_max)
        auto_rmax_check.grid(row=row, column=0, columnspan=3, pady=5, sticky=tk.W)
        row += 1
        
        # Bin width
        ttk.Label(control_frame, text="Bin Width:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        bin_frame = ttk.Frame(control_frame)
        bin_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=3)
        
        bin_entry = ttk.Entry(bin_frame, textvariable=self.bin_width, width=8)
        bin_entry.pack(side=tk.LEFT)
        ttk.Label(bin_frame, text="Å").pack(side=tk.LEFT, padx=5)
        row += 1
        
        # Calculate RDF button
        calc_btn = ttk.Button(control_frame, text="Calculate g(r)", 
                             command=self.calculate_g_r, width=20)
        calc_btn.grid(row=row, column=0, columnspan=3, pady=15)
        row += 1
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=15)
        row += 1
        
        # Section: Plot Appearance
        ttk.Label(control_frame, text="Plot Appearance", 
                 font=('Arial', 12, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        row += 1
        
        # Font size - labels
        ttk.Label(control_frame, text="Axis Label Font Size:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        label_font_frame = ttk.Frame(control_frame)
        label_font_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=3)
        
        ttk.Entry(label_font_frame, textvariable=self.font_size_labels, width=6).pack(side=tk.LEFT)
        ttk.Label(label_font_frame, text="pt").pack(side=tk.LEFT, padx=5)
        row += 1
        
        # Font size - ticks
        ttk.Label(control_frame, text="Axis Tick Font Size:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        tick_font_frame = ttk.Frame(control_frame)
        tick_font_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=3)
        
        ttk.Entry(tick_font_frame, textvariable=self.font_size_ticks, width=6).pack(side=tk.LEFT)
        ttk.Label(tick_font_frame, text="pt").pack(side=tk.LEFT, padx=5)
        row += 1
        
        # Line color
        ttk.Label(control_frame, text="Line Color:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        color_frame = ttk.Frame(control_frame)
        color_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=3)
        
        color_combo = ttk.Combobox(color_frame, textvariable=self.line_color,
                                  values=self.colors, width=10, state='readonly')
        color_combo.pack(side=tk.LEFT)
        # Color preview
        self.color_preview = tk.Label(color_frame, width=3, bg=self.line_color.get())
        self.color_preview.pack(side=tk.LEFT, padx=5)
        color_combo.bind('<<ComboboxSelected>>', self.update_color_preview)
        row += 1
        
        # Line width
        ttk.Label(control_frame, text="Line Width:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        width_frame = ttk.Frame(control_frame)
        width_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=3)
        
        ttk.Entry(width_frame, textvariable=self.line_width, width=6).pack(side=tk.LEFT)
        ttk.Label(width_frame, text="(1-5)").pack(side=tk.LEFT, padx=5)
        row += 1
        
        # DPI for saving
        ttk.Label(control_frame, text="DPI for Saving:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        dpi_frame = ttk.Frame(control_frame)
        dpi_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=3)
        
        ttk.Entry(dpi_frame, textvariable=self.plot_dpi, width=6).pack(side=tk.LEFT)
        ttk.Label(dpi_frame, text="(300-1200)").pack(side=tk.LEFT, padx=5)
        row += 1
        
        # Update plot button
        update_btn = ttk.Button(control_frame, text="Update Plot Appearance", 
                               command=self.update_plot_appearance, width=20)
        update_btn.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1
        
        # Save high-quality plot button
        save_plot_btn = ttk.Button(control_frame, text="Save High-Quality Plot", 
                                  command=self.save_high_quality_plot, width=20)
        save_plot_btn.grid(row=row, column=0, columnspan=3, pady=5)
        row += 1
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=15)
        row += 1
        
        # Section: Results
        ttk.Label(control_frame, text="Results", 
                 font=('Arial', 12, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        row += 1
        
        # Results text
        self.results_text = tk.Text(control_frame, height=8, width=45, wrap=tk.WORD,
                                   font=('Courier New', 9))
        self.results_text.grid(row=row, column=0, columnspan=3, pady=5)
        row += 1
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=15)
        row += 1
        
        # Section: Export
        ttk.Label(control_frame, text="Export Results", 
                 font=('Arial', 12, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        row += 1
        
        # Output file
        ttk.Label(control_frame, text="Output File:").grid(
            row=row, column=0, sticky=tk.W, pady=3)
        ttk.Entry(control_frame, textvariable=self.output_file, width=35).grid(
            row=row, column=1, pady=3)
        ttk.Button(control_frame, text="Browse...", 
                  command=self.browse_output_file).grid(
            row=row, column=2, pady=3, padx=(5,0))
        row += 1
        
        # Save button
        save_btn = ttk.Button(control_frame, text="Save g(r) Data", 
                             command=self.save_g_r_data, width=20)
        save_btn.grid(row=row, column=0, columnspan=3, pady=10)
        
    def create_visualization_panel(self, parent):
        # Create matplotlib figure with subplots
        self.fig = Figure(figsize=(10, 7), dpi=100)
        
        # Subplot for g(r)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Distance r (Å)', fontsize=self.font_size_labels.get())
        self.ax.set_ylabel('g(r)', fontsize=self.font_size_labels.get())
        self.ax.set_title('Pair Radial Distribution Function g(r)', fontsize=16, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(0, 10)
        self.ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)  # Reference line at g(r)=1
        
        # Adjust layout
        self.fig.tight_layout(pad=3.0)
        
        # Create Tkinter canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar for navigation
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar_frame, text="Reset View", 
                  command=self.reset_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="Toggle Grid", 
                  command=self.toggle_grid).pack(side=tk.LEFT, padx=2)
        
        # Status
        self.view_status = ttk.Label(toolbar_frame, text="Ready")
        self.view_status.pack(side=tk.RIGHT, padx=10)
        
    def update_color_preview(self, event=None):
        """Update color preview label"""
        self.color_preview.config(bg=self.line_color.get())
        
    def browse_input_file(self):
        """Browse for input XYZ file"""
        filename = filedialog.askopenfilename(
            title="Select XYZ File",
            filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
            # Auto-generate output filename
            if not self.output_file.get():
                base, ext = os.path.splitext(filename)
                output_filename = f"{base}_g_r_data.txt"
                self.output_file.set(output_filename)
                
    def browse_output_file(self):
        """Browse for output file"""
        filename = filedialog.asksaveasfilename(
            title="Save g(r) Data",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.output_file.set(filename)
            
    def load_xyz_file(self):
        """Load XYZ file"""
        filename = self.input_file.get()
        if not filename:
            messagebox.showerror("Error", "Please select an input file!")
            return
            
        if not os.path.exists(filename):
            messagebox.showerror("Error", f"File not found: {filename}")
            return
            
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                
            # Skip empty lines
            lines = [line for line in lines if line.strip()]
            
            # Read number of atoms
            num_atoms = int(lines[0].strip())
            
            # Read comment
            comment = lines[1].strip() if len(lines) > 1 else ""
            
            # Read atoms
            atoms = []
            symbols = []
            
            for i in range(2, min(2 + num_atoms, len(lines))):
                parts = lines[i].split()
                if len(parts) >= 4:
                    symbols.append(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    atoms.append([x, y, z])
                    
            self.current_atoms = np.array(atoms, dtype=np.float64)
            self.current_symbols = symbols
            self.current_comment = comment
            
            # Update unique atom types in combobox
            unique_atoms = set(symbols)
            self.atom_symbols = ['All'] + sorted(list(unique_atoms))
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Loaded {num_atoms} atoms from {os.path.basename(filename)}\n")
            self.results_text.insert(tk.END, f"Unique atom types: {', '.join(sorted(unique_atoms))}\n")
            
            messagebox.showinfo("Success", f"Loaded {num_atoms} atoms from {os.path.basename(filename)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def calculate_g_r(self):
        """Calculate Pair Radial Distribution Function g(r)"""
        if self.current_atoms is None:
            messagebox.showwarning("Warning", "No structure loaded!")
            return
            
        try:
            # Get parameters
            r_max = self.r_max.get()
            bin_width = self.bin_width.get()
            
            # Ensure r_max is at least bin_width
            if r_max < bin_width:
                r_max = bin_width * 10
                self.r_max.set(r_max)
                messagebox.showwarning("Warning", f"r_max increased to {r_max:.2f} Å to accommodate bin width")
            
            # Auto-calculate r_max if needed
            if self.auto_r_max.get():
                # Calculate maximum possible distance in the system
                all_distances = []
                chunk_size = 500
                for i in range(0, len(self.current_atoms), chunk_size):
                    end_i = min(i + chunk_size, len(self.current_atoms))
                    chunk_coords = self.current_atoms[i:end_i]
                    diff = chunk_coords[:, np.newaxis, :] - self.current_atoms[np.newaxis, :, :]
                    chunk_dists = np.sqrt(np.sum(diff**2, axis=2))
                    all_distances.append(chunk_dists.flatten())
                
                if all_distances:
                    all_distances = np.concatenate(all_distances)
                    r_max = np.max(all_distances) * 0.5  # Use half of max distance
                    self.r_max.set(round(r_max, 2))
            
            # Get atom types for analysis
            atom_type1 = self.atom_type1.get()
            atom_type2 = self.atom_type2.get()
            
            # Get indices of selected atom types
            if atom_type1 == 'All':
                indices1 = np.arange(len(self.current_atoms), dtype=int)
            else:
                indices1 = np.array([i for i, sym in enumerate(self.current_symbols) 
                                   if sym == atom_type1], dtype=int)
            
            if atom_type2 == 'All':
                indices2 = np.arange(len(self.current_atoms), dtype=int)
            else:
                indices2 = np.array([i for i, sym in enumerate(self.current_symbols) 
                                   if sym == atom_type2], dtype=int)
            
            if len(indices1) == 0 or len(indices2) == 0:
                messagebox.showerror("Error", "No atoms of selected type found!")
                return
            
            # Get coordinates
            coords1 = self.current_atoms[indices1]
            coords2 = self.current_atoms[indices2]
            
            # Calculate g(r) - the correct Pair RDF
            print(f"Calculating g(r) for {atom_type1}-{atom_type2} pairs...")
            print(f"Number of atoms type1: {len(indices1)}, type2: {len(indices2)}")
            
            # Create bins
            n_bins = int(r_max / bin_width)
            bin_edges = np.linspace(0, r_max, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Initialize histogram
            hist = np.zeros(n_bins)
            
            # Calculate distances and histogram in chunks to save memory
            chunk_size = 200
            n1 = len(coords1)
            
            for i in range(0, n1, chunk_size):
                end_i = min(i + chunk_size, n1)
                chunk_coords1 = coords1[i:end_i]
                
                # Calculate distances for this chunk
                # Broadcasting: shape (chunk_size, n2, 3)
                diff = chunk_coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
                distances = np.sqrt(np.sum(diff**2, axis=2))
                
                # If same atom types, exclude self-distances (diagonal for each chunk)
                if atom_type1 == atom_type2 and i == 0:
                    # For first chunk, exclude diagonal
                    np.fill_diagonal(distances[:min(chunk_size, n1)], np.inf)
                elif atom_type1 == atom_type2:
                    # For other chunks, all distances are valid
                    pass
                
                # Flatten and filter distances
                distances_flat = distances.flatten()
                valid_distances = distances_flat[(distances_flat > 0) & (distances_flat <= r_max)]
                
                # Update histogram
                hist_chunk, _ = np.histogram(valid_distances, bins=bin_edges)
                hist += hist_chunk
            
            # Calculate number of pairs
            n1_total = len(indices1)
            n2_total = len(indices2)
            
            if atom_type1 == atom_type2:
                # For same atom types: N*(N-1)/2 pairs
                n_pairs = n1_total * (n2_total - 1) / 2
            else:
                # For different atom types: N1 * N2 pairs
                n_pairs = n1_total * n2_total
            
            print(f"Total number of pairs: {n_pairs}")
            
            # Calculate volumes of spherical shells
            r_lower = bin_edges[:-1]
            r_upper = bin_edges[1:]
            shell_volumes = (4/3) * np.pi * (r_upper**3 - r_lower**3)
            
            # Avoid division by zero
            shell_volumes = np.where(shell_volumes == 0, 1e-12, shell_volumes)
            
            # Calculate average number density
            # Estimate system volume from bounding box
            min_coords = np.min(self.current_atoms, axis=0)
            max_coords = np.max(self.current_atoms, axis=0)
            box_size = max_coords - min_coords
            system_volume = np.prod(box_size)
            
            if system_volume <= 0:
                # Fallback: use sphere volume with radius = max distance from center
                center = np.mean(self.current_atoms, axis=0)
                distances_to_center = np.linalg.norm(self.current_atoms - center, axis=1)
                max_radius = np.max(distances_to_center)
                system_volume = (4/3) * np.pi * (max_radius**3)
            
            # Calculate number densities
            if atom_type1 == atom_type2:
                # For same atom types
                rho = n1_total / system_volume
                ideal_counts = rho * shell_volumes * n1_total
            else:
                # For different atom types
                rho1 = n1_total / system_volume
                rho2 = n2_total / system_volume
                ideal_counts = rho2 * shell_volumes * n1_total
            
            # Avoid division by zero in ideal_counts
            ideal_counts = np.where(ideal_counts == 0, 1e-12, ideal_counts)
            
            # Calculate g(r)
            g_r = hist / ideal_counts
            
            # Store results
            self.r_values = bin_centers
            self.g_r_values = g_r
            
            # Clear previous plot
            self.ax.clear()
            
            # Plot g(r)
            self.ax.plot(self.r_values, self.g_r_values, 
                        color=self.line_color.get(), 
                        linewidth=self.line_width.get(),
                        label=f'{atom_type1}-{atom_type2}')
            
            # Add horizontal line at g(r)=1
            self.ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='g(r)=1')
            
            # Configure plot with current font sizes
            self.ax.set_xlabel('Distance r (Å)', fontsize=self.font_size_labels.get())
            self.ax.set_ylabel('g(r)', fontsize=self.font_size_labels.get())
            
            # Create title
            if atom_type1 == atom_type2:
                title = f'Pair PRDF g(r): {atom_type1}-{atom_type1}'
            else:
                title = f'Pair PRDF g(r): {atom_type1}-{atom_type2}'
            
            self.ax.set_title(title, fontsize=16, fontweight='bold')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlim(0, r_max)
            
            # Auto-scale Y-axis with some margin
            g_r_valid = g_r[~np.isnan(g_r) & ~np.isinf(g_r)]
            if len(g_r_valid) > 0:
                y_min = max(0, np.min(g_r_valid) * 0.9)
                y_max = np.max(g_r_valid) * 1.1
                self.ax.set_ylim(y_min, y_max)
            
            # Set tick font size
            self.ax.tick_params(axis='both', which='major', labelsize=self.font_size_ticks.get())
            
            # Add legend
            self.ax.legend(loc='upper right', fontsize=10)
            
            # Update canvas
            self.fig.tight_layout(pad=3.0)
            self.canvas.draw()
            
            # Update results display
            self.update_results_display(atom_type1, atom_type2, n_pairs, system_volume)
            
            # Update status
            analysis_label = f"{atom_type1}-{atom_type2}"
            self.view_status.config(
                text=f"g(r) calculated: {analysis_label}")
            
            messagebox.showinfo("Success", 
                              f"g(r) calculation complete!\n"
                              f"Number of pairs: {int(n_pairs)}\n"
                              f"System volume: {system_volume:.2f} Å³\n"
                              f"Peak g(r) value: {np.max(g_r_valid):.2f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate g(r): {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def update_results_display(self, atom_type1, atom_type2, n_pairs, system_volume):
        """Update results display"""
        if self.r_values is not None and self.g_r_values is not None:
            # Find peak positions and values
            g_r_valid = self.g_r_values[~np.isnan(self.g_r_values) & ~np.isinf(self.g_r_values)]
            r_valid = self.r_values[~np.isnan(self.g_r_values) & ~np.isinf(self.g_r_values)]
            
            if len(g_r_valid) > 0:
                # Find first few peaks
                from scipy.signal import find_peaks
                peaks, properties = find_peaks(g_r_valid, height=1.0, distance=10)
                
                results = f"g(r) RESULTS:\n"
                results += f"Atom types: {atom_type1}-{atom_type2}\n"
                results += f"Number of pairs: {int(n_pairs)}\n"
                results += f"Estimated system volume: {system_volume:.2f} Å³\n"
                results += f"Maximum g(r): {np.max(g_r_valid):.4f}\n"
                results += f"Minimum g(r): {np.min(g_r_valid):.4f}\n"
                
                if len(peaks) > 0:
                    results += f"\nPEAK POSITIONS:\n"
                    for i, peak_idx in enumerate(peaks[:5]):  # Show first 5 peaks
                        results += f"  Peak {i+1}: r = {r_valid[peak_idx]:.3f} Å, g(r) = {g_r_valid[peak_idx]:.3f}\n"
                
                # Calculate coordination number for first peak
                if len(peaks) > 0:
                    first_peak_idx = peaks[0]
                    peak_start = max(0, first_peak_idx - 5)
                    peak_end = min(len(r_valid), first_peak_idx + 5)
                    
                    # Integration of g(r) around first peak
                    r_peak = r_valid[peak_start:peak_end]
                    g_r_peak = g_r_valid[peak_start:peak_end]
                    
                    # Simple integration (trapezoidal rule)
                    if len(r_peak) > 1:
                        coord_number = np.trapz(g_r_peak * 4 * np.pi * r_peak**2, r_peak)
                        results += f"\nFIRST COORDINATION SHELL:\n"
                        results += f"  Peak center: {r_valid[first_peak_idx]:.3f} Å\n"
                        results += f"  Peak height: {g_r_valid[first_peak_idx]:.3f}\n"
                        results += f"  Approx. coordination number: {coord_number:.2f}\n"
                
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(1.0, results)
    
    def update_plot_appearance(self):
        """Update plot appearance based on current settings"""
        if self.ax is None or self.r_values is None:
            messagebox.showwarning("Warning", "No plot to update. Please calculate g(r) first.")
            return
        
        try:
            # Update line properties if there are lines
            if self.ax.lines:
                for line in self.ax.lines:
                    if line.get_label() != 'g(r)=1':  # Don't change the reference line
                        line.set_color(self.line_color.get())
                        line.set_linewidth(self.line_width.get())
            
            # Update font sizes
            self.ax.set_xlabel(self.ax.get_xlabel(), fontsize=self.font_size_labels.get())
            self.ax.set_ylabel(self.ax.get_ylabel(), fontsize=self.font_size_labels.get())
            
            # Update tick font size
            self.ax.tick_params(axis='both', which='major', labelsize=self.font_size_ticks.get())
            
            # Update title font size
            current_title = self.ax.get_title()
            self.ax.set_title(current_title, fontsize=16, fontweight='bold')
            
            # Update canvas
            self.canvas.draw()
            
            messagebox.showinfo("Success", "Plot appearance updated successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update plot appearance: {str(e)}")
    
    def save_high_quality_plot(self):
        """Save plot as high-quality PNG for publication"""
        if self.fig is None:
            messagebox.showerror("Error", "No plot to save!")
            return
        
        # Ask for save location
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
            
            # Update plot with current settings before saving
            self.update_plot_appearance()
            
            # Save with high quality settings
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
                # Default to PNG
                filename = filename + '.png'
                self.fig.savefig(filename, dpi=dpi, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
            
            messagebox.showinfo("Success", 
                              f"High-quality plot saved to:\n{filename}\n"
                              f"Resolution: {dpi} DPI\n"
                              f"Font sizes: labels={self.font_size_labels.get()}pt, "
                              f"ticks={self.font_size_ticks.get()}pt")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plot: {str(e)}")
    
    def save_g_r_data(self):
        """Save g(r) data to file"""
        if self.r_values is None or self.g_r_values is None:
            messagebox.showerror("Error", "No g(r) data to save!")
            return
            
        filename = self.output_file.get()
        if not filename:
            messagebox.showerror("Error", "Please specify output file!")
            return
            
        try:
            # Get parameters for file header
            r_max = self.r_max.get()
            bin_width = self.bin_width.get()
            atom_type1 = self.atom_type1.get()
            atom_type2 = self.atom_type2.get()
            
            # Save data
            with open(filename, 'w', encoding='utf-8') as f:
                # Write header
                f.write("# Pair Radial Distribution Function g(r) Data\n")
                f.write(f"# Generated from: {self.input_file.get()}\n")
                f.write(f"# Atom types: {atom_type1} - {atom_type2}\n")
                f.write(f"# Analysis range: 0 - {r_max} Å\n")
                f.write(f"# Bin width: {bin_width} Å\n")
                f.write(f"# Columns: r(Å), g(r)\n")
                
                # Write data
                for r, g_r in zip(self.r_values, self.g_r_values):
                    if not np.isnan(g_r) and not np.isinf(g_r):
                        f.write(f"{r:.6f} {g_r:.6f}\n")
            
            messagebox.showinfo("Success", f"g(r) data saved to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {str(e)}")
    
    def reset_view(self):
        """Reset plot view"""
        if self.ax:
            r_max = self.r_max.get()
            self.ax.set_xlim(0, r_max)
            if self.g_r_values is not None:
                g_r_valid = self.g_r_values[~np.isnan(self.g_r_values) & ~np.isinf(self.g_r_values)]
                if len(g_r_valid) > 0:
                    y_min = max(0, np.min(g_r_valid) * 0.9)
                    y_max = np.max(g_r_valid) * 1.1
                    self.ax.set_ylim(y_min, y_max)
            self.canvas.draw()
    
    def toggle_grid(self):
        """Toggle grid on plot"""
        if self.ax:
            self.ax.grid(not self.ax.get_xgridlines()[0].get_visible())
            self.canvas.draw()

def main():
    root = tk.Tk()
    
    # Set application icon if available
    try:
        root.iconbitmap(default='icon.ico')
    except:
        pass
    
    app = PairRDFAnalyzer(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Minimum size
    root.minsize(1200, 800)
    
    root.mainloop()

if __name__ == "__main__":
    main()
