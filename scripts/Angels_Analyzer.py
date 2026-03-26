import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from scipy.spatial import KDTree
import math
from scipy.signal import find_peaks
import warnings
import os

warnings.filterwarnings('ignore')

class XYZAngleAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("XYZ File Analyzer - Angle Distribution")
        self.root.geometry("1200x800")
        
        # Variables
        self.atoms = []
        self.coords = []
        self.cutoff_radius = tk.StringVar(value="1.8")
        self.angles_history = []
        
        # Font and plot settings for publication
        self.axis_label_fontsize = tk.IntVar(value=14)
        self.axis_tick_fontsize = tk.IntVar(value=12)
        self.title_fontsize = tk.IntVar(value=16)
        self.font_family = tk.StringVar(value='Arial')
        self.plot_dpi = tk.IntVar(value=600)
        self.hist_color = tk.StringVar(value='#1f77b4')
        self.line_color = tk.StringVar(value='red')
        
        # Available fonts
        self.available_fonts = ['Arial', 'Times New Roman', 'Courier New', 
                               'Helvetica', 'DejaVu Sans', 'serif', 'sans-serif']
        
        # Colors for histogram
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Create UI
        self.create_widgets()
        
    def create_widgets(self):
        """Create the user interface"""
        # Main container
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left control panel
        control_frame = ttk.Frame(main_paned, width=350)
        main_paned.add(control_frame, weight=1)
        
        # Main area for plots
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
        
        # Pack scroll elements
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Control content
        content_frame = ttk.Frame(scrollable_frame, padding="10")
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(content_frame, text="XYZ File Analyzer", 
                              font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # File loading button
        tk.Button(content_frame, text="Load XYZ File", 
                 command=self.load_file, width=20).pack(pady=10)
        
        # Analysis parameters
        param_frame = ttk.LabelFrame(content_frame, text="Analysis Parameters", padding=10)
        param_frame.pack(pady=10, fill=tk.X)
        
        # Cutoff radius input
        tk.Label(param_frame, text="Cutoff Radius (Å):").grid(row=0, column=0, sticky='w', pady=5)
        tk.Entry(param_frame, textvariable=self.cutoff_radius, width=15).grid(row=0, column=1, pady=5, padx=5)
        
        # Calculate button
        tk.Button(param_frame, text="Calculate Angles", 
                 command=self.calculate_angles, width=20).grid(row=1, column=0, columnspan=2, pady=10)
        
        # Plot Settings for Publication
        plot_frame = ttk.LabelFrame(content_frame, text="Plot Settings for Publication", padding=10)
        plot_frame.pack(pady=10, fill=tk.X)
        
        row = 0
        
        # Font family
        tk.Label(plot_frame, text="Font Family:").grid(row=row, column=0, sticky='w', pady=3)
        font_combo = ttk.Combobox(plot_frame, textvariable=self.font_family,
                                 values=self.available_fonts, width=15, state='readonly')
        font_combo.grid(row=row, column=1, sticky='w', pady=3, padx=5)
        row += 1
        
        # Title font size
        tk.Label(plot_frame, text="Title Font Size:").grid(row=row, column=0, sticky='w', pady=3)
        tk.Entry(plot_frame, textvariable=self.title_fontsize, width=8).grid(row=row, column=1, sticky='w', pady=3, padx=5)
        tk.Label(plot_frame, text="pt").grid(row=row, column=2, sticky='w', pady=3)
        row += 1
        
        # Axis label font size
        tk.Label(plot_frame, text="Axis Label Font Size:").grid(row=row, column=0, sticky='w', pady=3)
        tk.Entry(plot_frame, textvariable=self.axis_label_fontsize, width=8).grid(row=row, column=1, sticky='w', pady=3, padx=5)
        tk.Label(plot_frame, text="pt").grid(row=row, column=2, sticky='w', pady=3)
        row += 1
        
        # Axis tick font size
        tk.Label(plot_frame, text="Axis Tick Font Size:").grid(row=row, column=0, sticky='w', pady=3)
        tk.Entry(plot_frame, textvariable=self.axis_tick_fontsize, width=8).grid(row=row, column=1, sticky='w', pady=3, padx=5)
        tk.Label(plot_frame, text="pt").grid(row=row, column=2, sticky='w', pady=3)
        row += 1
        
        # Histogram color
        tk.Label(plot_frame, text="Histogram Color:").grid(row=row, column=0, sticky='w', pady=3)
        color_frame = tk.Frame(plot_frame)
        color_frame.grid(row=row, column=1, columnspan=2, sticky='w', pady=3, padx=5)
        
        color_combo = ttk.Combobox(color_frame, textvariable=self.hist_color,
                                  values=self.colors, width=10, state='readonly')
        color_combo.pack(side=tk.LEFT)
        self.color_preview = tk.Label(color_frame, width=3, bg=self.hist_color.get())
        self.color_preview.pack(side=tk.LEFT, padx=5)
        color_combo.bind('<<ComboboxSelected>>', self.update_color_preview)
        row += 1
        
        # Line color
        tk.Label(plot_frame, text="Peak Line Color:").grid(row=row, column=0, sticky='w', pady=3)
        line_color_frame = tk.Frame(plot_frame)
        line_color_frame.grid(row=row, column=1, columnspan=2, sticky='w', pady=3, padx=5)
        
        line_color_combo = ttk.Combobox(line_color_frame, textvariable=self.line_color,
                                       values=self.colors, width=10, state='readonly')
        line_color_combo.pack(side=tk.LEFT)
        self.line_color_preview = tk.Label(line_color_frame, width=3, bg=self.line_color.get())
        self.line_color_preview.pack(side=tk.LEFT, padx=5)
        line_color_combo.bind('<<ComboboxSelected>>', self.update_line_color_preview)
        row += 1
        
        # DPI for saving
        tk.Label(plot_frame, text="DPI for Saving:").grid(row=row, column=0, sticky='w', pady=3)
        dpi_frame = tk.Frame(plot_frame)
        dpi_frame.grid(row=row, column=1, columnspan=2, sticky='w', pady=3, padx=5)
        
        tk.Entry(dpi_frame, textvariable=self.plot_dpi, width=6).pack(side=tk.LEFT)
        tk.Label(dpi_frame, text="(300-1200)").pack(side=tk.LEFT, padx=5)
        row += 1
        
        # Update plot appearance button
        tk.Button(plot_frame, text="Update Plot Appearance", 
                 command=self.update_plot_appearance, width=20).grid(row=row, column=0, columnspan=3, pady=10)
        row += 1
        
        # Save high-quality plot button
        tk.Button(plot_frame, text="Save High-Quality Plot", 
                 command=self.save_high_quality_plot, width=20).grid(row=row, column=0, columnspan=3, pady=5)
        
        # Action buttons frame
        action_frame = ttk.LabelFrame(content_frame, text="Actions", padding=10)
        action_frame.pack(pady=10, fill=tk.X)
        
        # Export button
        tk.Button(action_frame, text="Export Results", 
                 command=self.export_results, width=20).pack(pady=5)
        
        # File information display
        info_frame = ttk.LabelFrame(content_frame, text="File Information", padding=10)
        info_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.info_text = tk.Text(info_frame, height=10, width=40, wrap=tk.WORD,
                                font=('Courier New', 9))
        scrollbar_info = tk.Scrollbar(info_frame, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar_info.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_info.pack(side=tk.RIGHT, fill=tk.Y)
        
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
        """Update histogram color preview label"""
        self.color_preview.config(bg=self.hist_color.get())
    
    def update_line_color_preview(self, event=None):
        """Update peak line color preview label"""
        self.line_color_preview.config(bg=self.line_color.get())
    
    def apply_font_settings(self):
        """Apply font settings to the current plot"""
        if hasattr(self, 'ax') and self.ax:
            # Apply font family
            plt.rcParams['font.family'] = self.font_family.get()
            
            # Update title font size if exists
            if self.ax.get_title():
                self.ax.set_title(self.ax.get_title(), 
                                 fontsize=self.title_fontsize.get(), 
                                 fontweight='bold')
            
            # Update axis labels font size
            self.ax.set_xlabel(self.ax.get_xlabel(), 
                              fontsize=self.axis_label_fontsize.get())
            self.ax.set_ylabel(self.ax.get_ylabel(), 
                              fontsize=self.axis_label_fontsize.get())
            
            # Update tick font size
            self.ax.tick_params(axis='both', which='major', 
                               labelsize=self.axis_tick_fontsize.get())
    
    def update_plot_appearance(self):
        """Update plot appearance based on current settings"""
        if hasattr(self, 'ax') and self.ax and len(self.angles_history) > 0:
            try:
                # Re-plot with current settings
                self.calculate_angles()
                messagebox.showinfo("Success", "Plot appearance updated successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update plot: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No plot to update. Please calculate angles first.")
    
    def load_file(self):
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
            
            with open(filename, 'r') as f:
                lines = f.readlines()
                
            start_line = 0
            atom_count = 0
            
            for i, line in enumerate(lines):
                if line.strip() and line.strip()[0].isdigit():
                    try:
                        atom_count = int(line.strip())
                        start_line = i + 2
                        break
                    except:
                        pass
            
            for line in lines[start_line:start_line + atom_count]:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        self.atoms.append(parts[0])
                        self.coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
            self.coords = np.array(self.coords)
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, 
                f"File: {os.path.basename(filename)}\n"
                f"Atoms: {len(self.atoms)}\n"
                f"Atom types: {set(self.atoms)}\n"
                f"Coordinates:\n"
                f"  X: {self.coords[:, 0].min():.2f} - {self.coords[:, 0].max():.2f}\n"
                f"  Y: {self.coords[:, 1].min():.2f} - {self.coords[:, 1].max():.2f}\n"
                f"  Z: {self.coords[:, 2].min():.2f} - {self.coords[:, 2].max():.2f}"
            )
            
            self.status_var.set(f"Loaded file: {os.path.basename(filename)}")
            
        except Exception as e:
            self.status_var.set(f"Error loading file: {str(e)}")
    
    def calculate_angles(self):
        """Calculate angles between atoms with HIGH PRECISION"""
        if len(self.coords) == 0:
            self.status_var.set("First load an XYZ file")
            return
            
        try:
            cutoff = float(self.cutoff_radius.get())
            if cutoff <= 0:
                self.status_var.set("Radius must be positive")
                return
            
            self.status_var.set("Calculating angles...")
            self.root.update()
            
            angles = []
            
            # Use KDTree for efficient neighbor search
            kdtree = KDTree(self.coords)
            
            for i, center in enumerate(self.coords):
                neighbors_indices = kdtree.query_ball_point(center, cutoff)
                neighbors_indices = [idx for idx in neighbors_indices if idx != i]
                
                if len(neighbors_indices) < 2:
                    continue
                
                # Calculate vectors from central atom to neighbors
                neighbor_vectors = self.coords[neighbors_indices] - center
                
                # NORMALIZE WITH HIGH PRECISION
                norms = np.sqrt(np.sum(neighbor_vectors**2, axis=1))
                mask = norms > 1e-12
                neighbor_vectors = neighbor_vectors[mask]
                norms = norms[mask]
                
                # Normalize each vector separately with high precision
                normalized_vectors = []
                for vec, norm in zip(neighbor_vectors, norms):
                    if abs(norm) < 1e-12:
                        continue
                    normalized = vec.astype(np.float64) / float(norm)
                    normalized_vectors.append(normalized)
                
                if len(normalized_vectors) < 2:
                    continue
                
                # Calculate angles between all neighbor pairs
                for j in range(len(normalized_vectors)):
                    for k in range(j + 1, len(normalized_vectors)):
                        # Dot product with high precision
                        v1 = normalized_vectors[j].astype(np.float64)
                        v2 = normalized_vectors[k].astype(np.float64)
                        
                        dot_product = float(np.dot(v1, v2))
                        
                        # Handle floating-point errors near ±1
                        if dot_product > 1.0:
                            if dot_product - 1.0 < 1e-12:
                                dot_product = 1.0
                        elif dot_product < -1.0:
                            if dot_product + 1.0 < 1e-12:
                                dot_product = -1.0
                        
                        dot_product = max(-1.0, min(1.0, dot_product))
                        
                        # Calculate angle with math.acos for better precision
                        angle_rad = math.acos(dot_product)
                        angle_deg = math.degrees(angle_rad)
                        
                        angles.append(angle_deg)
            
            if not angles:
                self.status_var.set("No angles found for given radius")
                return
                
            angles = np.array(angles, dtype=np.float64)
            self.angles_history = angles
            
            # Create histogram with clean design for publication
            self.ax.clear()
            
            # Apply font settings before plotting
            self.apply_font_settings()
            
            # Use 0.01 degree bins for high resolution
            bin_edges = np.arange(0, 180.005, 0.005)  # 0.01° шаг
            hist, _ = np.histogram(angles, bins=bin_edges)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # For display, use 0.5 degree bins for better visibility
            display_bin_edges = np.arange(0, 180.25, 0.5)
            display_hist, _ = np.histogram(angles, bins=display_bin_edges)
            display_bin_centers = (display_bin_edges[:-1] + display_bin_edges[1:]) / 2
            
            # Plot histogram with bars centered at 0.5 degree intervals
            bars = self.ax.bar(display_bin_centers, display_hist, width=0.5*0.8, 
                              edgecolor='black', alpha=0.7, 
                              color=self.hist_color.get(), align='center')
            
            # Find peaks in FINE histogram for accurate angle detection
            height_threshold = np.max(hist) * 0.02  # 2% от максимальной высоты
            distance_min = 5  # Минимум 0.05° между пиками (0.01 * 5)
            
            peaks, properties = find_peaks(hist, 
                                          height=height_threshold,
                                          distance=distance_min,
                                          prominence=height_threshold/2)
            
            # Store peak information
            peak_info = []
            for peak_idx in peaks:
                peak_angle = bin_centers[peak_idx]
                peak_height = hist[peak_idx]
                
                # We find the corresponding height in the displayed histogram
                display_bin_idx = np.argmin(np.abs(display_bin_centers - peak_angle))
                if 0 <= display_bin_idx < len(display_hist):
                    display_height = display_hist[display_bin_idx]
                else:
                    display_height = 0
                
                peak_info.append({
                    'angle': peak_angle,
                    'height': peak_height,
                    'display_height': display_height,
                    'display_bin_idx': display_bin_idx
                })
            
            # We MUST add a check for 180° if there are angles close to 180°
            angles_near_180 = angles[(angles > 179.0) & (angles <= 180.0)]
            if len(angles_near_180) > 0:
                avg_180 = np.mean(angles_near_180)
                count_180 = len(angles_near_180)
                
                already_has_180 = False
                for peak in peak_info:
                    if abs(peak['angle'] - 180.0) < 0.5:
                        already_has_180 = True
                        break
                
                if not already_has_180 and count_180 > 0:
                    display_bin_idx = np.argmin(np.abs(display_bin_centers - 180.0))
                    if 0 <= display_bin_idx < len(display_hist):
                        display_height = display_hist[display_bin_idx]
                    else:
                        display_height = count_180
                    
                    peak_info.append({
                        'angle': avg_180,
                        'height': count_180,
                        'display_height': display_height,
                        'display_bin_idx': display_bin_idx
                    })
            
            # Sort peaks by height (descending)
            peak_info.sort(key=lambda x: x['height'], reverse=True)
            
            # Create vertical labels ABOVE peaks
            max_display_height = np.max(display_hist)
            if max_display_height == 0:
                max_display_height = 1
            
            labeled_positions = []
            
            # We sign ALL found peaks (except 0°, but including 180°)
            for peak in peak_info:
                peak_angle = peak['angle']
                
                # Ignore peaks close to 0° (less than 0.5°)
                if peak_angle < 0.5:
                    continue
                
                display_height = peak['display_height']
                
                # Draw vertical line at peak with selected color
                self.ax.axvline(x=peak_angle, color=self.line_color.get(), 
                               linestyle='--', alpha=0.7, linewidth=1.5)
                
                # Calculate vertical position for label - ABOVE the bar
                label_x = peak_angle
                
                # For peaks at the edges (especially 180°), move the mark inward
                if peak_angle > 179.0:
                    label_x = 179.5  # Move it a little to the left for 180°
                elif peak_angle < 1.0:
                    label_x = 1.0    # Move it a little to the right for 0°
                
                label_y = display_height + max_display_height * 0.05
                
                # Check for overlap with existing labels
                overlap_found = False
                for pos_x, pos_y in labeled_positions:
                    if abs(pos_x - label_x) < 2:
                        if abs(pos_y - label_y) < max_display_height * 0.08:
                            label_y = max(pos_y, label_y) + max_display_height * 0.12
                            overlap_found = True
                
                # Ensure label stays within plot bounds
                y_max_plot = max_display_height * 1.8
                if label_y > y_max_plot:
                    if not overlap_found:
                        label_y = max_display_height * 1.6
                    else:
                        continue
                
                # Format angle with 2 decimal places
                angle_text = f"{peak_angle:.2f}°"
                
                # Add VERTICAL text label with font settings
                self.ax.text(label_x, label_y, angle_text,
                           rotation=90,
                           ha='center', va='bottom',
                           fontsize=9, fontweight='bold', color='darkred',
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='yellow', alpha=0.8))
                
                labeled_positions.append((label_x, label_y))
            
            # Clean plot settings for publication - NO EXTRA TEXT
            self.ax.set_xlabel('Angle (degrees)', fontsize=self.axis_label_fontsize.get())
            self.ax.set_ylabel('Count', fontsize=self.axis_label_fontsize.get())
            
            # Title without extra information
            self.ax.set_title('Angle Distribution', 
                             fontsize=self.title_fontsize.get(), 
                             fontweight='bold')
            
            # Set axis limits
            self.ax.set_xlim(-5, 185)
            
            # Set y-axis limits with room for labels
            y_max_limit = max(max_display_height * 1.9, 10)
            self.ax.set_ylim(0, y_max_limit)
            
            # Add grid (only horizontal for cleaner look)
            self.ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            
            # Apply tick font size
            self.ax.tick_params(axis='both', which='major', 
                               labelsize=self.axis_tick_fontsize.get())
            
            # Add minimal legend (only if we have peaks)
            if len(peak_info) > 0:
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color=self.line_color.get(), 
                          linestyle='--', linewidth=1.5, label='Peak centers'),
                    Line2D([0], [0], marker='s', color=self.hist_color.get(), 
                          linestyle='None', markersize=10, 
                          label='Angle distribution')
                ]
                self.ax.legend(handles=legend_elements, loc='upper right',
                              fontsize=9)
            
            # Update canvas
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Print found peaks to console
            print("\n" + "="*60)
            print("ALL PEAKS FOUND in the structure (excluding near 0°):")
            print("="*60)
            valid_peaks_sorted = sorted([p for p in peak_info if p['angle'] >= 0.5], 
                                       key=lambda x: x['angle'])
            for i, peak in enumerate(valid_peaks_sorted):
                print(f"Peak {i+1:2d}: {peak['angle']:7.2f}° | "
                      f"Count: {int(peak['display_height']):5d} angles")
            
            self.status_var.set(f"Found {len(angles)} angles | {len(valid_peaks_sorted)} peaks")
            
        except Exception as e:
            self.status_var.set(f"Calculation error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def save_high_quality_plot(self):
        """Save plot as high-quality image for publication"""
        if len(self.angles_history) == 0:
            messagebox.showwarning("Warning", "No plot to save. Calculate angles first.")
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
                              f"Font: {self.font_family.get()}, "
                              f"Title: {self.title_fontsize.get()}pt, "
                              f"Labels: {self.axis_label_fontsize.get()}pt, "
                              f"Ticks: {self.axis_tick_fontsize.get()}pt")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plot: {str(e)}")
    
    def export_results(self):
        """Export calculated angles to file with high precision"""
        if len(self.angles_history) == 0:
            self.status_var.set("No angles to export. Calculate angles first.")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="angle_distribution.txt"
        )
        
        if filename:
            try:
                radius = self.cutoff_radius.get()
                
                with open(filename, 'w') as f:
                    # Write header
                    f.write(f"# Angle distribution analysis\n")
                    f.write(f"# Cutoff radius: {radius} Å\n")
                    f.write(f"# Total angles: {len(self.angles_history)}\n")
                    f.write(f"# Mean angle: {np.mean(self.angles_history):.6f}°\n")
                    f.write(f"# Median angle: {np.median(self.angles_history):.6f}°\n")
                    f.write(f"# Std deviation: {np.std(self.angles_history):.6f}°\n")
                    f.write(f"# Min angle: {np.min(self.angles_history):.6f}°\n")
                    f.write(f"# Max angle: {np.max(self.angles_history):.6f}°\n")
                    f.write(f"#\n")
                    
                    # Write histogram data (0.01 degree bins)
                    f.write(f"# High precision histogram (0.01° bins):\n")
                    f.write(f"# Bin Center (degrees), Count\n")
                    
                    bin_edges = np.arange(0, 180.005, 0.01)
                    hist, _ = np.histogram(self.angles_history, bins=bin_edges)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    for center, count in zip(bin_centers, hist):
                        if count > 0:
                            f.write(f"{center:.3f}, {count}\n")
                    
                    f.write(f"#\n")
                    f.write(f"# Display histogram (0.5° bins):\n")
                    f.write(f"# Bin Center (degrees), Count\n")
                    
                    display_bin_edges = np.arange(0, 180.25, 0.5)
                    display_hist, _ = np.histogram(self.angles_history, bins=display_bin_edges)
                    display_bin_centers = (display_bin_edges[:-1] + display_bin_edges[1:]) / 2
                    
                    for center, count in zip(display_bin_centers, display_hist):
                        f.write(f"{center:.1f}, {count}\n")
                    
                    f.write(f"#\n")
                    f.write(f"# Individual angles (sorted):\n")
                    f.write(f"# Angle (degrees)\n")
                    
                    # Write individual angles with 6 decimal places
                    sorted_angles = np.sort(self.angles_history)
                    for angle in sorted_angles:
                        f.write(f"{angle:.6f}\n")
                
                self.status_var.set(f"Results exported to {filename}")
                
            except Exception as e:
                self.status_var.set(f"Export error: {str(e)}")

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = XYZAngleAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
