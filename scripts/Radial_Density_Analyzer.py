import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import csv

class RadialDensityAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Radial Density Analyzer - Clean Plot")
        self.root.geometry("1200x850")
        
        # Variables
        self.filename = None
        self.atom_positions = None
        self.center = None
        self.radii = []
        self.densities = []
        self.N_values = []  # Store N(r) values (cumulative)
        self.atoms_on_sphere = []  # Store atoms on each specific sphere
        self.sphere_atom_indices = []  # Store indices of atoms for each sphere
        self.center_method = tk.StringVar(value="geometric")  # "geometric" or "central_atom"
        
        # Font size variables
        self.axis_label_fontsize = tk.IntVar(value=14)  # Размер шрифта подписей осей
        self.axis_tick_fontsize = tk.IntVar(value=12)   # Размер шрифта чисел на осях
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Top panel for buttons
        top_frame = tk.Frame(self.root, bg='#f0f0f0')
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(top_frame, text="Open XYZ File", command=self.open_file,
                 bg='#4CAF50', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        
        self.file_label = tk.Label(top_frame, text="No file selected", 
                                  bg='white', relief=tk.SUNKEN, width=40, anchor='w')
        self.file_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        tk.Button(top_frame, text="Analyze", command=self.analyze,
                 bg='#2196F3', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        
        # Parameter frame
        param_frame = tk.Frame(self.root, bg='#f0f0f0')
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Left side parameters
        left_param_frame = tk.Frame(param_frame, bg='#f0f0f0')
        left_param_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(left_param_frame, text="Tolerance (Å):", bg='#f0f0f0').pack(anchor=tk.W)
        self.tolerance_var = tk.StringVar(value="0.001")
        tk.Entry(left_param_frame, textvariable=self.tolerance_var, width=10).pack(anchor=tk.W)
        
        tk.Label(left_param_frame, text="Min radius (Å):", bg='#f0f0f0').pack(anchor=tk.W, pady=(10,0))
        self.min_radius_var = tk.StringVar(value="0.0")
        tk.Entry(left_param_frame, textvariable=self.min_radius_var, width=10).pack(anchor=tk.W)
        
        # Middle parameters
        middle_param_frame = tk.Frame(param_frame, bg='#f0f0f0')
        middle_param_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(middle_param_frame, text="Center calculation:", bg='#f0f0f0').pack(anchor=tk.W)
        
        # Center method radio buttons
        tk.Radiobutton(middle_param_frame, text="Geometric center", 
                      variable=self.center_method, value="geometric", 
                      bg='#f0f0f0').pack(anchor=tk.W)
        tk.Radiobutton(middle_param_frame, text="Central atom (index 0)", 
                      variable=self.center_method, value="central_atom",
                      bg='#f0f0f0').pack(anchor=tk.W)
        
        # Right side parameters - Font sizes
        right_param_frame = tk.Frame(param_frame, bg='#f0f0f0')
        right_param_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(right_param_frame, text="Font sizes:", bg='#f0f0f0', 
                font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0,5))
        
        # Axis label font size
        tk.Label(right_param_frame, text="Axis labels:", bg='#f0f0f0').pack(anchor=tk.W)
        axis_label_frame = tk.Frame(right_param_frame, bg='#f0f0f0')
        axis_label_frame.pack(anchor=tk.W)
        tk.Entry(axis_label_frame, textvariable=self.axis_label_fontsize, 
                width=4).pack(side=tk.LEFT)
        tk.Label(axis_label_frame, text="pt", bg='#f0f0f0', 
                font=('Arial', 8)).pack(side=tk.LEFT, padx=(5,0))
        
        # Axis tick font size
        tk.Label(right_param_frame, text="Axis ticks:", bg='#f0f0f0').pack(anchor=tk.W, pady=(5,0))
        axis_tick_frame = tk.Frame(right_param_frame, bg='#f0f0f0')
        axis_tick_frame.pack(anchor=tk.W)
        tk.Entry(axis_tick_frame, textvariable=self.axis_tick_fontsize, 
                width=4).pack(side=tk.LEFT)
        tk.Label(axis_tick_frame, text="pt", bg='#f0f0f0', 
                font=('Arial', 8)).pack(side=tk.LEFT, padx=(5,0))
        
        # DPI for saving
        tk.Label(right_param_frame, text="DPI for saving:", bg='#f0f0f0').pack(anchor=tk.W, pady=(10,0))
        self.dpi_var = tk.StringVar(value="600")
        tk.Entry(right_param_frame, textvariable=self.dpi_var, width=10).pack(anchor=tk.W)
        
        # Button frame
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(button_frame, text="Save Results", command=self.save_results,
                 bg='#FF9800', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Save Sphere Coords", command=self.save_sphere_coordinates,
                 bg='#9C27B0', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Save Plot (PNG)", command=self.save_plot_high_quality,
                 bg='#E91E63', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        
        # Plot area
        plot_frame = tk.Frame(self.root)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Info text with scrollbar
        info_frame = tk.Frame(self.root)
        info_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        self.info_text = tk.Text(info_frame, height=12, width=100, font=('Courier', 9))
        scrollbar = tk.Scrollbar(info_frame, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def open_file(self):
        filename = filedialog.askopenfilename(
            title="Select XYZ file",
            filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")]
        )
        
        if filename:
            self.filename = filename
            self.file_label.config(text=os.path.basename(filename))
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, f"File: {os.path.basename(filename)}\n")
            self.info_text.insert(tk.END, "Ready for analysis.\n")
            
    def read_xyz_file(self, filename):
        """Read XYZ file format"""
        atoms = []
        symbols = []
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # Skip first two lines
            for line in lines[2:]:
                parts = line.strip().split()
                if len(parts) >= 4:
                    symbols.append(parts[0])
                    x, y, z = map(float, parts[1:4])
                    atoms.append([x, y, z])
            
            return np.array(atoms), symbols
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file:\n{str(e)}")
            return None, None
    
    def analyze(self):
        if not self.filename:
            messagebox.showwarning("Warning", "Please select a file first")
            return
        
        try:
            # Read file
            self.atom_positions, self.atom_symbols = self.read_xyz_file(self.filename)
            if self.atom_positions is None:
                return
            
            # Find center of cluster
            center_method = self.center_method.get()
            if center_method == "geometric":
                self.center = np.mean(self.atom_positions, axis=0)
                center_source = "Geometric center (average)"
            else:  # central_atom
                self.center = self.atom_positions[0]  # First atom in file
                center_source = f"Central atom (index 0): {self.atom_symbols[0]} at ({self.center[0]:.4f}, {self.center[1]:.4f}, {self.center[2]:.4f})"
            
            # Calculate distances from center to all atoms
            distances = np.linalg.norm(self.atom_positions - self.center, axis=1)
            
            # Find unique distances (coordination spheres)
            tolerance = float(self.tolerance_var.get())
            min_radius = float(self.min_radius_var.get())
            
            # Sort distances with indices
            sorted_indices = np.argsort(distances)
            sorted_distances = distances[sorted_indices]
            
            # Filter out very small distances
            mask = sorted_distances >= min_radius
            sorted_distances = sorted_distances[mask]
            sorted_indices = sorted_indices[mask]
            
            if len(sorted_distances) == 0:
                messagebox.showwarning("Warning", f"No atoms found with radius >= {min_radius} Å")
                return
            
            # Group distances within tolerance and store atom indices for each group
            unique_radii = []
            atoms_in_sphere = []
            sphere_indices_list = []
            
            current_group_indices = [sorted_indices[0]]
            current_group_distances = [sorted_distances[0]]
            
            for i in range(1, len(sorted_distances)):
                if sorted_distances[i] - current_group_distances[-1] <= tolerance:
                    current_group_indices.append(sorted_indices[i])
                    current_group_distances.append(sorted_distances[i])
                else:
                    # Calculate average for this group and count atoms
                    avg_radius = np.mean(current_group_distances)
                    unique_radii.append(avg_radius)
                    atoms_in_sphere.append(len(current_group_indices))
                    sphere_indices_list.append(current_group_indices.copy())
                    
                    current_group_indices = [sorted_indices[i]]
                    current_group_distances = [sorted_distances[i]]
            
            # Don't forget the last group
            if current_group_indices:
                avg_radius = np.mean(current_group_distances)
                unique_radii.append(avg_radius)
                atoms_in_sphere.append(len(current_group_indices))
                sphere_indices_list.append(current_group_indices.copy())
            
            # Sort unique radii and corresponding data
            combined = list(zip(unique_radii, atoms_in_sphere, sphere_indices_list))
            combined.sort(key=lambda x: x[0])
            unique_radii, atoms_in_sphere, sphere_indices_list = zip(*combined)
            
            # Convert to numpy arrays
            unique_radii = np.array(unique_radii)
            atoms_in_sphere = np.array(atoms_in_sphere)
            
            # Store sphere atom indices
            self.sphere_atom_indices = list(sphere_indices_list)
            
            # For each unique radius, calculate N(r)/V(r)
            self.radii = []
            self.densities = []
            self.N_values = []  # Cumulative N(r)
            self.atoms_on_sphere = []  # Atoms on this specific sphere
            
            cumulative_atoms = 0
            for i, (r, atoms_on_this_sphere) in enumerate(zip(unique_radii, atoms_in_sphere)):
                # Update cumulative count (все атомы внутри сферы радиуса r + на ее поверхности)
                cumulative_atoms += atoms_on_this_sphere
                
                # Volume of sphere with radius r
                V_r = (4.0/3.0) * np.pi * (r**3)
                
                # Avoid division by zero for very small radii
                if V_r < 1e-12:
                    continue
                
                # Density = N(r)/V(r) где N(r) - все атомы внутри сферы радиуса r (включая поверхность)
                density = cumulative_atoms / V_r
                
                # Only add reasonable densities (avoid extreme values)
                if density < 1e30:  # Filter out extreme densities
                    self.radii.append(r)
                    self.N_values.append(cumulative_atoms)
                    self.densities.append(density)
                    self.atoms_on_sphere.append(atoms_on_this_sphere)
            
            if len(self.radii) == 0:
                messagebox.showwarning("Warning", "No valid coordination spheres found after filtering")
                return
            
            # Convert to numpy arrays for easier handling
            self.radii = np.array(self.radii)
            self.densities = np.array(self.densities)
            self.N_values = np.array(self.N_values)
            self.atoms_on_sphere = np.array(self.atoms_on_sphere)
            
            # Plot results - ЧИСТЫЙ ГРАФИК
            self.ax.clear()
            
            # Получаем размеры шрифтов из переменных
            axis_label_size = self.axis_label_fontsize.get()
            axis_tick_size = self.axis_tick_fontsize.get()
            
            # Просто точки, без лишней информации
            self.ax.scatter(self.radii, self.densities, color='blue', s=30, alpha=0.7)
            
            # Подписи осей с указанным размером шрифта
            self.ax.set_xlabel('Radius r (Å)', fontsize=axis_label_size)
            self.ax.set_ylabel('N(r) / V(r) (atoms/Å³)', fontsize=axis_label_size)
            
            # Настройка размера шрифта для чисел на осях
            self.ax.tick_params(axis='both', which='major', labelsize=axis_tick_size)
            
            # Сетка
            self.ax.grid(True, alpha=0.3)
            
            # Автоматическое масштабирование
            if len(self.densities) > 0:
                valid_densities = self.densities[self.densities > 0]
                if len(valid_densities) > 0:
                    min_density = np.min(valid_densities) * 0.9
                    max_density = np.max(valid_densities) * 1.1
                    self.ax.set_ylim(min_density, max_density)
            
            self.canvas.draw()
            
            # Update info - ВСЯ ИНФОРМАЦИЯ ЗДЕСЬ, В ТЕКСТОВОМ ПОЛЕ
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, f"File: {os.path.basename(self.filename)}\n")
            self.info_text.insert(tk.END, f"Total atoms: {len(self.atom_positions)}\n")
            self.info_text.insert(tk.END, f"{center_source}\n")
            self.info_text.insert(tk.END, f"Center coordinates: ({self.center[0]:.4f}, {self.center[1]:.4f}, {self.center[2]:.4f})\n")
            self.info_text.insert(tk.END, f"Number of coordination spheres: {len(self.radii)}\n")
            self.info_text.insert(tk.END, f"Tolerance: {tolerance} Å\n")
            self.info_text.insert(tk.END, f"Min radius: {min_radius} Å\n")
            self.info_text.insert(tk.END, f"Plot DPI: {self.dpi_var.get()} (for saving)\n")
            self.info_text.insert(tk.END, f"Font sizes: labels={axis_label_size}pt, ticks={axis_tick_size}pt\n\n")
            
            # Show first 10 spheres with atoms on sphere
            self.info_text.insert(tk.END, "First 10 coordination spheres:\n")
            self.info_text.insert(tk.END, "Sphere  Radius (Å)  Atoms_on  N(r)    N(r)/V(r) (atoms/Å³)  Volume (Å³)\n")
            self.info_text.insert(tk.END, "-" * 85 + "\n")
            
            for i in range(min(10, len(self.radii))):
                volume = (4.0/3.0) * np.pi * self.radii[i]**3
                self.info_text.insert(tk.END, 
                    f"{i+1:6d} {self.radii[i]:10.4f} {self.atoms_on_sphere[i]:8d} "
                    f"{self.N_values[i]:8d} {self.densities[i]:15.6f}  {volume:15.6f}\n")
            
            if len(self.radii) > 10:
                self.info_text.insert(tk.END, f"... and {len(self.radii)-10} more spheres\n")
            
            # Добавим расчет для проверки первой координационной сферы
            if len(self.radii) > 0:
                # Расчет для сравнения с ручным расчетом
                theoretical_volume = (4.0/3.0) * np.pi * (self.radii[0]**3)
                theoretical_density = self.N_values[0] / theoretical_volume
                
                self.info_text.insert(tk.END, f"\nCHECK - First sphere calculation:\n")
                self.info_text.insert(tk.END, f"  Sphere radius: {self.radii[0]:.4f} Å\n")
                self.info_text.insert(tk.END, f"  Atoms inside (N): {self.N_values[0]}\n")
                self.info_text.insert(tk.END, f"  Volume: {theoretical_volume:.6f} Å³\n")
                self.info_text.insert(tk.END, f"  Density (N/V): {theoretical_density:.6f} atoms/Å³\n")
                self.info_text.insert(tk.END, f"  Program density: {self.densities[0]:.6f} atoms/Å³\n")
                
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Analysis failed:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def save_plot_high_quality(self):
        """Save the current plot as high-quality PNG for publication"""
        if not hasattr(self, 'radii') or len(self.radii) == 0:
            messagebox.showwarning("Warning", "No plot to save. Please analyze data first.")
            return
        
        # Ask for save location
        save_path = filedialog.asksaveasfilename(
            title="Save High-Quality Plot",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("TIFF files", "*.tiff;*.tif"),
                ("PDF files", "*.pdf"),
                ("EPS files", "*.eps"),
                ("All files", "*.*")
            ]
        )
        
        if not save_path:
            return
        
        try:
            # Get DPI setting
            try:
                dpi = int(self.dpi_var.get())
                if dpi < 100 or dpi > 2400:
                    messagebox.showwarning("Warning", "DPI should be between 100 and 2400. Using 600.")
                    dpi = 600
            except:
                dpi = 600
            
            # Get current font sizes
            axis_label_size = self.axis_label_fontsize.get()
            axis_tick_size = self.axis_tick_fontsize.get()
            
            # Temporarily update font sizes for saving
            self.ax.set_xlabel(self.ax.get_xlabel(), fontsize=axis_label_size)
            self.ax.set_ylabel(self.ax.get_ylabel(), fontsize=axis_label_size)
            self.ax.tick_params(axis='both', which='major', labelsize=axis_tick_size)
            
            # Update figure
            self.fig.canvas.draw()
            
            # Get file extension
            ext = os.path.splitext(save_path)[1].lower()
            
            # Save with different parameters based on format
            if ext in ['.png', '.tiff', '.tif']:
                # For raster formats, use high DPI
                self.fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                                facecolor='white', edgecolor='none',
                                transparent=False, pad_inches=0.1)
            elif ext in ['.pdf', '.eps']:
                # For vector formats
                self.fig.savefig(save_path, format=ext[1:], bbox_inches='tight',
                                facecolor='white', edgecolor='none')
            else:
                # Default to PNG
                save_path = save_path + '.png'
                self.fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                                facecolor='white', edgecolor='none')
            
            messagebox.showinfo("Success", 
                              f"High-quality plot saved to:\n{save_path}\n"
                              f"Resolution: {dpi} DPI\n"
                              f"Font sizes: labels={axis_label_size}pt, ticks={axis_tick_size}pt\n"
                              f"Size: {self.fig.get_size_inches()[0]:.1f}×{self.fig.get_size_inches()[1]:.1f} inches")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save plot:\n{str(e)}")
    
    def save_results(self):
        if not self.filename or len(self.radii) == 0:
            messagebox.showwarning("Warning", "Please analyze the file first")
            return
        
        # Ask for save location
        save_path = filedialog.asksaveasfilename(
            title="Save results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not save_path:
            return
        
        try:
            if save_path.endswith('.csv'):
                # Save as CSV
                with open(save_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Sphere_Index', 'Radius_A', 'Atoms_on_sphere', 
                                   'Cumulative_N(r)', 'Volume_A3', 'N(r)/V(r)_atoms_A3'])
                    
                    for i, (r, atoms_on, N_r, density) in enumerate(zip(
                        self.radii, self.atoms_on_sphere, self.N_values, self.densities)):
                        volume = (4.0/3.0) * np.pi * r**3
                        writer.writerow([i+1, r, atoms_on, N_r, volume, density])
            else:
                # Save as text
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write("Radial Density Analysis Results\n")
                    f.write("="*100 + "\n")
                    f.write(f"File: {self.filename}\n")
                    f.write(f"Total atoms: {len(self.atom_positions)}\n")
                    f.write(f"Center method: {self.center_method.get()}\n")
                    f.write(f"Cluster center: ({self.center[0]:.6f}, {self.center[1]:.6f}, {self.center[2]:.6f})\n")
                    f.write(f"Tolerance: {self.tolerance_var.get()} Å\n")
                    f.write(f"Min radius: {self.min_radius_var.get()} Å\n")
                    f.write(f"Number of coordination spheres: {len(self.radii)}\n\n")
                    
                    f.write(f"{'Sphere':>6} {'Radius (Å)':>12} {'Atoms_on':>8} "
                           f"{'N(r)':>8} {'Volume (Å³)':>15} {'N(r)/V(r) (atoms/Å³)':>25}\n")
                    f.write("-"*100 + "\n")
                    
                    for i, (r, atoms_on, N_r, density) in enumerate(zip(
                        self.radii, self.atoms_on_sphere, self.N_values, self.densities)):
                        volume = (4.0/3.0) * np.pi * r**3
                        f.write(f"{i+1:6d} {r:12.6f} {atoms_on:8d} {N_r:8d} "
                               f"{volume:15.6e} {density:25.6e}\n")
            
            # Save plot
            plot_path = os.path.splitext(save_path)[0] + "_plot.png"
            self.fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            messagebox.showinfo("Success", 
                               f"Results saved to:\n{save_path}\n"
                               f"Plot saved to:\n{plot_path}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save results:\n{str(e)}")
    
    def save_sphere_coordinates(self):
        """Save coordinates of atoms for each coordination sphere in separate XYZ files"""
        if not self.filename or len(self.radii) == 0:
            messagebox.showwarning("Warning", "Please analyze the file first")
            return
        
        if not hasattr(self, 'sphere_atom_indices') or len(self.sphere_atom_indices) == 0:
            messagebox.showwarning("Warning", "No sphere atom data available. Please analyze first.")
            return
        
        # Ask for folder to save sphere coordinates
        folder_path = filedialog.askdirectory(title="Select folder to save sphere coordinates")
        if not folder_path:
            return
        
        try:
            # Create subfolder based on input filename
            base_name = os.path.splitext(os.path.basename(self.filename))[0]
            spheres_folder = os.path.join(folder_path, f"{base_name}_spheres")
            os.makedirs(spheres_folder, exist_ok=True)
            
            saved_files = []
            
            # Save coordinates for each sphere
            for i in range(len(self.radii)):
                sphere_index = i + 1
                radius = self.radii[i]
                atoms_count = self.atoms_on_sphere[i]
                
                # Get atom indices for this sphere
                atom_indices = self.sphere_atom_indices[i]
                
                # Create filename
                filename = f"sphere_{sphere_index:03d}_r{radius:.3f}_atoms{atoms_count}.xyz"
                filepath = os.path.join(spheres_folder, filename)
                
                # Write XYZ file
                with open(filepath, 'w', encoding='utf-8') as f:
                    # First line: number of atoms
                    f.write(f"{atoms_count}\n")
                    
                    # Second line: comment with sphere info
                    f.write(f"Sphere {sphere_index}: radius={radius:.6f} Å, atoms={atoms_count}, "
                           f"center=({self.center[0]:.6f}, {self.center[1]:.6f}, {self.center[2]:.6f})\n")
                    
                    # Write atom coordinates
                    for idx in atom_indices:
                        pos = self.atom_positions[idx]
                        symbol = self.atom_symbols[idx] if hasattr(self, 'atom_symbols') else 'C'
                        f.write(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
                
                saved_files.append(filename)
            
            # Also save a summary file
            summary_path = os.path.join(spheres_folder, "spheres_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("Coordination Spheres Summary\n")
                f.write("="*60 + "\n")
                f.write(f"Original file: {self.filename}\n")
                f.write(f"Total spheres: {len(self.radii)}\n")
                f.write(f"Cluster center: ({self.center[0]:.6f}, {self.center[1]:.6f}, {self.center[2]:.6f})\n\n")
                
                f.write("Sphere details:\n")
                f.write(f"{'Sphere':>6} {'Radius (Å)':>12} {'Atoms':>8} {'Filename':>30}\n")
                f.write("-"*60 + "\n")
                
                for i in range(len(self.radii)):
                    sphere_index = i + 1
                    radius = self.radii[i]
                    atoms_count = self.atoms_on_sphere[i]
                    filename = f"sphere_{sphere_index:03d}_r{radius:.3f}_atoms{atoms_count}.xyz"
                    
                    f.write(f"{sphere_index:6d} {radius:12.6f} {atoms_count:8d} {filename:>30}\n")
            
            messagebox.showinfo("Success", 
                               f"Saved {len(saved_files)} sphere coordinate files to:\n{spheres_folder}\n"
                               f"Also saved summary file: spheres_summary.txt")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save sphere coordinates:\n{str(e)}")

def main():
    root = tk.Tk()
    app = RadialDensityAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
