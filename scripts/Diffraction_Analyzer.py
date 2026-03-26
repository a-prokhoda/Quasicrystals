import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
import threading
from scipy.ndimage import maximum_filter, label
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit
from scipy.spatial import KDTree

class QuasicrystalDiffractionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quasicrystal Diffraction Analyzer 2026 (Pro)")
        self.root.geometry("1550x950")
        
        self.xyz_data = None
        self.intensity = None
        self.peaks_mask = None
        self.q_axis = None
        self.q_max = None
        
        # UI variables
        self.cutoff_label_var = tk.StringVar(value="1.0 %")
        self.delta_k_label_var = tk.StringVar(value="0.10 Å⁻¹")
        
        # Variables for analysis
        self.window_function_enabled = tk.BooleanVar(value=False)
        self.envelope_strength_var = tk.DoubleVar(value=0.0)
        
        # Debye-Waller factor (RMS displacement in Å)
        self.dw_factor_var = tk.DoubleVar(value=0.05)  # ⟨u²⟩^{1/2}
        
        self.create_main_layout()

    def create_main_layout(self):
            main = ttk.Frame(self.root)
            main.pack(fill=tk.BOTH, expand=True)
            
            # Left control panel - компактная версия
            left = ttk.LabelFrame(main, text="Physical Controls", width=320)
            left.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
            
            # Calculation parameters
            ttk.Label(left, text="Q Max (1/Å):").pack(pady=(3, 0))
            self.q_max_var = tk.DoubleVar(value=10.0)
            ttk.Entry(left, textvariable=self.q_max_var).pack(fill=tk.X, padx=5, pady=(0, 5))
            
            ttk.Label(left, text="Resolution (N x N):").pack(pady=(0, 0))
            self.nq_var = tk.IntVar(value=400)
            ttk.Entry(left, textvariable=self.nq_var).pack(fill=tk.X, padx=5, pady=(0, 8))
            
            # Peak detection - компактно
            ttk.Label(left, text="Peak Detection:", font=('Arial', 9, 'bold')).pack(pady=(2, 2))
            
            ttk.Label(left, text="Cutoff Intensity (%):", font=('Arial', 8)).pack(pady=(0, 0))
            self.cutoff_var = tk.DoubleVar(value=1.0)
            c_scale = ttk.Scale(left, from_=0.1, to=20, variable=self.cutoff_var,
                               command=lambda e: self.cutoff_label_var.set(f"{self.cutoff_var.get():.1f} %"))
            c_scale.pack(fill=tk.X, padx=5, pady=(0, 0))
            ttk.Label(left, textvariable=self.cutoff_label_var, foreground="darkgreen", font=('Arial', 8)).pack()
            
            ttk.Label(left, text="Δk Resolution (1/Å):", font=('Arial', 8)).pack(pady=(3, 0))
            self.delta_k_var = tk.DoubleVar(value=0.1)
            dk_scale = ttk.Scale(left, from_=0.01, to=1.0, variable=self.delta_k_var,
                                command=lambda e: self.delta_k_label_var.set(f"{self.delta_k_var.get():.2f} Å⁻¹"))
            dk_scale.pack(fill=tk.X, padx=5, pady=(0, 0))
            ttk.Label(left, textvariable=self.delta_k_label_var, foreground="darkgreen", font=('Arial', 8)).pack(pady=(0, 5))
            
            # FINITE-SIZE ENVELOPE - компактно
            envelope_frame = ttk.LabelFrame(left, text="Finite-Size Envelope", padding=3)
            envelope_frame.pack(fill=tk.X, padx=3, pady=2)
            
            ttk.Checkbutton(envelope_frame, text="Enable Window Function", 
                           variable=self.window_function_enabled).pack(anchor=tk.W, pady=(0, 2))
            
            ttk.Label(envelope_frame, text="Strength (0-1):", font=('Arial', 8)).pack(anchor=tk.W, pady=(0, 0))
            ttk.Scale(envelope_frame, from_=0, to=1, variable=self.envelope_strength_var,
                     orient='horizontal').pack(fill=tk.X, padx=2, pady=(0, 2))
            
            ttk.Label(envelope_frame, text="0: infinite crystal, 1: strong finite-size",
                     font=('Arial', 7), foreground="gray").pack(anchor=tk.W, pady=(0, 0))
            
            # Debye-Waller factor - компактно
            dw_frame = ttk.LabelFrame(left, text="Debye-Waller Factor", padding=3)
            dw_frame.pack(fill=tk.X, padx=3, pady=2)
            
            ttk.Label(dw_frame, text="RMS displacement (Å):", font=('Arial', 8)).pack(anchor=tk.W, pady=(0, 0))
            dw_scale = ttk.Scale(dw_frame, from_=0.0, to=0.2, variable=self.dw_factor_var,
                               orient='horizontal')
            dw_scale.pack(fill=tk.X, padx=2, pady=(0, 2))
            
            self.dw_label_var = tk.StringVar(value=f"{self.dw_factor_var.get():.3f} Å")
            ttk.Label(dw_frame, textvariable=self.dw_label_var, foreground="darkblue", 
                     font=('Arial', 8)).pack(anchor=tk.W, pady=(0, 0))
            
            ttk.Label(dw_frame, text="Typical: 0.05 Å", font=('Arial', 7), foreground="gray").pack(anchor=tk.W, pady=(0, 0))
            
            def update_dw_label(e):
                self.dw_label_var.set(f"{self.dw_factor_var.get():.3f} Å")
            dw_scale.configure(command=update_dw_label)
            
            # Progress bar - компактно
            self.progress_var = tk.DoubleVar(value=0.0)
            self.progress_bar = ttk.Progressbar(left, variable=self.progress_var, maximum=100, length=100)
            self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
            
            # Buttons - компактная группировка
            btn_frame = ttk.Frame(left)
            btn_frame.pack(fill=tk.X, padx=5, pady=(5, 5))
            
            ttk.Button(btn_frame, text="1. Load XYZ", command=self.load_xyz, width=12).pack(side=tk.LEFT, padx=(0, 2))
            self.calc_btn = ttk.Button(btn_frame, text="2. Calculate", command=self.start_calc_thread, width=12)
            self.calc_btn.pack(side=tk.LEFT, padx=(2, 0))
            
            ttk.Button(left, text="Export Plots (PNG)", command=self.export_all_plots).pack(fill=tk.X, padx=5, pady=(0, 5))
            
            self.status_var = tk.StringVar(value="Ready: 2D slice through 3D reciprocal space (Qz=0)")
            ttk.Label(left, textvariable=self.status_var, font=('Arial', 8), wraplength=300,
                     justify=tk.LEFT).pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
            
            # Notebook for tabs
            self.notebook = ttk.Notebook(main)
            self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Tabs
            self.tab1 = ttk.Frame(self.notebook)
            self.tab2 = ttk.Frame(self.notebook)
            self.notebook.add(self.tab1, text='2D Diffraction Patterns')
            self.notebook.add(self.tab2, text='Peak Statistics')
            
            self.create_diffraction_tab()
            self.create_analysis_tab()

        
    def create_diffraction_tab(self):
        self.fig_diff, (self.ax_theory, self.ax_exp) = plt.subplots(
            1, 2,
            figsize=(12, 5)
        )
        self.fig_diff.patch.set_facecolor('#f0f0f0')
        
        self.canvas_diff = FigureCanvasTkAgg(self.fig_diff, master=self.tab1)
        self.canvas_diff.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.toolbar_diff = NavigationToolbar2Tk(self.canvas_diff, self.tab1)
        self.toolbar_diff.update()

    def create_analysis_tab(self):
        self.fig_analysis = Figure(figsize=(12, 8))
        self.fig_analysis.patch.set_facecolor('#f0f0f0')
        
        gs = self.fig_analysis.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        self.ax_envelope = self.fig_analysis.add_subplot(gs[0, 0])
        self.ax_density = self.fig_analysis.add_subplot(gs[0, 1])
        self.ax_cumulative = self.fig_analysis.add_subplot(gs[1, :])
        
        self.ax_envelope.set_title('Window Function in Real Space')
        self.ax_envelope.set_xlabel('Radius (Å)')
        self.ax_envelope.set_ylabel('Envelope Weight')
        self.ax_envelope.grid(True, alpha=0.3)
        
        self.ax_density.set_title('Peak Distribution N(Q)/Area')
        self.ax_density.set_xlabel('|Q| (Å⁻¹)')
        self.ax_density.set_ylabel('Density (peaks/Å⁻²)')
        self.ax_density.set_yscale('log')
        self.ax_density.grid(True, alpha=0.3)
        
        self.ax_cumulative.set_title('Cumulative Peak Count')
        self.ax_cumulative.set_xlabel('|Q| (Å⁻¹)')
        self.ax_cumulative.set_ylabel('Number of Peaks')
        self.ax_cumulative.grid(True, alpha=0.3)
        
        self.canvas_analysis = FigureCanvasTkAgg(self.fig_analysis, master=self.tab2)
        self.canvas_analysis.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.toolbar_analysis = NavigationToolbar2Tk(self.canvas_analysis, self.tab2)
        self.toolbar_analysis.update()

    def load_xyz(self):
        path = filedialog.askopenfilename(filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")])
        if not path: return
        
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            
            # Skip header lines
            skip_lines = 0
            for i, line in enumerate(lines):
                if line.strip() and not any(char.isalpha() for char in line.split()[0] if line.split()):
                    if len(line.split()) >= 4:
                        skip_lines = i
                        break
            
            data = np.loadtxt(path, skiprows=skip_lines, usecols=(1, 2, 3))
            self.xyz_data = data - np.mean(data, axis=0)
            
            # Estimate minimum interatomic distance (10th percentile of nearest neighbour distances)
            kdtree = KDTree(self.xyz_data)
            dists, _ = kdtree.query(self.xyz_data, k=2)
            min_dist = np.percentile(dists[:, 1], 10)  # exclude self
            # Suggested Q_max: limit to 30 Å⁻¹ and at most 4π / min_dist (twice the maximum expected peak)
            suggested_q = min(30.0, 4 * np.pi / min_dist)
            self.q_max_var.set(round(suggested_q, 1))
            
            self.status_var.set(f"Loaded: {len(self.xyz_data)} atoms, d_min≈{min_dist:.2f}Å → Q_max≈{suggested_q:.1f}Å⁻¹")
            
            # Show cluster info
            radii = np.linalg.norm(self.xyz_data, axis=1)
            L = np.max(radii)
            volume = 4/3 * np.pi * L**3
            density_3d = len(self.xyz_data) / volume if volume > 0 else 0
            
            messagebox.showinfo("Success", 
                              f"Loaded {len(self.xyz_data)} atoms\n"
                              f"3D cluster radius: {L:.1f} Å\n"
                              f"Mean 3D density: {density_3d:.3e} atoms/Å³\n"
                              f"Suggested Q_max: {suggested_q:.1f} Å⁻¹")
            
        except Exception as e:
            messagebox.showerror("Error", f"Load error: {e}")


        
    def safe_lognorm(self, data):
        positive = data[data > 0]
        if positive.size == 0:
            return LogNorm(vmin=1, vmax=2)
        vmin = np.percentile(positive, 5)
        vmax = np.max(positive)
        return LogNorm(vmin=max(vmin, vmax*1e-5), vmax=vmax)

    def start_calc_thread(self):
        if self.xyz_data is None:
            messagebox.showwarning("Warning", "Please load XYZ file first")
            return
        
        self.calc_btn.config(state=tk.DISABLED)
        self.status_var.set("Computing Fourier Transform (Qz=0 slice)...")
        self.progress_var.set(0)
        threading.Thread(target=self.calculate_diffraction, daemon=True).start()

    def update_progress(self, value):
        self.root.after(0, lambda: self.progress_var.set(value))

    def calculate_diffraction(self):
        """Kinematic diffraction calculation: 2D slice at Qz=0 through 3D reciprocal space."""
        try:
            q_max = self.q_max_var.get()
            nq = self.nq_var.get()
            self.q_axis = np.linspace(-q_max, q_max, nq)
            qx, qy = np.meshgrid(self.q_axis, self.q_axis)

            # Build 3D Q vectors with Qz = 0
            qx_flat = qx.ravel()
            qy_flat = qy.ravel()
            qz_flat = np.zeros_like(qx_flat)
            q_points_3d = np.column_stack([qx_flat, qy_flat, qz_flat])

            # Window function (finite-size envelope) – Gaussian decay
            weights = np.ones(len(self.xyz_data))
            if self.window_function_enabled.get():
                radii = np.linalg.norm(self.xyz_data, axis=1)
                max_r = np.max(radii)
                strength = self.envelope_strength_var.get()
                if strength > 0:
                    sigma = max_r / strength   # strength=1 → sigma = max_r, strength→0 → sigma→∞ (no envelope)
                    weights = np.exp(- (radii**2) / (2 * sigma**2))
                # else strength=0: weights already 1 (no envelope)

            self.update_progress(5)

            # Chunked computation
            # Adaptive chunk size: aim for ~10000 Q points or half the number of Q points
            n_qpoints = len(q_points_3d)
            chunk_size = min(10000, n_qpoints)
            # Also avoid huge memory if atoms are many: we can reduce chunk size
            # but current approach is fine for typical <50000 atoms.
            intensity = np.zeros(n_qpoints)
            for start in range(0, n_qpoints, chunk_size):
                end = min(start + chunk_size, n_qpoints)
                q_chunk = q_points_3d[start:end]
                
                # Precompute Debye-Waller factor for this Q chunk: exp(-½ |q|² σ²)
                # |q|² = sum over components (qx²+qy²+qz²) – qz=0, so it's qx²+qy²
                q2_chunk = np.sum(q_chunk**2, axis=1)  # shape (chunk_size,)
                dw_factor = self.dw_factor_var.get()
                dw_q = np.exp(-0.5 * q2_chunk * (dw_factor**2))  # shape (chunk_size,)
                
                # Phases: q_chunk @ atom_coords.T  -> (chunk_size, N_atoms)
                phases = q_chunk @ self.xyz_data.T
                
                # Combine window and DW factors: the DW factor is per Q point, same for all atoms
                # So we can multiply the atomic weights by dw_q[:, None]
                weights_dw = weights * dw_q[:, None]   # broadcast: (chunk_size, N_atoms)
                
                cos_sum = np.sum(weights_dw * np.cos(phases), axis=1)
                sin_sum = np.sum(weights_dw * np.sin(phases), axis=1)
                
                # Intensity = |sum|², normalized by (N_atoms)²
                intensity[start:end] = (cos_sum**2 + sin_sum**2) / (len(self.xyz_data)**2)
                
                progress = 5 + 45 * (end / n_qpoints)
                self.update_progress(progress)

            # Reshape intensity to 2D grid
            intensity = intensity.reshape(nq, nq)

            # Peak detection with robust local maxima
            max_val = np.max(intensity)
            threshold = (self.cutoff_var.get() / 100.0) * max_val
            
            pixel_res = (2 * q_max) / nq
            size_px = max(3, int(self.delta_k_var.get() / pixel_res))
            local_max = maximum_filter(intensity, size=size_px)
            
            # Use the more robust condition: intensity is a local maximum (>= 0.999 of max in window)
            peaks_mask = (intensity >= local_max * 0.999) & (intensity > threshold)
            
            # Remove multiple peaks on plateaus using connected components
            labeled_array, num_features = label(peaks_mask)
            if num_features > 0:
                cleaned_mask = np.zeros_like(peaks_mask)
                for i in range(1, num_features + 1):
                    component_mask = labeled_array == i
                    component_intensity = intensity[component_mask]
                    max_idx = np.argmax(component_intensity)
                    component_coords = np.argwhere(component_mask)
                    max_coord = component_coords[max_idx]
                    cleaned_mask[max_coord[0], max_coord[1]] = True
                peaks_mask = cleaned_mask
            
            self.update_progress(90)
            
            # Store results
            self.intensity = intensity
            self.peaks_mask = peaks_mask
            self.q_max = q_max
            
            # Analysis data (requires 3D q_grid; we reuse q_points_3d)
            analysis_data = self.calculate_analysis_data(
                intensity, peaks_mask,
                q_points_3d.reshape(nq, nq, 3), weights
            )
            
            self.update_progress(100)
            self.root.after(0, lambda: self.update_all_plots(intensity, peaks_mask, analysis_data))
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        finally:
            self.root.after(0, lambda: self.calc_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_var.set("Analysis Complete"))

            
    def calculate_analysis_data(self, intensity, peaks_mask, q_grid, weights):
        ys, xs = np.nonzero(peaks_mask)
        peak_coords = np.column_stack([self.q_axis[xs], self.q_axis[ys]])
        peak_intensities = intensity[peaks_mask]

        peak_radii = np.linalg.norm(peak_coords, axis=1)

        n_bins = 50
        radial_bins = np.linspace(0, self.q_max, n_bins + 1)
        bin_centers = 0.5 * (radial_bins[:-1] + radial_bins[1:])

        peak_counts, _ = np.histogram(peak_radii, bins=radial_bins)

        ring_areas = np.pi * (radial_bins[1:]**2 - radial_bins[:-1]**2)
        ring_areas[ring_areas == 0] = np.nan

        peak_density = peak_counts / ring_areas
        cumulative_peaks = np.cumsum(peak_counts)

        # ---- Window function (real space, 3D) ----
        atom_radii = np.linalg.norm(self.xyz_data, axis=1)
        atom_bins = np.linspace(0, atom_radii.max(), n_bins + 1)
        atom_counts, _ = np.histogram(atom_radii, bins=atom_bins, weights=weights)

        # ---- Power-law fit ρ(Q)=A·Q^α ----
        fit_success = False
        alpha = np.nan
        A = np.nan
        r2 = np.nan

        mask = (
            (peak_density > 0) &
            (bin_centers > 0.1) &
            np.isfinite(peak_density)
        )

        if np.count_nonzero(mask) >= 5:
            logQ = np.log(bin_centers[mask])
            logRho = np.log(peak_density[mask])

            coeffs = np.polyfit(logQ, logRho, 1)
            alpha = coeffs[0]
            logA = coeffs[1]
            A = np.exp(logA)

            # R²
            logRho_fit = alpha * logQ + logA
            ss_res = np.sum((logRho - logRho_fit)**2)
            ss_tot = np.sum((logRho - logRho.mean())**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            fit_success = True

        return {
            'peak_radii': peak_radii,
            'peak_intensities': peak_intensities,
            'bin_centers': bin_centers,
            'peak_density': peak_density,
            'cumulative_peaks': cumulative_peaks,
            'atom_bins': atom_bins,
            'atom_counts': atom_counts,
            'alpha': alpha,
            'A': A,
            'fractal_dimension': alpha + 2 if fit_success else np.nan,
            'r2': r2,
            'fit_success': fit_success,
            'total_peaks': len(peak_radii)
        }

    def update_all_plots(self, intensity, peaks_mask, analysis_data):
        self.update_diffraction_plots(intensity, peaks_mask)
        self.update_analysis_plots(analysis_data)
        self.notebook.select(self.tab2)

    def update_diffraction_plots(self, intensity, peaks_mask):
        self.ax_theory.clear()
        self.ax_exp.clear()
        
        # Устанавливаем РАВНУЮ ОБЛАСТЬ для обоих графиков
        self.ax_theory.set_aspect('equal', adjustable='box')
        self.ax_exp.set_aspect('equal', adjustable='box')
        
        # Устанавливаем одинаковые пределы
        q_range = [-self.q_max, self.q_max]
        self.ax_theory.set_xlim(q_range)
        self.ax_theory.set_ylim(q_range)
        self.ax_exp.set_xlim(q_range)
        self.ax_exp.set_ylim(q_range)
        
        # 1. Theoretical diffraction pattern
        im = self.ax_theory.imshow(
            intensity,
            extent=[-self.q_max, self.q_max, -self.q_max, self.q_max],
            origin="lower",
            cmap="inferno",
            norm=self.safe_lognorm(intensity)
        )
        self.ax_theory.set_title("Kinematic Diffraction Pattern (2D slice, Qz=0)")
        self.ax_theory.set_xlabel("$Q_x$ (Å⁻¹)")
        self.ax_theory.set_ylabel("$Q_y$ (Å⁻¹)")
        
        try:
            if hasattr(self, 'cbar_theory'):
                self.cbar_theory.remove()
        except:
            pass
        
        self.cbar_theory = self.fig_diff.colorbar(im, ax=self.ax_theory)
        self.cbar_theory.set_label("Intensity (log scale)")
        
        # 2. Detected peaks
        ys, xs = np.nonzero(peaks_mask)
        if len(ys) > 0:
            peak_intensities = intensity[peaks_mask]
            peak_sizes = 20 + 80 * (peak_intensities / np.max(peak_intensities))**0.5
            
            self.ax_exp.scatter(
                self.q_axis[xs], self.q_axis[ys],
                s=peak_sizes, c=peak_intensities, cmap='viridis',
                edgecolors='white', linewidths=0.5, alpha=0.8
            )
        
        self.ax_exp.set_xlim(-self.q_max, self.q_max)
        self.ax_exp.set_ylim(-self.q_max, self.q_max)
        self.ax_exp.set_aspect('equal')
        
        envelope_status = "ON" if self.window_function_enabled.get() else "OFF"
        dw = self.dw_factor_var.get()
        self.ax_exp.set_title(f"Detected Bragg Peaks (Envelope={envelope_status}, σ_u={dw:.3f}Å)")
        self.ax_exp.set_xlabel("$Q_x$ (Å⁻¹)")
        self.ax_exp.set_ylabel("$Q_y$ (Å⁻¹)")
        
        # Add text info
        n_peaks = len(ys)
        total_pixels = intensity.size
        if total_pixels > 0:
            peak_fraction = n_peaks / total_pixels * 100
            self.ax_exp.text(0.02, 0.98, f"Peaks: {n_peaks}\n({peak_fraction:.3f}% of area)",
                            transform=self.ax_exp.transAxes, fontsize=9,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.canvas_diff.draw()

    def update_analysis_plots(self, data):
        self.ax_envelope.clear()
        self.ax_density.clear()
        self.ax_cumulative.clear()
        
        # 1. Window function in real space
        self.ax_envelope.bar(data['atom_bins'][:-1], data['atom_counts'],
                          width=np.diff(data['atom_bins']), alpha=0.7,
                          color='skyblue', edgecolor='navy', linewidth=0.5)
        
        envelope_status = "ON" if self.window_function_enabled.get() else "OFF"
        strength = self.envelope_strength_var.get() if self.window_function_enabled.get() else 0.0
        self.ax_envelope.set_title(f'Window Function (Envelope={envelope_status}, Strength={strength:.2f})')
        self.ax_envelope.set_xlabel('3D Radius (Å)')
        self.ax_envelope.set_ylabel('Weighted Atom Count')
        self.ax_envelope.grid(True, alpha=0.3)
        
        # 2. Peak density in Q-space
        mask = data['peak_density'] > 0

        self.ax_density.scatter(
            data['bin_centers'][mask],
            data['peak_density'][mask],
            s=30, color='red', alpha=0.7, label='Data'
        )

        if data['fit_success'] and not np.isnan(data['alpha']):
            x_fit = data['bin_centers'][mask]
            y_fit = data['A'] * x_fit**data['alpha']

            self.ax_density.plot(
                x_fit, y_fit, 'b-', linewidth=2,
                label=(
                    f'ρ(Q)=A·Q^α\n'
                    f'α = {data["alpha"]:.2f}\n'
                    f'D = {data["fractal_dimension"]:.2f}\n'
                    f'R² = {data["r2"]:.3f}'
                )
            )

        self.ax_density.set_yscale('log')
        self.ax_density.set_xlabel('|Q| (Å⁻¹)')
        self.ax_density.set_ylabel('ρ(Q) (peaks / Å²)')
        self.ax_density.set_title('Peak density in reciprocal space')
        self.ax_density.grid(True, alpha=0.3)
        self.ax_density.legend()
        
        # 3. Cumulative peak count
        if len(data['cumulative_peaks']) > 0:
            self.ax_cumulative.plot(data['bin_centers'], data['cumulative_peaks'],
                                   'g-', linewidth=2, marker='o', markersize=4)
        
        # Add reference line for total atoms
        total_atoms = len(self.xyz_data)
        self.ax_cumulative.axhline(y=total_atoms, color='r', linestyle='--',
                                  alpha=0.5, label=f'Total atoms: {total_atoms}')
        
        # Add annotation at 90% of total peaks
        if len(data['cumulative_peaks']) > 0 and data['cumulative_peaks'][-1] > 0:
            max_peaks = data['cumulative_peaks'][-1]
            idx_90 = np.argmax(data['cumulative_peaks'] > 0.9 * max_peaks)
            if idx_90 > 0:
                q90 = data['bin_centers'][idx_90]
                self.ax_cumulative.axvline(x=q90, color='orange', linestyle=':',
                                          alpha=0.7, label=f'90% at Q={q90:.1f} Å⁻¹')
        
        self.ax_cumulative.set_title(f'Cumulative Bragg Peak Count (Total: {data["total_peaks"]})')
        self.ax_cumulative.set_xlabel('|Q| (Å⁻¹)')
        self.ax_cumulative.set_ylabel('N(Q) = ∫ρ(Q)dA')
        self.ax_cumulative.legend()
        self.ax_cumulative.grid(True, alpha=0.3)
        
        # Add statistics box - ИСПРАВЛЕНО: используем правильные ключи
        alpha = data['alpha']
        fractal_dim = data['fractal_dimension']
        
        # Проверяем, что значения не NaN
        alpha_str = f"{alpha:.2f}" if not np.isnan(alpha) else "N/A"
        fractal_dim_str = f"{fractal_dim:.2f}" if not np.isnan(fractal_dim) else "N/A"
        
        stats_text = (
            f"Total peaks: {data['total_peaks']}\n"
            f"Effective scaling exponent α: {alpha_str}\n"
            f"Fractal dimension D = α+2: {fractal_dim_str}\n"
            f"Window function: {'ON' if self.window_function_enabled.get() else 'OFF'}\n"
            f"σ_u (DW): {self.dw_factor_var.get():.3f} Å"
        )
        
        self.ax_cumulative.text(0.02, 0.98, stats_text,
                               transform=self.ax_cumulative.transAxes,
                               fontsize=9, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        self.canvas_analysis.draw()
        
    def export_all_plots(self):
        if self.intensity is None:
            messagebox.showwarning("Warning", "No data to export. Please calculate patterns first.")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"), ("All files", "*.*")],
            initialfile="quasicrystal_analysis.png"
        )
        
        if path:
            fig_export = Figure(figsize=(16, 12))
            gs = fig_export.add_gridspec(3, 3)
            
            # 1. Theoretical diffraction
            ax1 = fig_export.add_subplot(gs[0, 0])
            im1 = ax1.imshow(self.intensity, extent=[-self.q_max, self.q_max, -self.q_max, self.q_max],
                           origin="lower", cmap="inferno", norm=self.safe_lognorm(self.intensity))
            ax1.set_title("Kinematic Diffraction Pattern (2D slice at Qz=0)")
            ax1.set_xlabel("Q_x (Å⁻¹)")
            ax1.set_ylabel("Q_y (Å⁻¹)")
            fig_export.colorbar(im1, ax=ax1, label="Intensity (log)")
            
            # 2. Detected peaks
            ax2 = fig_export.add_subplot(gs[0, 1])
            if self.peaks_mask is not None:
                ys, xs = np.nonzero(self.peaks_mask)
                if len(ys) > 0:
                    peak_intensities = self.intensity[self.peaks_mask]
                    ax2.scatter(self.q_axis[xs], self.q_axis[ys], s=10, c=peak_intensities, cmap='viridis')
            ax2.set_xlim(-self.q_max, self.q_max)
            ax2.set_ylim(-self.q_max, self.q_max)
            ax2.set_aspect('equal')
            ax2.set_title(f"Detected Bragg Peaks")
            ax2.set_xlabel("Q_x (Å⁻¹)")
            
            # 3. Window function
            ax3 = fig_export.add_subplot(gs[0, 2])
            if hasattr(self, 'xyz_data'):
                atom_radii = np.linalg.norm(self.xyz_data, axis=1)  # 3D radii
                ax3.hist(atom_radii, bins=50, alpha=0.7, color='skyblue')
                ax3.set_title("Window Function Profile (3D radii)")
                ax3.set_xlabel("3D Radius (Å)")
                ax3.set_ylabel("Weighted Count")
            
            # 4. Peak density
            ax4 = fig_export.add_subplot(gs[1, :])
            if self.peaks_mask is not None:
                ys, xs = np.nonzero(self.peaks_mask)
                peak_radii = np.linalg.norm(np.column_stack([self.q_axis[xs], self.q_axis[ys]]), axis=1)
                radial_bins = np.linspace(0, self.q_max, 51)
                peak_counts, _ = np.histogram(peak_radii, bins=radial_bins)
                ring_areas = np.pi * (radial_bins[1:]**2 - radial_bins[:-1]**2)
                peak_density = peak_counts / ring_areas
                bin_centers = (radial_bins[:-1] + radial_bins[1:]) / 2
                
                mask = peak_density > 0
                ax4.scatter(bin_centers[mask], peak_density[mask], s=30, c='red')
                ax4.set_yscale('log')
                ax4.set_title("Peak Density ρ(Q) = dN/dA")
                ax4.set_xlabel("Q (Å⁻¹)")
                ax4.set_ylabel("ρ(Q) (peaks/Å⁻²)")
                ax4.grid(True, alpha=0.3)
            
            # 5. Cumulative peaks
            ax5 = fig_export.add_subplot(gs[2, :])
            if self.peaks_mask is not None:
                cumulative = np.cumsum(peak_counts)
                ax5.plot(bin_centers, cumulative, 'g-', linewidth=2)
                ax5.set_title("Cumulative Peak Count N(Q)")
                ax5.set_xlabel("Q (Å⁻¹)")
                ax5.set_ylabel("Total Peaks")
                ax5.grid(True, alpha=0.3)
            
            # Add text box with parameters
            params_text = (
                f"PHYSICAL PARAMETERS:\n"
                f"Q_max = {self.q_max:.1f} Å⁻¹\n"
                f"Cutoff = {self.cutoff_var.get():.1f}%\n"
                f"Δk = {self.delta_k_var.get():.2f} Å⁻¹\n"
                f"Atoms = {len(self.xyz_data) if self.xyz_data is not None else 0}\n"
                f"Window function = {self.window_function_enabled.get()}\n"
                f"Envelope strength = {self.envelope_strength_var.get():.2f}\n"
                f"RMS displacement σ_u = {self.dw_factor_var.get():.3f} Å\n"
                f"2D slice through 3D reciprocal space (Qz=0)\n"
                f"Finite cluster size effects included"
            )
            
            fig_export.text(0.02, 0.02, params_text, fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            fig_export.suptitle(f"Quasicrystal Diffraction Analysis", fontsize=16, fontweight='bold')
            fig_export.tight_layout()
            
            fig_export.savefig(path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Export Successful", f"All plots saved to:\n{path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = QuasicrystalDiffractionApp(root)
    root.mainloop()
