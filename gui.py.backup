"""
Texture Pixelator GUI - Simple interface for parameter tweaking
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path
from texture_pixelator import TexturePixelator
from batch_process import BatchProcessor
from PIL import Image, ImageTk, ImageFilter


class PixelatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Texture Pixelator")
        self.root.geometry("1100x820")
        self.root.resizable(True, True)
        
        self.pixelator = TexturePixelator()
        self.batch_processor = BatchProcessor()
        
        self.preview_image = None
        self.preview_photo = None
        self.preview_zoom = 1.0
        self.current_preview_path = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create the GUI layout"""
        
        # Main container with two columns
        container = ttk.Frame(self.root, padding="10")
        container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel for controls
        main_frame = ttk.Frame(container)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Pipeline tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Create tab frames
        file_tab = ttk.Frame(self.notebook, padding="10")
        prep_tab = ttk.Frame(self.notebook, padding="10")
        ao_tab = ttk.Frame(self.notebook, padding="10")
        pixel_tab = ttk.Frame(self.notebook, padding="10")
        
        self.notebook.add(file_tab, text="1. Files")
        self.notebook.add(prep_tab, text="2. Pre-Process")
        self.notebook.add(ao_tab, text="3. Surface Effects")
        self.notebook.add(pixel_tab, text="4. Pixelate")
        
        # Build each tab
        self.setup_file_tab(file_tab)
        self.setup_prep_tab(prep_tab)
        self.setup_surface_tab(ao_tab)
        self.setup_pixel_tab(pixel_tab)
        
        # Right panel for preview
        preview_frame = ttk.LabelFrame(container, text="Preview", padding="10")
        preview_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Zoom controls
        zoom_frame = ttk.Frame(preview_frame)
        zoom_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(zoom_frame, text="-", command=self.zoom_out, width=3).pack(side=tk.LEFT, padx=2)
        self.zoom_label = ttk.Label(zoom_frame, text="100%", width=6)
        self.zoom_label.pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="+", command=self.zoom_in, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Fit", command=self.zoom_fit, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="1:1", command=self.zoom_actual, width=5).pack(side=tk.LEFT, padx=2)
        
        # Preview canvas with scrollbars
        canvas_frame = ttk.Frame(preview_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.preview_canvas = tk.Canvas(canvas_frame, width=400, height=640, bg='#2b2b2b', highlightthickness=1, highlightbackground='#555')
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.preview_canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.preview_canvas.xview)
        
        self.preview_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.preview_canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.E, tk.W))
        
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Mouse wheel zoom
        self.preview_canvas.bind("<MouseWheel>", self.on_mousewheel_zoom)
        self.preview_canvas.bind("<Button-4>", self.on_mousewheel_zoom)  # Linux
        self.preview_canvas.bind("<Button-5>", self.on_mousewheel_zoom)  # Linux
        
        # Preview placeholder text
        self.preview_text_id = self.preview_canvas.create_text(
            200, 320, 
            text="No preview yet\n\nProcess a file to see preview", 
            fill='#888', 
            font=("TkDefaultFont", 11),
            justify=tk.CENTER
        )
        
        # Process buttons at bottom of main frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, pady=10)
        
        ttk.Button(button_frame, text="Process Single File", 
                  command=self.process_single, width=20).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Process Batch", 
                  command=self.process_batch, width=20).grid(row=0, column=1, padx=5)
        
        # Status messages
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X, expand=True)
        
        # Success message area
        self.success_frame = ttk.Frame(main_frame)
        self.success_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        self.success_label = ttk.Label(self.success_frame, text="", foreground="green", wraplength=500, justify=tk.LEFT)
        self.success_label.pack(fill=tk.X)
        
        # Error message area
        self.error_frame = ttk.Frame(main_frame)
        self.error_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        self.error_label = ttk.Label(self.error_frame, text="", foreground="red", wraplength=500, justify=tk.LEFT)
        self.error_label.pack(fill=tk.X)
        
        # Configure grid weights
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=2)
        container.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Initialize UI state AFTER all widgets are created
        self.on_quantize_method_change()
        self.on_dither_mode_change()
    
    def setup_file_tab(self, parent):
        """Setup Files tab"""
        # === FILE SELECTION ===
        file_frame = ttk.LabelFrame(parent, text="File Selection", padding="10")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Help text
        help_text = ttk.Label(file_frame, text="Choose ONE: Single file OR batch folder (not both)", 
                             foreground="gray", font=("TkDefaultFont", 9, "italic"))
        help_text.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        
        # Single file mode
        ttk.Label(file_frame, text="Single File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.input_file_var = tk.StringVar()
        single_entry = ttk.Entry(file_frame, textvariable=self.input_file_var, width=50)
        single_entry.grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_input_file).grid(row=1, column=2)
        ttk.Label(file_frame, text="Select one image file to pixelate", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Batch mode
        ttk.Label(file_frame, text="Batch Folder:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.input_folder_var = tk.StringVar()
        batch_entry = ttk.Entry(file_frame, textvariable=self.input_folder_var, width=50)
        batch_entry.grid(row=3, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_input_folder).grid(row=3, column=2)
        ttk.Label(file_frame, text="Select folder containing multiple images to process all at once", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=4, column=1, sticky=tk.W, padx=5)
        
        # Output location
        ttk.Label(file_frame, text="Output:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.output_var = tk.StringVar()
        output_entry = ttk.Entry(file_frame, textvariable=self.output_var, width=50)
        output_entry.grid(row=5, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_output).grid(row=5, column=2)
        ttk.Label(file_frame, text="Where to save: file path (single) or folder (batch)", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=6, column=1, sticky=tk.W, padx=5)
    
    def setup_prep_tab(self, parent):
        """Setup Pre-Process tab"""
        # === PRE-PROCESSING ===
        prep_frame = ttk.LabelFrame(parent, text="Pre-Processing (Phase 1)", padding="10")
        prep_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(prep_frame, text="Add complexity to solid colors before pixelation", 
                 foreground="gray", font=("TkDefaultFont", 9, "italic")).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        
        # Blur amount
        ttk.Label(prep_frame, text="Blur Amount:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.blur_amount_var = tk.DoubleVar(value=0.0)
        blur_scale = ttk.Scale(prep_frame, from_=0.0, to=5.0, variable=self.blur_amount_var, 
                              orient=tk.HORIZONTAL, length=200)
        blur_scale.grid(row=1, column=1, sticky=tk.W, padx=5)
        self.blur_label = ttk.Label(prep_frame, text="0.0")
        self.blur_label.grid(row=1, column=2, sticky=tk.W)
        self.blur_amount_var.trace_add('write', self.update_blur_label)
        ttk.Label(prep_frame, text="Softens edges, creates subtle gradients", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Noise amount
        ttk.Label(prep_frame, text="Noise Amount:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.noise_amount_var = tk.DoubleVar(value=0.0)
        noise_scale = ttk.Scale(prep_frame, from_=0.0, to=50.0, variable=self.noise_amount_var, 
                               orient=tk.HORIZONTAL, length=200)
        noise_scale.grid(row=3, column=1, sticky=tk.W, padx=5)
        self.noise_label = ttk.Label(prep_frame, text="0.0")
        self.noise_label.grid(row=3, column=2, sticky=tk.W)
        self.noise_amount_var.trace_add('write', self.update_noise_label)
        ttk.Label(prep_frame, text="Adds grain/texture variation", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=4, column=1, sticky=tk.W, padx=5)
        
        # Color variation
        ttk.Label(prep_frame, text="Color Variation:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.color_var_var = tk.DoubleVar(value=0.0)
        color_scale = ttk.Scale(prep_frame, from_=0.0, to=30.0, variable=self.color_var_var, 
                               orient=tk.HORIZONTAL, length=200)
        color_scale.grid(row=5, column=1, sticky=tk.W, padx=5)
        self.color_var_label = ttk.Label(prep_frame, text="0.0")
        self.color_var_label.grid(row=5, column=2, sticky=tk.W)
        self.color_var_var.trace_add('write', self.update_color_var_label)
        ttk.Label(prep_frame, text="Randomly shifts hue/saturation slightly", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=6, column=1, sticky=tk.W, padx=5)
    
    def setup_surface_tab(self, parent):
        """Setup Surface Effects tab (curvature + AO)"""
        # === SURFACE EFFECTS ENABLE/DISABLE ===
        enable_frame = ttk.LabelFrame(parent, text="Surface-Based Effects", padding="10")
        enable_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.enable_surface_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(enable_frame, text="Enable Surface Effects (requires 3D model)", 
                       variable=self.enable_surface_var, command=self.on_surface_toggle).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        ttk.Label(enable_frame, text="Add highlights to edges/ridges, darken crevices based on 3D mesh", 
                 foreground="gray", font=("TkDefaultFont", 9, "italic")).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        
        # === 3D MODEL INPUT ===
        model_frame = ttk.LabelFrame(parent, text="3D Model", padding="10")
        model_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(model_frame, text="Model File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_path_var = tk.StringVar()
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=35, state="disabled")
        model_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        self.model_browse_btn = ttk.Button(model_frame, text="Browse...", command=self.browse_model_file, state="disabled")
        self.model_browse_btn.grid(row=0, column=2, padx=5)
        
        ttk.Label(model_frame, text="Supports: OBJ, FBX, glTF/GLB", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        self.model_entry = model_entry  # Store reference for enable/disable
        
        # === CURVATURE SETTINGS ===
        curv_frame = ttk.LabelFrame(parent, text="Curvature Detection", padding="10")
        curv_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(curv_frame, text="Detect edges, ridges, and sharp features", 
                 foreground="gray", font=("TkDefaultFont", 9, "italic")).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        
        # Curvature Strength
        ttk.Label(curv_frame, text="Detection Strength:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.curv_strength_var = tk.DoubleVar(value=5.0)
        self.curv_strength_scale = ttk.Scale(curv_frame, from_=0.1, to=20.0, 
                                            variable=self.curv_strength_var, orient=tk.HORIZONTAL, 
                                            length=200, state="disabled")
        self.curv_strength_scale.grid(row=1, column=1, sticky=tk.W, padx=5)
        self.curv_strength_label = ttk.Label(curv_frame, text="0.50")
        self.curv_strength_label.grid(row=1, column=2, sticky=tk.W)
        self.curv_strength_var.trace_add('write', self.update_curv_strength_label)
        ttk.Label(curv_frame, text="How sensitive to surface curvature changes", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Edge Highlight
        ttk.Label(curv_frame, text="Edge Highlight:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.edge_highlight_var = tk.DoubleVar(value=0.3)
        self.edge_highlight_scale = ttk.Scale(curv_frame, from_=0.0, to=1.0, 
                                             variable=self.edge_highlight_var, orient=tk.HORIZONTAL, 
                                             length=200, state="disabled")
        self.edge_highlight_scale.grid(row=3, column=1, sticky=tk.W, padx=5)
        self.edge_highlight_label = ttk.Label(curv_frame, text="0.30")
        self.edge_highlight_label.grid(row=3, column=2, sticky=tk.W)
        self.edge_highlight_var.trace_add('write', self.update_edge_highlight_label)
        ttk.Label(curv_frame, text="Brighten convex edges/ridges (0 = none, 1 = white)", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=4, column=1, sticky=tk.W, padx=5)
        
        # Crevice Darken
        ttk.Label(curv_frame, text="Crevice Darken:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.crevice_darken_var = tk.DoubleVar(value=0.4)
        self.crevice_darken_scale = ttk.Scale(curv_frame, from_=0.0, to=1.0, 
                                             variable=self.crevice_darken_var, orient=tk.HORIZONTAL, 
                                             length=200, state="disabled")
        self.crevice_darken_scale.grid(row=5, column=1, sticky=tk.W, padx=5)
        self.crevice_darken_label = ttk.Label(curv_frame, text="0.40")
        self.crevice_darken_label.grid(row=5, column=2, sticky=tk.W)
        self.crevice_darken_var.trace_add('write', self.update_crevice_darken_label)
        ttk.Label(curv_frame, text="Darken concave areas/crevices (0 = none, 1 = black)", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=6, column=1, sticky=tk.W, padx=5)
        
        # === COLOR TINTING ===
        tint_frame = ttk.LabelFrame(parent, text="Edge/Crevice Color Tinting", padding="10")
        tint_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(tint_frame, text="Shift colors on edges and in crevices", 
                 foreground="gray", font=("TkDefaultFont", 9, "italic")).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        
        # Edge Hue shift
        ttk.Label(tint_frame, text="Edge Hue Shift:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.edge_hue_shift_var = tk.DoubleVar(value=30.0)
        self.edge_hue_scale = ttk.Scale(tint_frame, from_=-180, to=180, 
                                       variable=self.edge_hue_shift_var, orient=tk.HORIZONTAL, 
                                       length=200, state="disabled")
        self.edge_hue_scale.grid(row=1, column=1, sticky=tk.W, padx=5)
        self.edge_hue_label = ttk.Label(tint_frame, text="30°")
        self.edge_hue_label.grid(row=1, column=2, sticky=tk.W)
        self.edge_hue_shift_var.trace_add('write', self.update_edge_hue_label)
        ttk.Label(tint_frame, text="Hue shift on edges (e.g., +30° = warmer highlights)", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Crevice Hue shift
        ttk.Label(tint_frame, text="Crevice Hue Shift:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.crevice_hue_shift_var = tk.DoubleVar(value=-30.0)
        self.crevice_hue_scale = ttk.Scale(tint_frame, from_=-180, to=180, 
                                          variable=self.crevice_hue_shift_var, orient=tk.HORIZONTAL, 
                                          length=200, state="disabled")
        self.crevice_hue_scale.grid(row=3, column=1, sticky=tk.W, padx=5)
        self.crevice_hue_label = ttk.Label(tint_frame, text="-30°")
        self.crevice_hue_label.grid(row=3, column=2, sticky=tk.W)
        self.crevice_hue_shift_var.trace_add('write', self.update_crevice_hue_label)
        ttk.Label(tint_frame, text="Hue shift in crevices (e.g., -30° = cooler shadows)", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=4, column=1, sticky=tk.W, padx=5)
        
        # Saturation boost
        ttk.Label(tint_frame, text="Edge Saturation:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.edge_sat_var = tk.DoubleVar(value=0.2)
        self.edge_sat_scale = ttk.Scale(tint_frame, from_=-1.0, to=1.0, 
                                       variable=self.edge_sat_var, orient=tk.HORIZONTAL, 
                                       length=200, state="disabled")
        self.edge_sat_scale.grid(row=5, column=1, sticky=tk.W, padx=5)
        self.edge_sat_label = ttk.Label(tint_frame, text="0.20")
        self.edge_sat_label.grid(row=5, column=2, sticky=tk.W)
        self.edge_sat_var.trace_add('write', self.update_edge_sat_label)
        ttk.Label(tint_frame, text="Saturation change on edges (-1 = gray, +1 = vibrant)", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=6, column=1, sticky=tk.W, padx=5)
        
        # Store references for enable/disable
        self.surface_widgets = {
            'scales': [
                self.curv_strength_scale, self.edge_highlight_scale, self.crevice_darken_scale,
                self.edge_hue_scale, self.crevice_hue_scale, self.edge_sat_scale
            ]
        }
    
    def setup_pixel_tab(self, parent):
        """Setup Pixelate tab"""
        
        # === PIXELATION SETTINGS ===
        pixel_frame = ttk.LabelFrame(parent, text="Pixelation Settings (Phase 2)", padding="10")
        pixel_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Pixel width
        ttk.Label(pixel_frame, text="Pixel Width:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.pixel_width_var = tk.IntVar(value=64)
        pixel_width_spinbox = ttk.Spinbox(pixel_frame, from_=8, to=512, textvariable=self.pixel_width_var, width=10)
        pixel_width_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(pixel_frame, text="pixels (lower = more pixelated)").grid(row=0, column=2, sticky=tk.W)
        
        # Resample mode
        ttk.Label(pixel_frame, text="Resample Mode:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.resample_var = tk.StringVar(value="nearest")
        resample_combo = ttk.Combobox(pixel_frame, textvariable=self.resample_var, 
                                      values=["nearest", "bilinear"], state="readonly", width=15)
        resample_combo.grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Label(pixel_frame, text="nearest = sharp, bilinear = softer", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=1, column=2, sticky=tk.W, padx=5)
        
        # === COLOR QUANTIZATION ===
        color_frame = ttk.LabelFrame(parent, text="Color Quantization (Phase 2)", padding="10")
        color_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Quantization method
        ttk.Label(color_frame, text="Method:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.quantize_method_var = tk.StringVar(value="bit_depth")
        quantize_combo = ttk.Combobox(color_frame, textvariable=self.quantize_method_var,
                                     values=["none", "bit_depth", "palette"], state="readonly", width=15)
        quantize_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        quantize_combo.bind('<<ComboboxSelected>>', self.on_quantize_method_change)
        ttk.Label(color_frame, text="How to reduce colors", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=0, column=2, sticky=tk.W, padx=5)
        
        # Bits per channel (for bit_depth mode)
        ttk.Label(color_frame, text="Bits per Channel:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.bits_per_channel_var = tk.IntVar(value=5)
        self.bits_spinbox = ttk.Spinbox(color_frame, from_=1, to=8, textvariable=self.bits_per_channel_var, width=10)
        self.bits_spinbox.grid(row=1, column=1, sticky=tk.W, padx=5)
        self.bits_label = ttk.Label(color_frame, text="(32 colors per channel)")
        self.bits_label.grid(row=1, column=2, sticky=tk.W)
        
        # Palette colors (for palette mode)
        ttk.Label(color_frame, text="Palette Colors:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.palette_colors_var = tk.IntVar(value=32)
        self.palette_spinbox = ttk.Spinbox(color_frame, from_=2, to=256, textvariable=self.palette_colors_var, width=10)
        self.palette_spinbox.grid(row=2, column=1, sticky=tk.W, padx=5)
        self.palette_label = ttk.Label(color_frame, text="(total colors)")
        self.palette_label.grid(row=2, column=2, sticky=tk.W)
        
        # === DITHERING ===
        dither_frame = ttk.LabelFrame(parent, text="Dithering (Phase 2)", padding="10")
        dither_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Dither mode
        ttk.Label(dither_frame, text="Dither Mode:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.dither_mode_var = tk.StringVar(value="none")
        dither_combo = ttk.Combobox(dither_frame, textvariable=self.dither_mode_var,
                                   values=["none", "bayer", "floyd_steinberg"], state="readonly", width=15)
        dither_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        dither_combo.bind('<<ComboboxSelected>>', self.on_dither_mode_change)
        ttk.Label(dither_frame, text="Adds retro dot pattern", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=0, column=2, sticky=tk.W, padx=5)
        
        # Bayer matrix size (for bayer mode)
        ttk.Label(dither_frame, text="Bayer Matrix:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.bayer_size_var = tk.StringVar(value="4x4")
        self.bayer_combo = ttk.Combobox(dither_frame, textvariable=self.bayer_size_var,
                                       values=["2x2", "4x4", "8x8"], state="readonly", width=15)
        self.bayer_combo.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Dither strength
        ttk.Label(dither_frame, text="Dither Strength:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.dither_strength_var = tk.DoubleVar(value=0.5)
        self.strength_scale = ttk.Scale(dither_frame, from_=0.0, to=1.0, 
                                       variable=self.dither_strength_var, orient=tk.HORIZONTAL, length=200)
        self.strength_scale.grid(row=2, column=1, sticky=tk.W, padx=5)
        self.strength_label = ttk.Label(dither_frame, text="0.50")
        self.strength_label.grid(row=2, column=2, sticky=tk.W)
        self.dither_strength_var.trace_add('write', self.update_strength_label)
        
        # === ADVANCED OPTIONS ===
        advanced_frame = ttk.LabelFrame(parent, text="Advanced Options", padding="10")
        advanced_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Normal map checkbox (stubbed out for future)
        self.is_normal_map_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(advanced_frame, text="Normal Map (future use)", 
                       variable=self.is_normal_map_var, state="disabled").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Add suffix in batch mode
        self.add_suffix_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="Add '_pixelated' suffix (batch mode)", 
                       variable=self.add_suffix_var).grid(row=1, column=0, sticky=tk.W, pady=5)
    
    def update_blur_label(self, *args):
        self.blur_label.config(text=f"{self.blur_amount_var.get():.1f}")
    
    def update_noise_label(self, *args):
        self.noise_label.config(text=f"{self.noise_amount_var.get():.1f}")
    
    def update_color_var_label(self, *args):
        self.color_var_label.config(text=f"{self.color_var_var.get():.1f}")
    
    def update_curv_strength_label(self, *args):
        self.curv_strength_label.config(text=f"{self.curv_strength_var.get():.2f}")
    
    def update_edge_highlight_label(self, *args):
        self.edge_highlight_label.config(text=f"{self.edge_highlight_var.get():.2f}")
    
    def update_crevice_darken_label(self, *args):
        self.crevice_darken_label.config(text=f"{self.crevice_darken_var.get():.2f}")
    
    def update_edge_hue_label(self, *args):
        self.edge_hue_label.config(text=f"{self.edge_hue_shift_var.get():.0f}°")
    
    def update_crevice_hue_label(self, *args):
        self.crevice_hue_label.config(text=f"{self.crevice_hue_shift_var.get():.0f}°")
    
    def update_edge_sat_label(self, *args):
        self.edge_sat_label.config(text=f"{self.edge_sat_var.get():.2f}")
    
    def on_surface_toggle(self):
        """Enable/disable surface effect controls based on checkbox"""
        enabled = self.enable_surface_var.get()
        state = "normal" if enabled else "disabled"
        btn_state = "normal" if enabled else "disabled"
        
        self.model_entry.config(state=state)
        self.model_browse_btn.config(state=btn_state)
        
        for scale in self.surface_widgets['scales']:
            scale.config(state=state)
    
    def browse_model_file(self):
        filename = filedialog.askopenfilename(
            title="Select 3D Model",
            filetypes=[
                ("3D Models", "*.obj *.fbx *.gltf *.glb"),
                ("Wavefront OBJ", "*.obj"),
                ("Autodesk FBX", "*.fbx"),
                ("glTF", "*.gltf *.glb"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            self.model_path_var.set(filename)
    
    def browse_input_file(self):
        filename = filedialog.askopenfilename(
            title="Select Input Texture",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tga *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.input_file_var.set(filename)
            # Auto-set output if not set
            if not self.output_var.get():
                base, ext = os.path.splitext(filename)
                self.output_var.set(f"{base}_pixelated{ext}")
    
    def browse_input_folder(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_folder_var.set(folder)
            # Auto-set output if not set
            if not self.output_var.get():
                self.output_var.set(f"{folder}_pixelated")
    
    def browse_output(self):
        # Determine if single file or folder based on which input is filled
        if self.input_file_var.get():
            filename = filedialog.asksaveasfilename(
                title="Save Output As",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("All files", "*.*")
                ],
                defaultextension=".png"
            )
            if filename:
                self.output_var.set(filename)
        else:
            folder = filedialog.askdirectory(title="Select Output Folder")
            if folder:
                self.output_var.set(folder)
    
    def on_quantize_method_change(self, event=None):
        method = self.quantize_method_var.get()
        if method == "bit_depth":
            self.bits_spinbox.config(state="normal")
            self.palette_spinbox.config(state="readonly")
        elif method == "palette":
            self.bits_spinbox.config(state="readonly")
            self.palette_spinbox.config(state="normal")
        else:  # none
            self.bits_spinbox.config(state="readonly")
            self.palette_spinbox.config(state="readonly")
    
    def on_dither_mode_change(self, event=None):
        mode = self.dither_mode_var.get()
        if mode == "bayer":
            self.bayer_combo.config(state="readonly")
            self.strength_scale.config(state="normal")
        else:
            self.bayer_combo.config(state="disabled")
            if mode == "none":
                self.strength_scale.config(state="disabled")
            else:
                self.strength_scale.config(state="normal")
    
    def update_strength_label(self, *args):
        self.strength_label.config(text=f"{self.dither_strength_var.get():.2f}")
    
    def get_settings(self):
        """Get current settings as dictionary"""
        return {
            'blur_amount': self.blur_amount_var.get(),
            'noise_amount': self.noise_amount_var.get(),
            'color_variation': self.color_var_var.get(),
            'enable_surface': self.enable_surface_var.get(),
            'model_path': self.model_path_var.get() if self.enable_surface_var.get() else None,
            'curvature_strength': self.curv_strength_var.get(),
            'edge_highlight': self.edge_highlight_var.get(),
            'crevice_darken': self.crevice_darken_var.get(),
            'edge_hue_shift': self.edge_hue_shift_var.get(),
            'crevice_hue_shift': self.crevice_hue_shift_var.get(),
            'edge_saturation': self.edge_sat_var.get(),
            'pixel_width': self.pixel_width_var.get(),
            'resample_mode': self.resample_var.get(),
            'quantize_method': self.quantize_method_var.get(),
            'bits_per_channel': self.bits_per_channel_var.get(),
            'palette_colors': self.palette_colors_var.get(),
            'dither_mode': self.dither_mode_var.get(),
            'bayer_size': self.bayer_size_var.get(),
            'dither_strength': self.dither_strength_var.get(),
            'is_normal_map': self.is_normal_map_var.get()
        }
    
    def get_batch_settings(self):
        """Get settings for batch processing (includes add_suffix)"""
        settings = self.get_settings()
        settings['add_suffix'] = self.add_suffix_var.get()
        return settings
    
    def update_preview(self, image_path):
        """Update the preview with the processed image"""
        try:
            from PIL import Image
            
            # Store the path for re-rendering at different zooms
            self.current_preview_path = image_path
            
            # Load image
            self.preview_image = Image.open(image_path)
            
            # Reset zoom to fit
            self.preview_zoom = 1.0
            self.zoom_fit()
            
        except Exception as e:
            print(f"Preview error: {e}")
    
    def render_preview(self):
        """Render the preview at current zoom level"""
        if not self.preview_image:
            return
        
        try:
            img = self.preview_image.copy()
            
            # Calculate zoomed size
            new_width = int(img.width * self.preview_zoom)
            new_height = int(img.height * self.preview_zoom)
            
            # Resize with nearest neighbor to preserve pixels
            img = img.resize((new_width, new_height), Image.NEAREST)
            
            # Convert to PhotoImage
            self.preview_photo = ImageTk.PhotoImage(img)
            
            # Clear canvas
            self.preview_canvas.delete("all")
            
            # Update scroll region
            self.preview_canvas.config(scrollregion=(0, 0, new_width, new_height))
            
            # Draw image
            self.preview_canvas.create_image(0, 0, image=self.preview_photo, anchor=tk.NW)
            
            # Update zoom label
            self.zoom_label.config(text=f"{int(self.preview_zoom * 100)}%")
            
        except Exception as e:
            print(f"Render error: {e}")
    
    def zoom_in(self):
        """Zoom in by 25%"""
        if self.preview_image:
            self.preview_zoom = min(self.preview_zoom * 1.25, 10.0)  # Max 1000%
            self.render_preview()
    
    def zoom_out(self):
        """Zoom out by 25%"""
        if self.preview_image:
            self.preview_zoom = max(self.preview_zoom / 1.25, 0.1)  # Min 10%
            self.render_preview()
    
    def zoom_fit(self):
        """Zoom to fit in canvas"""
        if not self.preview_image:
            return
        
        # Get canvas size
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        # Use default size if not yet rendered
        if canvas_width <= 1:
            canvas_width = 400
            canvas_height = 640
        
        # Calculate zoom to fit
        zoom_x = (canvas_width - 20) / self.preview_image.width
        zoom_y = (canvas_height - 20) / self.preview_image.height
        
        self.preview_zoom = min(zoom_x, zoom_y, 1.0)  # Don't zoom in past 100% for fit
        self.render_preview()
    
    def zoom_actual(self):
        """Zoom to actual size (100%)"""
        if self.preview_image:
            self.preview_zoom = 1.0
            self.render_preview()
    
    def on_mousewheel_zoom(self, event):
        """Handle mouse wheel zoom"""
        if not self.preview_image:
            return
        
        # Determine zoom direction
        if event.num == 4 or event.delta > 0:
            # Zoom in
            self.preview_zoom = min(self.preview_zoom * 1.1, 10.0)
        elif event.num == 5 or event.delta < 0:
            # Zoom out
            self.preview_zoom = max(self.preview_zoom / 1.1, 0.1)
        
        self.render_preview()
    
    def clear_messages(self):
        """Clear success and error messages"""
        self.success_label.config(text="")
        self.error_label.config(text="")
    
    def show_success(self, message):
        """Show success message in GUI"""
        self.clear_messages()
        self.success_label.config(text=f"✓ {message}")
    
    def show_error(self, message):
        """Show error message in GUI"""
        self.clear_messages()
        self.error_label.config(text=f"✗ {message}")
    
    def process_single(self):
        input_file = self.input_file_var.get()
        output_file = self.output_var.get()
        
        if not input_file or not os.path.exists(input_file):
            self.show_error("Please select a valid input file")
            return
        
        if not output_file:
            self.show_error("Please specify an output file")
            return
        
        self.clear_messages()
        self.status_var.set("Processing...")
        self.root.update()
        
        settings = self.get_settings()
        
        success = self.pixelator.process_texture(
            input_path=input_file,
            output_path=output_file,
            **settings
        )
        
        if success:
            self.status_var.set("Ready")
            self.show_success(f"Saved to: {os.path.basename(output_file)}")
            self.update_preview(output_file)
        else:
            self.status_var.set("Ready")
            self.show_error("Failed to process texture. Check console for details.")
    
    def process_batch(self):
        input_folder = self.input_folder_var.get()
        output_folder = self.output_var.get()
        
        if not input_folder or not os.path.exists(input_folder):
            self.show_error("Please select a valid input folder")
            return
        
        if not output_folder:
            self.show_error("Please specify an output folder")
            return
        
        self.clear_messages()
        self.status_var.set("Processing batch...")
        self.root.update()
        
        settings = self.get_batch_settings()
        results = self.batch_processor.process_batch(input_folder, output_folder, settings)
        
        self.status_var.set("Ready")
        
        if results['success'] > 0:
            message = f"Processed {results['success']}/{results['total']} files successfully"
            if results['failed'] > 0:
                message += f" ({results['failed']} failed)"
            self.show_success(message)
        else:
            self.show_error(f"Batch failed: {results['failed']}/{results['total']} files failed to process")


def main():
    root = tk.Tk()
    app = PixelatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
