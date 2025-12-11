"""
Texture Pixelator GUI - Simple interface for parameter tweaking
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser, simpledialog
import os
from pathlib import Path
from texture_pixelator import TexturePixelator
from batch_process import BatchProcessor
from project_manager import ProjectManager
from PIL import Image, ImageTk, ImageFilter


class PixelatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Texture Pixelator")
        self.root.geometry("1200x900")
        self.root.minsize(900, 600)
        self.root.resizable(True, True)
        
        self.pixelator = TexturePixelator()
        self.batch_processor = BatchProcessor()
        self.project_manager = ProjectManager()
        
        self.preview_image = None
        self.preview_photo = None
        self.preview_zoom = 1.0
        self.current_preview_path = None
        self.current_project_path = None
        self.project_modified = False
        
        # Recent projects list (max 10)
        self.recent_projects = self.load_recent_projects()
        
        # Color picker variables
        self.edge_color_var = '#FF8800'  # Orange
        self.ao_color_var = '#321E14'  # Dark brown for AO
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create the GUI layout"""
        
        # Create menu bar
        self.setup_menu()
        
        # Main container with two columns
        container = ttk.Frame(self.root, padding="10")
        container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel for controls
        main_frame = ttk.Frame(container)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Pipeline tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Create scrollable tab frames
        file_tab_canvas, file_tab = self.create_scrollable_tab()
        ao_tab_canvas, ao_tab = self.create_scrollable_tab()
        prep_tab_canvas, prep_tab = self.create_scrollable_tab()
        pixel_tab_canvas, pixel_tab = self.create_scrollable_tab()
        
        self.notebook.add(file_tab_canvas, text="1. Files")
        self.notebook.add(ao_tab_canvas, text="2. Surface Effects")
        self.notebook.add(prep_tab_canvas, text="3. Pre-Process")
        self.notebook.add(pixel_tab_canvas, text="4. Pixelate")
        
        # Build each tab
        self.setup_file_tab(file_tab)
        self.setup_surface_tab(ao_tab)
        self.setup_prep_tab(prep_tab)
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
        
        self.preview_canvas = tk.Canvas(canvas_frame, bg='#2b2b2b', highlightthickness=1, highlightbackground='#555')
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
        
        # Preview placeholder text (will be repositioned on first update)
        self.preview_text_id = self.preview_canvas.create_text(
            0, 0, 
            text="No preview yet\n\nProcess a file to see preview", 
            fill='#888', 
            font=("TkDefaultFont", 11),
            justify=tk.CENTER
        )
        # Update placeholder position after canvas is rendered
        self.root.after(100, self.center_placeholder_text)
        
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
        
        # Configure grid weights for responsive layout
        container.columnconfigure(0, weight=2)  # Left panel (controls)
        container.columnconfigure(1, weight=3)  # Right panel (preview)
        container.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=10)  # Notebook takes most space
        main_frame.rowconfigure(1, weight=0)   # Process buttons fixed size
        main_frame.rowconfigure(2, weight=0)   # Status fixed size
        main_frame.rowconfigure(3, weight=0)   # Success fixed size
        main_frame.rowconfigure(4, weight=0)   # Error fixed size
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
        single_entry = ttk.Entry(file_frame, textvariable=self.input_file_var, width=30)
        single_entry.grid(row=1, column=1, padx=5, sticky=(tk.W, tk.E))
        ttk.Button(file_frame, text="Browse", command=self.browse_input_file).grid(row=1, column=2)
        ttk.Label(file_frame, text="Select one image file to pixelate", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Batch mode
        ttk.Label(file_frame, text="Batch Folder:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.input_folder_var = tk.StringVar()
        batch_entry = ttk.Entry(file_frame, textvariable=self.input_folder_var, width=30)
        batch_entry.grid(row=3, column=1, padx=5, sticky=(tk.W, tk.E))
        ttk.Button(file_frame, text="Browse", command=self.browse_input_folder).grid(row=3, column=2)
        ttk.Label(file_frame, text="Select folder containing multiple images to process all at once", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=4, column=1, sticky=tk.W, padx=5)
        
        # Output location
        ttk.Label(file_frame, text="Output:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.output_var = tk.StringVar()
        output_entry = ttk.Entry(file_frame, textvariable=self.output_var, width=30)
        output_entry.grid(row=5, column=1, padx=5, sticky=(tk.W, tk.E))
        ttk.Button(file_frame, text="Browse", command=self.browse_output).grid(row=5, column=2)
        ttk.Label(file_frame, text="Where to save: file path (single) or folder (batch)", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=6, column=1, sticky=tk.W, padx=5)
        
        # Make column 1 expandable
        file_frame.columnconfigure(1, weight=1)
    
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
        
        # Flood fill color overlay
        ttk.Label(prep_frame, text="Flood Fill Color:").grid(row=7, column=0, sticky=tk.W, pady=5)
        self.flood_fill_var = tk.StringVar(value="#FFFFFF")
        self.flood_fill_canvas = tk.Canvas(prep_frame, width=30, height=30, 
                                           highlightthickness=1, highlightbackground="#999", cursor="hand2")
        self.flood_fill_canvas.grid(row=7, column=1, sticky=tk.W, padx=5)
        self.flood_fill_canvas.bind("<Button-1>", lambda e: self.pick_flood_fill_color())
        self.flood_fill_circle = self.flood_fill_canvas.create_oval(2, 2, 28, 28, 
                                                                     fill=self.flood_fill_var.get(), outline="#666", width=2)
        ttk.Label(prep_frame, text="Color to overlay on entire image", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=8, column=1, sticky=tk.W, padx=5)
        
        # Flood fill opacity
        ttk.Label(prep_frame, text="Flood Fill Opacity:").grid(row=9, column=0, sticky=tk.W, pady=5)
        self.flood_fill_opacity_var = tk.DoubleVar(value=0.0)
        flood_opacity_scale = ttk.Scale(prep_frame, from_=0.0, to=1.0, variable=self.flood_fill_opacity_var, 
                                        orient=tk.HORIZONTAL, length=200)
        flood_opacity_scale.grid(row=9, column=1, sticky=tk.W, padx=5)
        self.flood_opacity_label = ttk.Label(prep_frame, text="0.00")
        self.flood_opacity_label.grid(row=9, column=2, sticky=tk.W)
        self.flood_fill_opacity_var.trace_add('write', self.update_flood_opacity_label)
        ttk.Label(prep_frame, text="0 = none, 1 = full color overlay", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=10, column=1, sticky=tk.W, padx=5)
        
        # Hue shift
        ttk.Label(prep_frame, text="Hue Shift:").grid(row=11, column=0, sticky=tk.W, pady=5)
        self.hue_shift_var = tk.DoubleVar(value=0.0)
        hue_shift_scale = ttk.Scale(prep_frame, from_=-180.0, to=180.0, variable=self.hue_shift_var, 
                                    orient=tk.HORIZONTAL, length=200)
        hue_shift_scale.grid(row=11, column=1, sticky=tk.W, padx=5)
        self.hue_shift_label = ttk.Label(prep_frame, text="0")
        self.hue_shift_label.grid(row=11, column=2, sticky=tk.W)
        self.hue_shift_var.trace_add('write', self.update_hue_shift_label)
        ttk.Label(prep_frame, text="Rotate hue around color wheel (-180 to 180)", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=12, column=1, sticky=tk.W, padx=5)
        
        # Tint strength
        ttk.Label(prep_frame, text="Tint Strength:").grid(row=13, column=0, sticky=tk.W, pady=5)
        self.tint_strength_var = tk.DoubleVar(value=0.0)
        tint_strength_scale = ttk.Scale(prep_frame, from_=0.0, to=1.0, variable=self.tint_strength_var, 
                                        orient=tk.HORIZONTAL, length=200)
        tint_strength_scale.grid(row=13, column=1, sticky=tk.W, padx=5)
        self.tint_strength_label = ttk.Label(prep_frame, text="0.00")
        self.tint_strength_label.grid(row=13, column=2, sticky=tk.W)
        self.tint_strength_var.trace_add('write', self.update_tint_strength_label)
        ttk.Label(prep_frame, text="Push colors toward flood fill color (0=none, 1=full)", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=14, column=1, sticky=tk.W, padx=5)
    
    def setup_surface_tab(self, parent):
        """Setup Surface Effects tab (curvature + AO)"""
        # === SURFACE EFFECTS ENABLE/DISABLE ===
        enable_frame = ttk.LabelFrame(parent, text="Surface-Based Effects", padding="10")
        enable_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.enable_surface_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(enable_frame, text="Enable Surface Effects (requires 3D model)", 
                       variable=self.enable_surface_var, command=self.on_surface_toggle).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        ttk.Label(enable_frame, text="Blend custom colors onto edges from baked edge map", 
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
        
        # Edge Saturation
        ttk.Label(curv_frame, text="Edge Saturation:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.edge_saturation_var = tk.DoubleVar(value=0.0)
        self.edge_saturation_scale = ttk.Scale(curv_frame, from_=-1.0, to=1.0, 
                                               variable=self.edge_saturation_var, orient=tk.HORIZONTAL, 
                                               length=200, state="disabled")
        self.edge_saturation_scale.grid(row=5, column=1, sticky=tk.W, padx=5)
        self.edge_saturation_label = ttk.Label(curv_frame, text="0.00")
        self.edge_saturation_label.grid(row=5, column=2, sticky=tk.W)
        self.edge_saturation_var.trace_add('write', self.update_edge_saturation_label)
        ttk.Label(curv_frame, text="Boost saturation at edges (-1 = desaturate, 1 = boost)", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=6, column=1, sticky=tk.W, padx=5)
        
        # === COLOR TINTING ===
        tint_frame = ttk.LabelFrame(parent, text="Edge Color Tinting", padding="10")
        tint_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Enable Edge checkbox
        self.enable_edge_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tint_frame, text="Enable Edge Effects", variable=self.enable_edge_var).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        
        ttk.Label(tint_frame, text="Blend custom color onto edges (requires baked edge map)", 
                 foreground="gray", font=("TkDefaultFont", 9, "italic")).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        
        # Edge Color Picker
        ttk.Label(tint_frame, text="Edge Color:").grid(row=2, column=0, sticky=tk.W, pady=5)
        
        # Create canvas for circular color indicator
        self.edge_color_canvas = tk.Canvas(tint_frame, width=30, height=30, 
                                          highlightthickness=1, highlightbackground="#999", cursor="")
        self.edge_color_canvas.grid(row=2, column=1, sticky=tk.W, padx=5)
        self.edge_color_canvas.bind("<Button-1>", lambda e: self.pick_edge_color())
        self.edge_color_circle = self.edge_color_canvas.create_oval(2, 2, 28, 28, 
                                                                    fill=self.edge_color_var, outline="#666", width=2)
        
        ttk.Label(tint_frame, text="Color to blend onto convex edges/ridges", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=3, column=1, sticky=tk.W, padx=5)
        
        # Edge Color Blend Strength
        ttk.Label(tint_frame, text="Edge Color Blend:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.edge_blend_var = tk.DoubleVar(value=0.7)
        self.edge_blend_scale = ttk.Scale(tint_frame, from_=0.0, to=1.0, 
                                         variable=self.edge_blend_var, orient=tk.HORIZONTAL, 
                                         length=200, state="disabled")
        self.edge_blend_scale.grid(row=3, column=1, sticky=tk.W, padx=5)
        self.edge_blend_label = ttk.Label(tint_frame, text="0.70")
        self.edge_blend_label.grid(row=3, column=2, sticky=tk.W)
        self.edge_blend_var.trace_add('write', self.update_edge_blend_label)
        ttk.Label(tint_frame, text="How much edge color to blend (0 = none, 1 = full)", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=4, column=1, sticky=tk.W, padx=5)
        
        # === AO / CREVICE EFFECTS ===
        ao_frame = ttk.LabelFrame(parent, text="Ambient Occlusion (Crevices)", padding="10")
        ao_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Enable AO checkbox
        self.enable_ao_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ao_frame, text="Enable AO Effects", variable=self.enable_ao_var).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        
        ttk.Label(ao_frame, text="Darken and tint crevices (optional, requires baked AO map)", 
                 foreground="gray", font=("TkDefaultFont", 9, "italic")).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        
        # AO Darken
        ttk.Label(ao_frame, text="AO Darken:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.ao_darken_var = tk.DoubleVar(value=0.5)
        self.ao_darken_scale = ttk.Scale(ao_frame, from_=0.0, to=1.0, 
                                         variable=self.ao_darken_var, orient=tk.HORIZONTAL, 
                                         length=200, state="disabled")
        self.ao_darken_scale.grid(row=2, column=1, sticky=tk.W, padx=5)
        self.ao_darken_label = ttk.Label(ao_frame, text="0.50")
        self.ao_darken_label.grid(row=2, column=2, sticky=tk.W)
        self.ao_darken_var.trace_add('write', self.update_ao_darken_label)
        ttk.Label(ao_frame, text="Darken occluded areas (0 = none, 1 = black)", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=3, column=1, sticky=tk.W, padx=5)
        
        # AO Color Picker
        ttk.Label(ao_frame, text="AO Color:").grid(row=4, column=0, sticky=tk.W, pady=5)
        
        self.ao_color_canvas = tk.Canvas(ao_frame, width=30, height=30, 
                                         highlightthickness=1, highlightbackground="#999", cursor="")
        self.ao_color_canvas.grid(row=3, column=1, sticky=tk.W, padx=5)
        self.ao_color_canvas.bind("<Button-1>", lambda e: self.pick_ao_color())
        self.ao_color_circle = self.ao_color_canvas.create_oval(2, 2, 28, 28, 
                                                                fill=self.ao_color_var, outline="#666", width=2)
        
        ttk.Label(ao_frame, text="Color to blend into occluded areas", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=4, column=1, sticky=tk.W, padx=5)
        
        # AO Color Blend Strength
        ttk.Label(ao_frame, text="AO Color Blend:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.ao_blend_var = tk.DoubleVar(value=0.5)
        self.ao_blend_scale = ttk.Scale(ao_frame, from_=0.0, to=1.0, 
                                       variable=self.ao_blend_var, orient=tk.HORIZONTAL, 
                                       length=200, state="disabled")
        self.ao_blend_scale.grid(row=5, column=1, sticky=tk.W, padx=5)
        self.ao_blend_label = ttk.Label(ao_frame, text="0.50")
        self.ao_blend_label.grid(row=5, column=2, sticky=tk.W)
        self.ao_blend_var.trace_add('write', self.update_ao_blend_label)
        ttk.Label(ao_frame, text="How much AO color to blend (0 = none, 1 = full)", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=6, column=1, sticky=tk.W, padx=5)
        
        # === BAKE MAPS ===
        bake_frame = ttk.LabelFrame(parent, text="Bake Maps", padding="10")
        bake_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(bake_frame, text="Pre-generate curvature maps to save processing time", 
                 foreground="gray", font=("TkDefaultFont", 9, "italic")).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        
        # Map resolution
        ttk.Label(bake_frame, text="Map Resolution:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.bake_resolution_var = tk.IntVar(value=1024)
        ttk.Spinbox(bake_frame, from_=256, to=4096, increment=256, textvariable=self.bake_resolution_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Label(bake_frame, text="pixels (higher = more detail)", foreground="gray", font=("TkDefaultFont", 8)).grid(row=1, column=2, sticky=tk.W, padx=5)
        
        # Bake buttons and status
        button_row = ttk.Frame(bake_frame)
        button_row.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(10, 5))
        
        self.bake_edge_button = ttk.Button(button_row, text="Bake Edge Map", 
                                           command=self.bake_edge_maps, state="disabled")
        self.bake_edge_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.bake_ao_button = ttk.Button(button_row, text="Bake AO Map", 
                                         command=self.bake_ao_map, state="disabled")
        self.bake_ao_button.pack(side=tk.LEFT)
        
        self.edge_status_label = ttk.Label(bake_frame, text="Edge: Not found", 
                                           foreground="gray", font=("TkDefaultFont", 8))
        self.edge_status_label.grid(row=3, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(0, 2))
        
        self.ao_status_label = ttk.Label(bake_frame, text="AO: Not found", 
                                         foreground="gray", font=("TkDefaultFont", 8))
        self.ao_status_label.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(0, 5))
        
        # Store references for enable/disable
        self.surface_widgets = {
            'scales': [
                self.curv_strength_scale, self.edge_highlight_scale, self.edge_saturation_scale, self.ao_darken_scale
            ],
            'canvases': [self.edge_color_canvas, self.ao_color_canvas],
            'buttons': [self.bake_edge_button, self.bake_ao_button]
        }
        
        # Check for existing baked map on project load
        self.check_baked_map_status()
    
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
        
        # Greedy expansion toggle
        self.enable_greedy_expand_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pixel_frame, text="Greedy Pixel Expansion (prevents background bleed at UV seams)", 
                       variable=self.enable_greedy_expand_var, command=self.on_greedy_toggle).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(10, 5))
        ttk.Label(pixel_frame, text="Expands foreground pixels outward to prevent background showing at mesh seams", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=3, column=0, columnspan=3, sticky=tk.W, padx=(20, 5))
        
        # Greedy expansion strength
        ttk.Label(pixel_frame, text="Expansion Strength:").grid(row=4, column=0, sticky=tk.W, pady=5, padx=(20, 0))
        self.greedy_iterations_var = tk.IntVar(value=5)
        self.greedy_slider = ttk.Scale(pixel_frame, from_=1, to=20, 
                                       variable=self.greedy_iterations_var, orient=tk.HORIZONTAL, 
                                       length=200)
        self.greedy_slider.grid(row=4, column=1, sticky=tk.W, padx=5)
        self.greedy_label = ttk.Label(pixel_frame, text="5")
        self.greedy_label.grid(row=4, column=2, sticky=tk.W)
        self.greedy_iterations_var.trace_add('write', self.update_greedy_label)
        ttk.Label(pixel_frame, text="How many pixels outward to expand (1=thin, 20=thick)", 
                 foreground="gray", font=("TkDefaultFont", 8)).grid(row=5, column=1, columnspan=2, sticky=tk.W, padx=5)
        
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
    
    def update_flood_opacity_label(self, *args):
        self.flood_opacity_label.config(text=f"{self.flood_fill_opacity_var.get():.2f}")
    
    def update_hue_shift_label(self, *args):
        self.hue_shift_label.config(text=f"{int(self.hue_shift_var.get())}")
    
    def update_tint_strength_label(self, *args):
        self.tint_strength_label.config(text=f"{self.tint_strength_var.get():.2f}")
    
    def update_curv_strength_label(self, *args):
        self.curv_strength_label.config(text=f"{self.curv_strength_var.get():.2f}")
    
    def update_edge_blend_label(self, *args):
        self.edge_blend_label.config(text=f"{self.edge_blend_var.get():.2f}")
    
    def update_ao_blend_label(self, *args):
        self.ao_blend_label.config(text=f"{self.ao_blend_var.get():.2f}")
    
    def update_edge_highlight_label(self, *args):
        self.edge_highlight_label.config(text=f"{self.edge_highlight_var.get():.2f}")
        self.mark_modified()
    
    def update_edge_saturation_label(self, *args):
        self.edge_saturation_label.config(text=f"{self.edge_saturation_var.get():.2f}")
        self.mark_modified()
    
    def update_ao_darken_label(self, *args):
        self.ao_darken_label.config(text=f"{self.ao_darken_var.get():.2f}")
        self.mark_modified()
    
    def update_greedy_label(self, *args):
        self.greedy_label.config(text=f"{self.greedy_iterations_var.get()}")
    
    def on_greedy_toggle(self):
        """Enable/disable greedy expansion slider"""
        enabled = self.enable_greedy_expand_var.get()
        state = "normal" if enabled else "disabled"
        self.greedy_slider.config(state=state)
    
    def on_surface_toggle(self):
        """Enable/disable surface effect controls based on checkbox"""
        enabled = self.enable_surface_var.get()
        state = "normal" if enabled else "disabled"
        btn_state = "normal" if enabled else "disabled"
        
        self.model_entry.config(state=state)
        self.model_browse_btn.config(state=btn_state)
        
        for scale in self.surface_widgets['scales']:
            scale.config(state=state)
        
        for canvas in self.surface_widgets['canvases']:
            if enabled:
                canvas.config(cursor="hand2")
            else:
                canvas.config(cursor="")
        
        for button in self.surface_widgets['buttons']:
            button.config(state=btn_state)
    
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
        settings = {
            'blur_amount': self.blur_amount_var.get(),
            'noise_amount': self.noise_amount_var.get(),
            'color_variation': self.color_var_var.get(),
            'flood_fill_color': self.hex_to_rgb(self.flood_fill_var.get()),
            'flood_fill_opacity': self.flood_fill_opacity_var.get(),
            'hue_shift': self.hue_shift_var.get(),
            'tint_strength': self.tint_strength_var.get(),
            'enable_surface': self.enable_surface_var.get(),
            'model_path': self.model_path_var.get() if self.enable_surface_var.get() else None,
            'curvature_strength': self.curv_strength_var.get(),
            'enable_edge': self.enable_edge_var.get(),
            'edge_highlight': self.edge_highlight_var.get(),
            'edge_saturation': self.edge_saturation_var.get(),
            'edge_blend': self.edge_blend_var.get(),
            'enable_ao': self.enable_ao_var.get(),
            'ao_darken': self.ao_darken_var.get(),
            'ao_blend': self.ao_blend_var.get(),
            'edge_color': self.hex_to_rgb(self.edge_color_var),
            'ao_color': self.hex_to_rgb(self.ao_color_var),
            'pixel_width': self.pixel_width_var.get(),
            'resample_mode': self.resample_var.get(),
            'enable_greedy_expand': self.enable_greedy_expand_var.get(),
            'greedy_iterations': self.greedy_iterations_var.get(),
            'quantize_method': self.quantize_method_var.get(),
            'bits_per_channel': self.bits_per_channel_var.get(),
            'palette_colors': self.palette_colors_var.get(),
            'dither_mode': self.dither_mode_var.get(),
            'bayer_size': self.bayer_size_var.get(),
            'dither_strength': self.dither_strength_var.get(),
            'is_normal_map': self.is_normal_map_var.get()
        }
        
        # Add baked map paths if project is saved and maps exist
        if self.current_project_path:
            project_dir = os.path.dirname(self.current_project_path)
            
            edge_map_path = os.path.join(project_dir, "edge.png")
            if os.path.exists(edge_map_path):
                settings['baked_map_path'] = edge_map_path
            
            ao_map_path = os.path.join(project_dir, "ao.png")
            if os.path.exists(ao_map_path):
                settings['ao_map_path'] = ao_map_path
        
        return settings
    
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
    
    # === PROJECT MANAGEMENT ===
    
    def create_scrollable_tab(self):
        """Create a scrollable canvas container for a tab"""
        # Container frame
        container = ttk.Frame(self.notebook)
        
        # Canvas with scrollbar
        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding="10")
        
        # Update scroll region only after idle to reduce lag
        def update_scrollregion():
            canvas.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.after_idle(update_scrollregion))
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel only when mouse enters this canvas (not bind_all)
        def on_enter(event):
            def on_mousewheel(e):
                canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
            canvas.bind("<MouseWheel>", on_mousewheel)
        
        def on_leave(event):
            canvas.unbind("<MouseWheel>")
        
        canvas.bind("<Enter>", on_enter)
        canvas.bind("<Leave>", on_leave)
        
        return container, scrollable_frame
    
    def setup_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        
        file_menu.add_command(label="New Project", command=self.new_project, accelerator="Cmd+N")
        file_menu.add_separator()
        file_menu.add_command(label="Open Project...", command=self.load_project, accelerator="Cmd+O")
        
        # Recent projects submenu
        self.recent_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Open Recent", menu=self.recent_menu)
        self.update_recent_menu()
        
        file_menu.add_separator()
        file_menu.add_command(label="Save Project", command=self.save_project, accelerator="Cmd+S")
        file_menu.add_command(label="Save Project As...", command=self.save_project_as, accelerator="Cmd+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.quit, accelerator="Cmd+Q")
        
        # Keyboard shortcuts
        self.root.bind('<Command-n>', lambda e: self.new_project())
        self.root.bind('<Command-o>', lambda e: self.load_project())
        self.root.bind('<Command-s>', lambda e: self.save_project())
        self.root.bind('<Command-Shift-s>', lambda e: self.save_project_as())
        self.root.bind('<Command-q>', lambda e: self.root.quit())
    
    def new_project(self):
        """Create a new project (reset all settings)"""
        if self.project_modified:
            result = messagebox.askyesnocancel("Unsaved Changes", 
                                              "Save current project before creating new?")
            if result is None:  # Cancel
                return
            elif result:  # Yes
                if not self.save_project():
                    return
        
        # Reset all settings to defaults
        self.input_file_var.set('')
        self.input_folder_var.set('')
        self.output_var.set('')
        self.blur_amount_var.set(0.0)
        self.noise_amount_var.set(0.0)
        self.color_var_var.set(0.0)
        self.enable_surface_var.set(False)
        self.model_path_var.set('')
        self.curv_strength_var.set(5.0)
        self.edge_highlight_var.set(0.3)
        self.ao_darken_var.set(0.5)
        self.edge_color_var = '#FF8800'
        self.ao_color_var = '#321E14'
        self.pixel_width_var.set(64)
        self.resample_var.set('nearest')
        self.quantize_method_var.set('bit_depth')
        self.bits_per_channel_var.set(5)
        self.palette_colors_var.set(32)
        self.dither_mode_var.set('bayer')
        self.bayer_size_var.set('4x4')
        self.dither_strength_var.set(0.5)
        
        self.current_project_path = None
        self.project_modified = False
        self.update_title()
        self.on_surface_toggle()
    
    def save_project(self):
        """Save current project (use existing path or prompt)"""
        if self.current_project_path:
            settings = self.project_manager.get_current_settings(self)
            if self.project_manager.save_project(self.current_project_path, settings):
                self.project_modified = False
                self.update_title()
                self.check_baked_map_status()
                self.show_success(f"Project saved: {os.path.basename(self.current_project_path)}")
                return True
            else:
                self.show_error("Failed to save project")
                return False
        else:
            return self.save_project_as()
    
    def save_project_as(self):
        """Save project with new filename in folder structure"""
        project_name = simpledialog.askstring("Project Name", 
                                              "Enter project name:",
                                              initialvalue="MyProject")
        if not project_name:
            return False
        
        # Ask for parent directory
        parent_dir = filedialog.askdirectory(title="Select Location for Project Folder")
        if not parent_dir:
            return False
        
        # Create project folder
        project_folder = os.path.join(parent_dir, project_name)
        try:
            os.makedirs(project_folder, exist_ok=True)
        except Exception as e:
            self.show_error(f"Failed to create project folder: {e}")
            return False
        
        # Save project file inside folder
        filepath = os.path.join(project_folder, f"{project_name}.pixproj")
        
        settings = self.project_manager.get_current_settings(self)
        if self.project_manager.save_project(filepath, settings):
            self.current_project_path = filepath
            self.project_modified = False
            self.update_title()
            self.check_baked_map_status()
            self.add_to_recent_projects(filepath)
            self.show_success(f"Project saved in folder: {project_name}")
            return True
        else:
            self.show_error("Failed to save project")
            return False
    
    def load_project(self):
        """Load project from file"""
        if self.project_modified:
            result = messagebox.askyesnocancel("Unsaved Changes", 
                                              "Save current project before loading?")
            if result is None:  # Cancel
                return
            elif result:  # Yes
                if not self.save_project():
                    return
        
        filepath = filedialog.askopenfilename(
            title="Open Project",
            filetypes=[("Pixelator Project", "*.pixproj"), ("All Files", "*.*")]
        )
        
        if filepath:
            settings = self.project_manager.load_project(filepath)
            if settings:
                if self.project_manager.apply_settings_to_gui(self, settings):
                    self.current_project_path = filepath
                    self.project_modified = False
                    self.add_to_recent_projects(filepath)
                    self.update_title()
                    self.check_baked_map_status()
                    self.show_success(f"Project loaded: {os.path.basename(filepath)}")
                else:
                    self.show_error("Failed to apply project settings")
            else:
                self.show_error("Failed to load project file")
    
    def load_recent_project(self, filepath):
        """Load a project from the recent projects list"""
        if not os.path.exists(filepath):
            messagebox.showerror("File Not Found", f"Project file not found:\n{filepath}")
            self.remove_from_recent_projects(filepath)
            return
        
        if self.project_modified:
            result = messagebox.askyesnocancel("Unsaved Changes", 
                                              "Save current project before loading?")
            if result is None:  # Cancel
                return
            elif result:  # Yes
                if not self.save_project():
                    return
        
        settings = self.project_manager.load_project(filepath)
        if settings:
            if self.project_manager.apply_settings_to_gui(self, settings):
                self.current_project_path = filepath
                self.project_modified = False
                self.add_to_recent_projects(filepath)
                self.update_title()
                self.check_baked_map_status()
                self.show_success(f"Project loaded: {os.path.basename(filepath)}")
            else:
                self.show_error("Failed to apply project settings")
        else:
            self.show_error("Failed to load project file")
    
    def load_recent_projects(self):
        """Load recent projects list from config file"""
        config_dir = Path.home() / '.texture_pixelator'
        config_file = config_dir / 'recent_projects.txt'
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    projects = [line.strip() for line in f.readlines() if line.strip()]
                    # Filter out non-existent files
                    return [p for p in projects if os.path.exists(p)][:10]
            except Exception as e:
                print(f"Failed to load recent projects: {e}")
        return []
    
    def save_recent_projects(self):
        """Save recent projects list to config file"""
        config_dir = Path.home() / '.texture_pixelator'
        config_dir.mkdir(exist_ok=True)
        config_file = config_dir / 'recent_projects.txt'
        
        try:
            with open(config_file, 'w') as f:
                for project in self.recent_projects:
                    f.write(project + '\n')
        except Exception as e:
            print(f"Failed to save recent projects: {e}")
    
    def add_to_recent_projects(self, filepath):
        """Add a project to the recent projects list"""
        filepath = os.path.abspath(filepath)
        
        # Remove if already in list
        if filepath in self.recent_projects:
            self.recent_projects.remove(filepath)
        
        # Add to front
        self.recent_projects.insert(0, filepath)
        
        # Keep only last 10
        self.recent_projects = self.recent_projects[:10]
        
        self.save_recent_projects()
        self.update_recent_menu()
    
    def remove_from_recent_projects(self, filepath):
        """Remove a project from recent projects list"""
        if filepath in self.recent_projects:
            self.recent_projects.remove(filepath)
            self.save_recent_projects()
            self.update_recent_menu()
    
    def clear_recent_projects(self):
        """Clear all recent projects"""
        self.recent_projects = []
        self.save_recent_projects()
        self.update_recent_menu()
    
    def update_recent_menu(self):
        """Update the recent projects submenu"""
        # Clear existing items
        self.recent_menu.delete(0, tk.END)
        
        if self.recent_projects:
            for i, filepath in enumerate(self.recent_projects):
                display_name = os.path.basename(filepath)
                project_dir = os.path.dirname(filepath)
                
                # Add menu item with keyboard shortcut for first 9
                if i < 9:
                    self.recent_menu.add_command(
                        label=f"{display_name}  ({project_dir})",
                        command=lambda p=filepath: self.load_recent_project(p),
                        accelerator=f"Cmd+{i+1}"
                    )
                    # Bind keyboard shortcut
                    self.root.bind(f'<Command-{i+1}>', lambda e, p=filepath: self.load_recent_project(p))
                else:
                    self.recent_menu.add_command(
                        label=f"{display_name}  ({project_dir})",
                        command=lambda p=filepath: self.load_recent_project(p)
                    )
            
            self.recent_menu.add_separator()
            self.recent_menu.add_command(label="Clear Recent", command=self.clear_recent_projects)
        else:
            self.recent_menu.add_command(label="(No Recent Projects)", state="disabled")
    
    def update_title(self):
        """Update window title with project name and modified status"""
        title = "Texture Pixelator"
        if self.current_project_path:
            title += f" - {os.path.basename(self.current_project_path)}"
        if self.project_modified:
            title += " *"
        self.root.title(title)
    
    def mark_modified(self, *args):
        """Mark project as modified"""
        self.project_modified = True
        self.update_title()
    
    def pick_edge_color(self):
        """Open color picker for edge color"""
        color = colorchooser.askcolor(
            title="Choose Edge Color",
            initialcolor=self.edge_color_var
        )
        if color[1]:  # color[1] is the hex string
            self.edge_color_var = color[1]
            self.edge_color_canvas.itemconfig(self.edge_color_circle, fill=color[1])
            self.mark_modified()
    
    def pick_ao_color(self):
        """Open color picker for AO color"""
        color = colorchooser.askcolor(
            title="Choose AO Color",
            initialcolor=self.ao_color_var
        )
        if color[1]:  # color[1] is the hex string
            self.ao_color_var = color[1]
            self.ao_color_canvas.itemconfig(self.ao_color_circle, fill=color[1])
            self.mark_modified()
    
    def pick_flood_fill_color(self):
        """Open color picker for flood fill color"""
        color = colorchooser.askcolor(
            title="Choose Flood Fill Color",
            initialcolor=self.flood_fill_var.get()
        )
        if color[1]:  # color[1] is the hex string
            self.flood_fill_var.set(color[1])
            self.flood_fill_canvas.itemconfig(self.flood_fill_circle, fill=color[1])
            self.mark_modified()
    
    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def center_placeholder_text(self):
        """Center the placeholder text in the canvas"""
        if self.preview_text_id and not self.preview_image:
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:  # Canvas is rendered
                self.preview_canvas.coords(self.preview_text_id, canvas_width // 2, canvas_height // 2)
    
    def check_baked_map_status(self):
        """Check if baked edge and AO maps exist and update UI"""
        if not self.current_project_path:
            self.edge_status_label.config(text="Edge: Save project first", foreground="gray")
            self.ao_status_label.config(text="AO: Save project first", foreground="gray")
            return
        
        project_dir = os.path.dirname(self.current_project_path)
        edge_path = os.path.join(project_dir, "edge.png")
        ao_path = os.path.join(project_dir, "ao.png")
        
        # Check edge map
        if os.path.exists(edge_path):
            size_mb = os.path.getsize(edge_path) / (1024 * 1024)
            self.edge_status_label.config(
                text=f"✓ Edge: edge.png ({size_mb:.1f} MB)",
                foreground="green"
            )
            # Enable edge controls
            self.edge_highlight_scale.config(state="normal")
            self.edge_blend_scale.config(state="normal")
        else:
            self.edge_status_label.config(
                text="Edge: Not found (required)",
                foreground="orange"
            )
            self.edge_highlight_scale.config(state="disabled")
            self.edge_blend_scale.config(state="disabled")
        
        # Check AO map
        if os.path.exists(ao_path):
            size_mb = os.path.getsize(ao_path) / (1024 * 1024)
            self.ao_status_label.config(
                text=f"✓ AO: ao.png ({size_mb:.1f} MB)",
                foreground="green"
            )
            # Enable AO controls
            self.ao_darken_scale.config(state="normal")
            self.ao_blend_scale.config(state="normal")
        else:
            self.ao_status_label.config(
                text="AO: Not found (optional)",
                foreground="gray"
            )
            self.ao_darken_scale.config(state="disabled")
            self.ao_blend_scale.config(state="disabled")
    
    def bake_edge_maps(self):
        """Bake edge map from 3D model"""
        # Check if project is saved
        if not self.current_project_path:
            result = messagebox.askyesno("Save Project First", 
                                        "Baked maps are saved in your project folder.\n\n"
                                        "Would you like to save your project now?")
            if result:
                if not self.save_project_as():
                    return
            else:
                return
        
        model_path = self.model_path_var.get()
        if not model_path or not os.path.exists(model_path):
            self.show_error("Please select a valid 3D model file first")
            return
        
        # Output path in project folder
        project_dir = os.path.dirname(self.current_project_path)
        output_path = os.path.join(project_dir, "edge.png")
        
        # Get resolution from UI
        resolution = self.bake_resolution_var.get()
        
        self.status_var.set("Baking edge map...")
        self.root.update()
        
        # Bake the edge map
        strength = self.curv_strength_var.get()
        success = self.pixelator.surface_baker.bake_curvature_map(
            model_path, output_path, resolution, strength
        )
        
        self.status_var.set("Ready")
        
        if success:
            self.check_baked_map_status()
            self.show_success(f"Edge map saved to project folder\n"
                            f"It will be automatically used for faster processing.")
        else:
            self.show_error("Failed to bake edge map")
    
    def bake_ao_map(self):
        """Bake AO map from 3D model"""
        # Check if project is saved
        if not self.current_project_path:
            result = messagebox.askyesno("Save Project First", 
                                        "Baked maps are saved in your project folder.\n\n"
                                        "Would you like to save your project now?")
            if result:
                if not self.save_project_as():
                    return
            else:
                return
        
        model_path = self.model_path_var.get()
        if not model_path or not os.path.exists(model_path):
            self.show_error("Please select a valid 3D model file first")
            return
        
        # Output path in project folder
        project_dir = os.path.dirname(self.current_project_path)
        output_path = os.path.join(project_dir, "ao.png")
        
        # Get resolution from UI
        resolution = self.bake_resolution_var.get()
        
        self.status_var.set("Baking AO map...")
        self.root.update()
        
        # Bake the AO map
        success = self.pixelator.surface_baker.bake_ao_map(
            model_path, output_path, resolution, samples=32, distance=0.5
        )
        
        self.status_var.set("Ready")
        
        if success:
            self.check_baked_map_status()
            self.show_success(f"AO map saved to project folder\n"
                            f"It will be automatically used for crevice effects.")
        else:
            self.show_error("Failed to bake AO map")


def main():
    root = tk.Tk()
    app = PixelatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
