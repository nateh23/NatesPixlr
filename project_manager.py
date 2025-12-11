"""
Project Manager - Save and load project settings
"""
import json
import os
from typing import Dict, Any, Optional


class ProjectManager:
    """Handles saving and loading project settings to/from JSON files"""
    
    @staticmethod
    def save_project(filepath: str, settings: Dict[str, Any]) -> bool:
        """
        Save project settings to a JSON file
        
        Args:
            filepath: Path to save the .pixproj file
            settings: Dictionary containing all project settings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure .pixproj extension
            if not filepath.endswith('.pixproj'):
                filepath += '.pixproj'
            
            # Save to JSON with pretty formatting
            with open(filepath, 'w') as f:
                json.dump(settings, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving project: {e}")
            return False
    
    @staticmethod
    def load_project(filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load project settings from a JSON file
        
        Args:
            filepath: Path to the .pixproj file
            
        Returns:
            Dictionary containing project settings, or None if failed
        """
        try:
            with open(filepath, 'r') as f:
                settings = json.load(f)
            return settings
        except Exception as e:
            print(f"Error loading project: {e}")
            return None
    
    @staticmethod
    def get_current_settings(gui) -> Dict[str, Any]:
        """
        Extract current settings from GUI
        
        Args:
            gui: PixelatorGUI instance
            
        Returns:
            Dictionary containing all current settings
        """
        settings = {
            # File paths
            'input_file': gui.input_file_var.get(),
            'input_folder': gui.input_folder_var.get(),
            'output': gui.output_var.get(),
            
            # Preprocessing
            'blur_amount': gui.blur_amount_var.get(),
            'noise_amount': gui.noise_amount_var.get(),
            'color_variation': gui.color_var_var.get(),
            
            # Surface Effects
            'enable_surface': gui.enable_surface_var.get(),
            'model_path': gui.model_path_var.get(),
            'curvature_strength': gui.curv_strength_var.get(),
            'edge_highlight': gui.edge_highlight_var.get(),
            'ao_darken': gui.ao_darken_var.get(),
            'edge_color': gui.edge_color_var if hasattr(gui, 'edge_color_var') else '#FF8800',
            'ao_color': gui.ao_color_var if hasattr(gui, 'ao_color_var') else '#321E14',
            
            # Pixelation
            'pixel_width': gui.pixel_width_var.get(),
            'resample_mode': gui.resample_var.get(),
            'quantize_method': gui.quantize_method_var.get(),
            'bits_per_channel': gui.bits_per_channel_var.get(),
            'palette_colors': gui.palette_colors_var.get(),
            'dither_mode': gui.dither_mode_var.get(),
            'bayer_size': gui.bayer_size_var.get(),
            'dither_strength': gui.dither_strength_var.get(),
            'add_suffix': gui.suffix_var.get() if hasattr(gui, 'suffix_var') else False,
        }
        
        return settings
    
    @staticmethod
    def apply_settings_to_gui(gui, settings: Dict[str, Any]) -> bool:
        """
        Apply loaded settings to GUI
        
        Args:
            gui: PixelatorGUI instance
            settings: Dictionary containing project settings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # File paths
            if 'input_file' in settings:
                gui.input_file_var.set(settings['input_file'])
            if 'input_folder' in settings:
                gui.input_folder_var.set(settings['input_folder'])
            if 'output' in settings:
                gui.output_var.set(settings['output'])
            
            # Preprocessing
            if 'blur_amount' in settings:
                gui.blur_amount_var.set(settings['blur_amount'])
            if 'noise_amount' in settings:
                gui.noise_amount_var.set(settings['noise_amount'])
            if 'color_variation' in settings:
                gui.color_var_var.set(settings['color_variation'])
            
            # Surface Effects
            if 'enable_surface' in settings:
                gui.enable_surface_var.set(settings['enable_surface'])
                gui.on_surface_toggle()  # Update UI state
            if 'model_path' in settings:
                gui.model_path_var.set(settings['model_path'])
            if 'curvature_strength' in settings:
                gui.curv_strength_var.set(settings['curvature_strength'])
            if 'edge_highlight' in settings:
                gui.edge_highlight_var.set(settings['edge_highlight'])
            if 'ao_darken' in settings:
                gui.ao_darken_var.set(settings['ao_darken'])
            elif 'crevice_darken' in settings:  # Backward compatibility
                gui.ao_darken_var.set(settings['crevice_darken'])
            if 'edge_color' in settings and hasattr(gui, 'edge_color_var'):
                gui.edge_color_var = settings['edge_color']
                if hasattr(gui, 'edge_color_canvas'):
                    gui.edge_color_canvas.itemconfig(gui.edge_color_circle, fill=settings['edge_color'])
            if 'ao_color' in settings and hasattr(gui, 'ao_color_var'):
                gui.ao_color_var = settings['ao_color']
                if hasattr(gui, 'ao_color_canvas'):
                    gui.ao_color_canvas.itemconfig(gui.ao_color_circle, fill=settings['ao_color'])
            elif 'crevice_color' in settings and hasattr(gui, 'ao_color_var'):  # Backward compatibility
                gui.ao_color_var = settings['crevice_color']
                if hasattr(gui, 'ao_color_canvas'):
                    gui.ao_color_canvas.itemconfig(gui.ao_color_circle, fill=settings['crevice_color'])
            
            # Pixelation
            if 'pixel_width' in settings:
                gui.pixel_width_var.set(settings['pixel_width'])
            if 'resample_mode' in settings:
                gui.resample_var.set(settings['resample_mode'])
            if 'quantize_method' in settings:
                gui.quantize_method_var.set(settings['quantize_method'])
                gui.on_quantize_method_change()  # Update UI state
            if 'bits_per_channel' in settings:
                gui.bits_per_channel_var.set(settings['bits_per_channel'])
            if 'palette_colors' in settings:
                gui.palette_colors_var.set(settings['palette_colors'])
            if 'dither_mode' in settings:
                gui.dither_mode_var.set(settings['dither_mode'])
                gui.on_dither_mode_change()  # Update UI state
            if 'bayer_size' in settings:
                gui.bayer_size_var.set(settings['bayer_size'])
            if 'dither_strength' in settings:
                gui.dither_strength_var.set(settings['dither_strength'])
            if 'add_suffix' in settings and hasattr(gui, 'suffix_var'):
                gui.suffix_var.set(settings['add_suffix'])
            
            return True
        except Exception as e:
            print(f"Error applying settings: {e}")
            return False
