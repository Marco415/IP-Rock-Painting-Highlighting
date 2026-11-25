
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from pathlib import Path

import cv2
import numpy as np

from PIL import Image, ImageTk

from scipy import ndimage
from skimage import color, morphology, transform


# ============================================================
# IMAGE SEGMENTATION AND COLOUR TOOL
# ============================================================

class ImageSegmentationTool:

    def __init__(self, root):

        self.root = root

        self.root.title("Image Segmentation and Colour Tool")
        self.root.geometry("1300x900")
        self.root.minsize(1100, 850)

        # ----------------------------------------------------
        # VARIABLES
        # ----------------------------------------------------

        self.original_image = None
        self.processed_image = None
        self.colour_processed_image = None

        self.segmentation_mask = None

        self.segmentation_available = False
        self.mask_display_mode = False

        # Saturation
        self.saturation_value = 1.0
        self.saturation_target = 1

        # Smoothing
        self.smoothing_value = 2

        # Background display mode
        # 0 = Original
        # 1 = White
        # 2 = Black
        self.background_mode = 0

        # Subject style
        # 1 = Colour
        # 2 = Two-Tone
        self.subject_style = 1

        # Image display references
        self.original_photo = None
        self.result_photo = None

        # ----------------------------------------------------
        # BUILD GUI
        # ----------------------------------------------------

        self.create_gui()

        # Load initial image
        self.root.after(200, self.select_new_image)

    # ========================================================
    # GUI CREATION
    # ========================================================

    def create_gui(self):

        # Main image area
        image_frame = ttk.Frame(self.root)

        image_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=20,
            pady=15
        )

        # Original image panel
        original_frame = ttk.LabelFrame(
            image_frame,
            text="Original Image"
        )

        original_frame.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=10
        )

        self.original_canvas = tk.Label(
            original_frame,
            background="black"
        )

        self.original_canvas.pack(
            fill=tk.BOTH,
            expand=True,
            padx=5,
            pady=5
        )

        # Processed image panel
        result_frame = ttk.LabelFrame(
            image_frame,
            text="Processed Image"
        )

        result_frame.pack(
            side=tk.RIGHT,
            fill=tk.BOTH,
            expand=True,
            padx=10
        )

        self.result_canvas = tk.Label(
            result_frame,
            background="black"
        )

        self.result_canvas.pack(
            fill=tk.BOTH,
            expand=True,
            padx=5,
            pady=5
        )

        # ----------------------------------------------------
        # SATURATION CONTROLS
        # ----------------------------------------------------

        saturation_frame = ttk.Frame(self.root)

        saturation_frame.pack(
            fill=tk.X,
            padx=30,
            pady=5
        )

        ttk.Label(
            saturation_frame,
            text="Saturation"
        ).pack(side=tk.LEFT)

        self.saturation_slider = tk.Scale(
            saturation_frame,
            from_=0,
            to=10,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            length=250,
            command=self.update_saturation_value
        )

        self.saturation_slider.set(1.0)

        self.saturation_slider.pack(
            side=tk.LEFT,
            padx=10
        )

        self.saturation_label = ttk.Label(
            saturation_frame,
            text="1.00",
            width=6
        )

        self.saturation_label.pack(side=tk.LEFT)

        ttk.Label(
            saturation_frame,
            text="Apply to"
        ).pack(
            side=tk.LEFT,
            padx=(20, 5)
        )

        self.saturation_target_dropdown = ttk.Combobox(
            saturation_frame,
            values=[
                "Subject Only",
                "Whole Image"
            ],
            state="readonly",
            width=18
        )

        self.saturation_target_dropdown.set("Subject Only")

        self.saturation_target_dropdown.pack(
            side=tk.LEFT
        )

        self.saturation_target_dropdown.bind(
            "<<ComboboxSelected>>",
            self.update_saturation_target
        )

        # ----------------------------------------------------
        # SMOOTHING CONTROLS
        # ----------------------------------------------------

        smoothing_frame = ttk.Frame(self.root)

        smoothing_frame.pack(
            fill=tk.X,
            padx=30,
            pady=5
        )

        ttk.Label(
            smoothing_frame,
            text="Smoothing"
        ).pack(side=tk.LEFT)

        self.smoothing_slider = tk.Scale(
            smoothing_frame,
            from_=0,
            to=20,
            resolution=1,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.update_smoothing_value
        )

        self.smoothing_slider.set(2)

        self.smoothing_slider.pack(
            side=tk.LEFT,
            padx=10
        )

        self.smoothing_label = ttk.Label(
            smoothing_frame,
            text="2",
            width=5
        )

        self.smoothing_label.pack(side=tk.LEFT)

        # ----------------------------------------------------
        # BUTTON ROW 1
        # ----------------------------------------------------

        button_frame_1 = ttk.Frame(self.root)

        button_frame_1.pack(
            fill=tk.X,
            padx=30,
            pady=(15, 5)
        )

        buttons_1 = [
            ("Select Image", self.select_new_image),
            ("Segment Subject", self.segment_subject),
            ("Clean Mask", self.clean_mask_noise),
            ("Smooth", self.apply_smoothing),
            ("Show Mask", self.toggle_mask_display)
        ]

        for text, command in buttons_1:

            ttk.Button(
                button_frame_1,
                text=text,
                command=command
            ).pack(
                side=tk.LEFT,
                fill=tk.X,
                expand=True,
                padx=5,
                ipady=10
            )

        # ----------------------------------------------------
        # BUTTON ROW 2
        # ----------------------------------------------------

        button_frame_2 = ttk.Frame(self.root)

        button_frame_2.pack(
            fill=tk.X,
            padx=30,
            pady=5
        )

        ttk.Button(
            button_frame_2,
            text="Saturate Colours",
            command=self.apply_saturation
        ).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=5,
            ipady=10
        )

        # Background dropdown
        self.background_dropdown = ttk.Combobox(
            button_frame_2,
            values=[
                "Background: Original",
                "Background: White",
                "Background: Black"
            ],
            state="readonly"
        )

        self.background_dropdown.set(
            "Background: Original"
        )

        self.background_dropdown.pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=5,
            ipady=5
        )

        self.background_dropdown.bind(
            "<<ComboboxSelected>>",
            self.update_background_mode
        )

        # Subject style dropdown
        self.subject_style_dropdown = ttk.Combobox(
            button_frame_2,
            values=[
                "Subject: Colour",
                "Subject: Two-Tone"
            ],
            state="readonly"
        )

        self.subject_style_dropdown.set(
            "Subject: Colour"
        )

        self.subject_style_dropdown.pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=5,
            ipady=5
        )

        self.subject_style_dropdown.bind(
            "<<ComboboxSelected>>",
            self.update_subject_style
        )

        ttk.Button(
            button_frame_2,
            text="Reset",
            command=self.reset_image
        ).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=5,
            ipady=10
        )

        ttk.Button(
            button_frame_2,
            text="Save Image",
            command=self.save_image
        ).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=5,
            ipady=10
        )

        ttk.Button(
            button_frame_2,
            text="Close",
            command=self.close_application
        ).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=5,
            ipady=10
        )

        # ----------------------------------------------------
        # STATUS LABEL
        # ----------------------------------------------------

        self.status_label = ttk.Label(
            self.root,
            text="Ready",
            anchor=tk.CENTER,
            relief=tk.SUNKEN
        )

        self.status_label.pack(
            side=tk.BOTTOM,
            fill=tk.X,
            padx=30,
            pady=(5, 12),
            ipady=5
        )

    # ========================================================
    # STATUS
    # ========================================================

    def set_status(self, message):

        self.status_label.config(text=message)

        self.root.update_idletasks()

    # ========================================================
    # IMAGE LOADING
    # ========================================================

    @staticmethod
    def load_image(image_path):

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            raise ValueError("Unable to load image.")

        # Grayscale to RGB
        if image.ndim == 2:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )

        elif image.shape[2] == 4:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2RGB
            )

        else:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        # Equivalent to MATLAB im2uint8
        if image.dtype == np.uint8:
            return image.copy()

        if image.dtype == np.uint16:

            image = (
                image.astype(np.float64) / 65535.0
            )

        elif np.issubdtype(image.dtype, np.floating):

            image = image.astype(np.float64)

            if image.max() > 1:
                image = image / 255.0

        else:

            max_value = np.iinfo(image.dtype).max

            image = (
                image.astype(np.float64) / max_value
            )

        image = np.clip(
            image * 255,
            0,
            255
        )

        return np.round(image).astype(np.uint8)

    # ========================================================
    # SELECT NEW IMAGE
    # ========================================================

    def select_new_image(self):

        file_path = filedialog.askopenfilename(
            title="Select an image",
            initialdir=str(Path.home() / "Pictures"),
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"
                )
            ]
        )

        if not file_path:

            self.set_status("Selection cancelled")
            return

        try:

            new_image = self.load_image(file_path)

            self.original_image = new_image.copy()

            self.processed_image = new_image.copy()

            self.colour_processed_image = new_image.copy()

            self.segmentation_mask = None

            self.segmentation_available = False

            self.background_mode = 0

            self.subject_style = 1

            self.mask_display_mode = False

            # Reset controls
            self.saturation_slider.set(1)

            self.saturation_label.config(text="1.00")

            self.saturation_target_dropdown.set(
                "Subject Only"
            )

            self.smoothing_slider.set(3)

            self.smoothing_label.config(text="3")

            self.saturation_value = 1

            self.saturation_target = 1

            self.smoothing_value = 3

            self.background_dropdown.set(
                "Background: Original"
            )

            self.subject_style_dropdown.set(
                "Subject: Colour"
            )

            self.update_displays()

            self.set_status(
                f"Loaded: {Path(file_path).name}"
            )

        except Exception as error:

            messagebox.showerror(
                "Image Loading Error",
                str(error)
            )

    # ========================================================
    # DISPLAY IMAGES
    # ========================================================


    def display_image(self, image, canvas, attribute):

        if image is None:
            return

        # Fixed image display dimensions
        canvas_width = 560
        canvas_height = 480

        height, width = image.shape[:2]

        # Always calculate scale from original image dimensions
        scale = min(
            canvas_width / width,
            canvas_height / height
        )

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        # Resize directly from the original image
        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        )

        pil_image = Image.fromarray(resized)

        photo = ImageTk.PhotoImage(pil_image)

        canvas.config(
            image=photo,
            width=canvas_width,
            height=canvas_height
        )

        canvas.image = photo

        if attribute == "original":
            self.original_photo = photo
        else:
            self.result_photo = photo

    def update_displays(self):

        if self.original_image is None:
            return

        self.display_image(
            self.original_image,
            self.original_canvas,
            "original"
        )

        self.display_image(
            self.processed_image,
            self.result_canvas,
            "result"
        )

    # ========================================================
    # ADAPTIVE SEGMENTATION
    # ========================================================

    def segment_subject(self):

        if self.original_image is None:
            self.set_status("Load an image first")
            return

        self.set_status("Segmenting subject...")

        try:

            original = self.original_image

            orig_h, orig_w = original.shape[:2]

            # ------------------------------------------------
            # STEP 1 - DOWNSCALE FOR SPEED
            # ------------------------------------------------

            max_dim = 600

            scale_factor = min(
                1.0,
                max_dim / max(orig_h, orig_w)
            )

            if scale_factor < 1:

                new_width = max(
                    1,
                    int(orig_w * scale_factor)
                )

                new_height = max(
                    1,
                    int(orig_h * scale_factor)
                )

                img_small = cv2.resize(
                    original,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA
                )

            else:

                img_small = original.copy()

            img_double = (
                img_small.astype(np.float64) / 255.0
            )

            h, w = img_small.shape[:2]

            # ------------------------------------------------
            # STEP 2 - CHROMATIC SIGNAL EXTRACTION
            # ------------------------------------------------

            lab_img = color.rgb2lab(img_double)

            hsv_img = color.rgb2hsv(img_double)

            a_chan = lab_img[:, :, 1]

            b_chan = lab_img[:, :, 2]

            sat_chan = hsv_img[:, :, 1]

            # Enhance ochre/pigment signal
            pigment_signal = (
                (a_chan + 0.35 * b_chan)
                * (1 + 0.8 * sat_chan)
            )

            # Subtract background illumination trend
            sigma_bg = max(
                15,
                round(max(h, w) * 0.12)
            )

            bg_signal = ndimage.gaussian_filter(
                pigment_signal,
                sigma=sigma_bg
            )

            enhanced_signal = (
                pigment_signal - bg_signal
            )

            enhanced_signal = ndimage.median_filter(
                enhanced_signal,
                size=3
            )

            enhanced_signal = ndimage.gaussian_filter(
                enhanced_signal,
                sigma=1.1
            )

            # ------------------------------------------------
            # STEP 3 - ADAPTIVE SEED GENERATION
            # ------------------------------------------------

            mean_val = np.mean(enhanced_signal)

            std_val = np.std(
                enhanced_signal,
                ddof=1
            )

            seed_mask = (
                enhanced_signal > mean_val + 1.8 * std_val
            )

            allow_mask = (
                enhanced_signal > mean_val + 0.7 * std_val
            )

            seed_mask = seed_mask & allow_mask

            # Binary reconstruction using 8-connectivity
            structure = np.ones(
                (3, 3),
                dtype=bool
            )

            reconstructed_mask = ndimage.binary_propagation(
                seed_mask,
                structure=structure,
                mask=allow_mask
            )

            # ------------------------------------------------
            # STEP 4 - NEIGHBOURHOOD DENSITY FILTER
            # ------------------------------------------------

            local_density = ndimage.uniform_filter(
                reconstructed_mask.astype(np.float64),
                size=11
            )

            dense_mask = (
                reconstructed_mask
                & (local_density > 0.24)
            )

            # ------------------------------------------------
            # STEP 5 - INITIAL AREA FILTER
            # ------------------------------------------------

            min_area = max(
                40,
                round(h * w * 0.001)
            )

            clean_mask = morphology.remove_small_objects(
                dense_mask,
                min_size=min_area,
                connectivity=2
            )

            clean_mask = morphology.binary_closing(
                clean_mask,
                morphology.disk(1)
            )

            # ------------------------------------------------
            # STEP 6 - RESCALE MASK
            # ------------------------------------------------

            if scale_factor < 1:

                segmentation_mask = cv2.resize(
                    clean_mask.astype(np.uint8),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            else:

                segmentation_mask = clean_mask

            # ------------------------------------------------
            # STEP 7 - STORE STATE AND RENDER
            # ------------------------------------------------

            self.segmentation_mask = (
                segmentation_mask.astype(bool)
            )

            self.segmentation_available = True

            self.background_mode = 0

            self.subject_style = 1

            self.colour_processed_image = (
                self.original_image.copy()
            )

            self.background_dropdown.set(
                "Background: Original"
            )

            self.subject_style_dropdown.set(
                "Subject: Colour"
            )

            self.mask_display_mode = True

            self.show_mask_overlay()

            self.set_status("Segmentation complete.")

        except Exception as error:

            self.set_status("Segmentation failed")

            messagebox.showerror(
                "Segmentation Error",
                str(error)
            )

    # ========================================================
    # ADAPTIVE NOISE CLEANING
    # ========================================================

    def clean_mask_noise(self):

        if (
            not self.segmentation_available
            or self.segmentation_mask is None
        ):

            messagebox.showwarning(
                "No Mask Found",
                "Please segment the image before cleaning."
            )

            return

        self.set_status(
            "Cleaning mask background noise..."
        )

        mask = self.segmentation_mask.copy()

        h, w = mask.shape

        # ----------------------------------------------------
        # STEP 1 - CONNECTED COMPONENT ANALYSIS
        # ----------------------------------------------------

        labels, num_objects = ndimage.label(
            mask,
            structure=np.ones((3, 3))
        )

        if num_objects == 0:

            self.set_status("Mask is empty.")
            return

        objects = ndimage.find_objects(labels)

        stats = []

        for index, slices in enumerate(objects, start=1):

            if slices is None:
                continue

            component = labels[slices] == index

            area = int(np.sum(component))

            rows, cols = np.where(component)

            y_start, x_start = (
                slices[0].start,
                slices[1].start
            )

            centroid_x = float(
                np.mean(cols) + x_start
            )

            centroid_y = float(
                np.mean(rows) + y_start
            )

            pixel_indices = np.flatnonzero(
                labels.ravel() == index
            )

            stats.append({
                "label": index,
                "area": area,
                "centroid": (
                    centroid_x,
                    centroid_y
                ),
                "pixels": pixel_indices
            })

        areas = np.array(
            [item["area"] for item in stats]
        )

        sort_indices = np.argsort(areas)[::-1]

        largest_area = areas[sort_indices[0]]

        # ----------------------------------------------------
        # STEP 2 - IDENTIFY CENTRAL ANCHORS
        # ----------------------------------------------------

        anchor_indices = []

        for k, item in enumerate(stats):

            cx, cy = item["centroid"]

            if (
                item["area"] >= largest_area * 0.12
                and cy > h * 0.12
                and cy < h * 0.88
                and cx > w * 0.15
                and cx < w * 0.85
            ):

                anchor_indices.append(k)

        # Fallback: largest component
        if len(anchor_indices) == 0:

            anchor_indices = [
                int(sort_indices[0])
            ]

        anchor_centroids = np.array([
            stats[k]["centroid"]
            for k in anchor_indices
        ])

        new_mask = np.zeros(
            (h, w),
            dtype=bool
        )

        # ----------------------------------------------------
        # STEP 3 - DISTANCE AND ALIGNMENT FILTER
        # ----------------------------------------------------

        for k, item in enumerate(stats):

            comp_area = item["area"]

            cx, cy = item["centroid"]

            distances = np.sqrt(
                (anchor_centroids[:, 0] - cx) ** 2
                + (anchor_centroids[:, 1] - cy) ** 2
            )

            min_dist = np.min(distances)

            dx = np.abs(
                anchor_centroids[:, 0] - cx
            )

            min_dx = np.min(dx)

            is_anchor = k in anchor_indices

            # Near central anchors
            is_near_subject = (
                comp_area >= largest_area * 0.05
                and min_dist < w * 0.32
            )

            # Vertical extensions
            is_vertical_limb = (
                comp_area >= 25
                and min_dx < w * 0.10
                and min_dist < h * 0.45
            )

            if (
                is_anchor
                or is_near_subject
                or is_vertical_limb
            ):

                new_mask.ravel()[
                    item["pixels"]
                ] = True

        # ----------------------------------------------------
        # STEP 4 - MARGIN AND EDGE PURGING
        # ----------------------------------------------------

        labels_new, num_new = ndimage.label(
            new_mask,
            structure=np.ones((3, 3))
        )

        for index in range(1, num_new + 1):

            pixels = np.flatnonzero(
                labels_new.ravel() == index
            )

            rows, cols = np.unravel_index(
                pixels,
                (h, w)
            )

            touching_top = np.min(rows) < h * 0.08

            touching_bottom = np.max(rows) > h * 0.92

            touching_sides = (
                np.min(cols) < w * 0.03
                or np.max(cols) > w * 0.97
            )

            # Purge top/edge artifacts
            if (
                (touching_top or touching_sides)
                and np.max(rows) < h * 0.25
            ):

                new_mask.ravel()[pixels] = False

            elif (
                touching_bottom
                and np.min(rows) > h * 0.85
                and len(pixels) < largest_area * 0.3
            ):

                new_mask.ravel()[pixels] = False

        # ----------------------------------------------------
        # STEP 5 - FINAL CLEANUP
        # ----------------------------------------------------

        new_mask = morphology.remove_small_objects(
            new_mask,
            min_size=50,
            connectivity=2
        )

        self.segmentation_mask = new_mask.astype(bool)

        self.mask_display_mode = True

        self.show_mask_overlay()

        self.set_status(
            "Clean complete. Peripheral background noise removed."
        )

    # ========================================================
    # SHOW / HIDE MASK
    # ========================================================

    def toggle_mask_display(self):

        if not self.segmentation_available:

            self.set_status(
                "Segment the subject first"
            )

            return

        self.mask_display_mode = (
            not self.mask_display_mode
        )

        if self.mask_display_mode:

            self.show_mask_overlay()

            self.set_status(
                "Green = selected subject"
            )

        else:

            self.refresh_display()

            self.set_status(
                "Showing processed image"
            )

    # ========================================================
    # MASK OVERLAY
    # ========================================================

    def show_mask_overlay(self):

        if self.segmentation_mask is None:
            return

        base = (
            self.original_image.astype(np.float64) / 255.0
        )

        overlay = base.copy()

        background_pixels = ~self.segmentation_mask

        # Darken background
        overlay[background_pixels] *= 0.35

        # Make selected subject green
        overlay[:, :, 0][self.segmentation_mask] *= 0.25

        overlay[:, :, 1][self.segmentation_mask] = np.minimum(
            overlay[:, :, 1][self.segmentation_mask] * 0.25
            + 0.75,
            1
        )

        overlay[:, :, 2][self.segmentation_mask] *= 0.25

        overlay = np.clip(
            overlay * 255,
            0,
            255
        ).astype(np.uint8)

        self.display_image(
            overlay,
            self.result_canvas,
            "result"
        )

    # ========================================================
    # SATURATION VALUE
    # ========================================================

    def update_saturation_value(self, value):

        self.saturation_value = float(value)

        self.saturation_label.config(
            text=f"{self.saturation_value:.2f}"
        )

        self.set_status(
            "Click Saturate Colours to apply"
        )

    # ========================================================
    # SATURATION TARGET
    # ========================================================

    def update_saturation_target(self, event=None):

        value = self.saturation_target_dropdown.get()

        if value == "Subject Only":

            self.saturation_target = 1

        else:

            self.saturation_target = 2

        self.set_status(
            "Click Saturate Colours to apply"
        )

    # ========================================================
    # APPLY SATURATION
    # ========================================================

    def apply_saturation(self):

        if self.original_image is None:
            return

        self.set_status("Applying saturation...")

        # Convert original image to [0, 1]
        rgb_image = (
            self.original_image.astype(np.float64) / 255.0
        )

        hsv_image = color.rgb2hsv(rgb_image)

        # Subject only
        if self.saturation_target == 1:

            if not self.segmentation_available:

                self.set_status(
                    "Segment the subject first"
                )

                return

            saturation = hsv_image[:, :, 1]

            saturation[self.segmentation_mask] = np.minimum(
                saturation[self.segmentation_mask]
                * self.saturation_value,
                1
            )

            hsv_image[:, :, 1] = saturation

        # Whole image
        else:

            hsv_image[:, :, 1] = np.minimum(
                hsv_image[:, :, 1]
                * self.saturation_value,
                1
            )

        # Convert back to RGB uint8
        rgb_result = color.hsv2rgb(hsv_image)

        self.colour_processed_image = np.round(
            np.clip(rgb_result, 0, 1) * 255
        ).astype(np.uint8)

        self.refresh_display()

        if self.saturation_target == 1:

            self.set_status(
                "Saturation applied to subject"
            )

        else:

            self.set_status(
                "Saturation applied to whole image"
            )

    # ========================================================
    # SMOOTHING VALUE
    # ========================================================

    def update_smoothing_value(self, value):

        self.smoothing_value = round(float(value))

        self.smoothing_label.config(
            text=str(self.smoothing_value)
        )

        self.set_status(
            "Click Smooth to apply"
        )

    # ========================================================
    # APPLY SMOOTHING
    # ========================================================

    def apply_smoothing(self):

        if not self.segmentation_available:

            self.set_status(
                "Segment the subject first"
            )

            return

        if self.smoothing_value == 0:

            self.set_status(
                "Smoothing is set to 0"
            )

            return

        self.set_status(
            "Smoothing subject mask..."
        )

        mask = self.segmentation_mask.copy()

        # Morphological closing
        disk = morphology.disk(
            self.smoothing_value
        )

        mask = morphology.binary_closing(
            mask,
            disk
        )

        # Fill holes
        mask = ndimage.binary_fill_holes(mask)

        # Remove small objects
        min_size = max(
            20,
            round(mask.size * 0.0002)
        )

        mask = morphology.remove_small_objects(
            mask,
            min_size=min_size,
            connectivity=2
        )

        self.segmentation_mask = mask.astype(bool)

        self.refresh_display()

        self.set_status(
            f"Mask smoothed - radius {self.smoothing_value}"
        )

    # ========================================================
    # BACKGROUND MODE
    # ========================================================

    def update_background_mode(self, event=None):

        value = self.background_dropdown.get()

        if not self.segmentation_available:

            self.set_status(
                "Segment the subject first"
            )

            self.background_dropdown.set(
                "Background: Original"
            )

            return

        if value == "Background: White":

            self.background_mode = 1

        elif value == "Background: Black":

            self.background_mode = 2

        else:

            self.background_mode = 0

        self.refresh_display()

        self.set_status(
            f"{value} applied"
        )

    # ========================================================
    # SUBJECT STYLE
    # ========================================================

    def update_subject_style(self, event=None):

        value = self.subject_style_dropdown.get()

        if value == "Subject: Two-Tone":

            self.subject_style = 2

        else:

            self.subject_style = 1

        self.refresh_display()

        self.set_status(
            f"{value} selected"
        )

    # ========================================================
    # REFRESH DISPLAY
    # ========================================================

    def refresh_display(self):

        if self.mask_display_mode:

            self.show_mask_overlay()

            return

        self.processed_image = self.create_display_image()

        self.display_image(
            self.processed_image,
            self.result_canvas,
            "result"
        )

    # ========================================================
    # CREATE DISPLAY IMAGE
    # ========================================================

    def create_display_image(self):

        if (
            not self.segmentation_available
            or self.background_mode == 0
        ):

            return self.colour_processed_image.copy()

        result = self.colour_processed_image.copy()

        bg_mask = ~self.segmentation_mask

        if self.background_mode == 1:

            bg_colour = 255
            fg_colour = 0

        else:

            bg_colour = 0
            fg_colour = 255

        for channel in range(3):

            channel_data = result[:, :, channel]

            # Two-tone subject
            if self.subject_style == 2:

                channel_data[self.segmentation_mask] = (
                    fg_colour
                )

            # Replace background
            channel_data[bg_mask] = bg_colour

            result[:, :, channel] = channel_data

        return result

    # ========================================================
    # RESET IMAGE
    # ========================================================

    def reset_image(self):

        if self.original_image is None:
            return

        self.processed_image = self.original_image.copy()

        self.colour_processed_image = (
            self.original_image.copy()
        )

        self.segmentation_mask = None

        self.segmentation_available = False

        # Reset sliders
        self.saturation_slider.set(1)

        self.saturation_label.config(text="1.00")

        self.saturation_target_dropdown.set(
            "Subject Only"
        )

        self.smoothing_slider.set(3)

        self.smoothing_label.config(text="3")

        self.saturation_value = 1

        self.saturation_target = 1

        self.smoothing_value = 3

        # Reset modes
        self.background_mode = 0

        self.subject_style = 1

        self.mask_display_mode = False

        self.background_dropdown.set(
            "Background: Original"
        )

        self.subject_style_dropdown.set(
            "Subject: Colour"
        )

        self.update_displays()

        self.set_status(
            "Reset to original image"
        )

    # ========================================================
    # SAVE IMAGE
    # ========================================================

    def save_image(self):

        if self.processed_image is None:

            self.set_status(
                "No image to save"
            )

            return

        file_path = filedialog.asksaveasfilename(
            title="Save Processed Image",
            initialfile="segmented_rock_art.png",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
                ("BMP Image", "*.bmp")
            ]
        )

        if not file_path:

            self.set_status(
                "Save cancelled"
            )

            return

        try:

            # Save current displayed result
            output_image = self.processed_image

            Image.fromarray(
                output_image
            ).save(file_path)

            self.set_status(
                "Image saved successfully"
            )

        except Exception as error:

            messagebox.showerror(
                "Save Error",
                str(error)
            )

    # ========================================================
    # CLOSE APPLICATION
    # ========================================================

    def close_application(self):

        self.root.destroy()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = ImageSegmentationTool(root)

    root.mainloop()
