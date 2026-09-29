# ::48317:fft_pyg6.py:

import os
from datetime import datetime

import numpy as np
import pygame


# ============================================================
# Configuration
# ============================================================

WIDTH = 1400
HEIGHT = 900

BG = (28, 28, 32)
PANEL = (42, 42, 48)
PANEL_DARK = (34, 34, 39)
BORDER = (95, 95, 105)

TEXT = (235, 235, 240)
TEXT_DIM = (160, 160, 170)

ORANGE = (245, 150, 40)
BLUE = (60, 140, 255)

RENDER_DIR = "renders"
os.makedirs(RENDER_DIR, exist_ok=True)


# ============================================================
# Utility
# ============================================================

def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def pygame_surface_to_array(surface):
    rgb = pygame.surfarray.array3d(surface)
    rgb = np.transpose(rgb, (1, 0, 2))
    return rgb.astype(np.float32)


def array_to_surface(arr):
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return pygame.surfarray.make_surface(
        np.transpose(arr, (1, 0, 2))
    )


def load_image(path, size):
    image = pygame.image.load(path).convert()

    return pygame.transform.smoothscale(
        image,
        size
    )


# ============================================================
# RGB adjustment
# ============================================================

def adjust_rgb(
    image,
    luminance,
    vibrance,
    saturation
):
    x = image.astype(np.float32) / 255.0

    # Luminance
    x *= luminance

    # Saturation
    gray = (
        0.2126 * x[:, :, 0]
        + 0.7152 * x[:, :, 1]
        + 0.0722 * x[:, :, 2]
    )

    gray3 = gray[:, :, None]

    x = (
        gray3
        + saturation * (x - gray3)
    )

    # Vibrance
    max_channel = np.max(
        x,
        axis=2
    )

    min_channel = np.min(
        x,
        axis=2
    )

    sat = max_channel - min_channel

    factor = (
        1.0
        + vibrance * (1.0 - sat)
    )

    x = (
        gray3
        + (x - gray3)
        * factor[:, :, None]
    )

    return np.clip(
        x * 255.0,
        0,
        255
    )


# ============================================================
# Fourier image
# ============================================================

class FourierImage:

    def __init__(self, rgb):

        self.rgb = rgb.astype(
            np.float32
        )

        self.fft = np.fft.fft2(
            self.rgb,
            axes=(0, 1)
        )

        self.shifted = np.fft.fftshift(
            self.fft,
            axes=(0, 1)
        )

        self.magnitude = np.abs(
            self.shifted
        )

        self.phase = np.angle(
            self.shifted
        )


# ============================================================
# Fourier processor
# ============================================================

class FourierProcessor:

    def __init__(
        self,
        original1,
        original2
    ):

        self.image1 = FourierImage(
            original1
        )

        self.image2 = FourierImage(
            original2
        )

    def reconstruct(
        self,
        magnitude_source,
        phase_source,
        magnitude_fraction,
        phase_fraction
    ):

        if magnitude_source == 1:

            magnitude1 = (
                self.image1.magnitude
            )

            magnitude2 = (
                self.image2.magnitude
            )

        else:

            magnitude1 = (
                self.image2.magnitude
            )

            magnitude2 = (
                self.image1.magnitude
            )

        if phase_source == 1:

            phase1 = (
                self.image1.phase
            )

            phase2 = (
                self.image2.phase
            )

        else:

            phase1 = (
                self.image2.phase
            )

            phase2 = (
                self.image1.phase
            )

        # Linear magnitude interpolation.
        magnitude = (
            (1.0 - magnitude_fraction)
            * magnitude1
            + magnitude_fraction
            * magnitude2
        )

        # Shortest circular phase interpolation.
        phase_difference = np.angle(
            np.exp(
                1j * (
                    phase2
                    - phase1
                )
            )
        )

        phase = (
            phase1
            + phase_fraction
            * phase_difference
        )

        spectrum_shifted = (
            magnitude
            * np.exp(
                1j * phase
            )
        )

        spectrum = np.fft.ifftshift(
            spectrum_shifted,
            axes=(0, 1)
        )

        result = np.fft.ifft2(
            spectrum,
            axes=(0, 1)
        ).real

        low = result.min()
        high = result.max()

        if high > low:

            result = (
                (result - low)
                / (high - low)
                * 255.0
            )

        else:

            result[:] = 0

        return np.clip(
            result,
            0,
            255
        ).astype(np.float32)


# ============================================================
# Slider
# ============================================================

class Slider:

    def __init__(
        self,
        rect,
        label,
        minimum,
        maximum,
        value
    ):

        self.rect = pygame.Rect(rect)

        self.label = label

        self.minimum = minimum
        self.maximum = maximum
        self.value = value

    def knob_x(self):

        fraction = (
            self.value
            - self.minimum
        ) / (
            self.maximum
            - self.minimum
        )

        return (
            self.rect.left
            + int(
                fraction
                * self.rect.width
            )
        )

    def set_from_mouse(self, x):

        fraction = (
            x
            - self.rect.left
        ) / self.rect.width

        fraction = clamp(
            fraction,
            0.0,
            1.0
        )

        self.value = (
            self.minimum
            + fraction
            * (
                self.maximum
                - self.minimum
            )
        )

    def hit(self, pos):

        return self.rect.inflate(
            0,
            22
        ).collidepoint(pos)

    def draw(
        self,
        screen,
        font
    ):

        label_surface = font.render(
            f"{self.label}: {self.value:.2f}",
            True,
            TEXT
        )

        screen.blit(
            label_surface,
            (
                self.rect.left,
                self.rect.top - 22
            )
        )

        y = self.rect.centery

        pygame.draw.line(
            screen,
            (100, 100, 110),
            (
                self.rect.left,
                y
            ),
            (
                self.rect.right,
                y
            ),
            5
        )

        pygame.draw.circle(
            screen,
            ORANGE,
            (
                self.knob_x(),
                y
            ),
            8
        )


# ============================================================
# Main viewer
# ============================================================

class FourierViewer:

    MATRIX_N = 9
    MATRIX_SIZE = 150

    IMAGE_Y = 295

    def __init__(
        self,
        image1_path,
        image2_path
    ):

        pygame.init()

        self.screen = pygame.display.set_mode(
            (
                WIDTH,
                HEIGHT
            )
        )

        pygame.display.set_caption(
            "RGB Fourier Image Viewer"
        )

        self.clock = pygame.time.Clock()

        # ----------------------------------------------------
        # Fonts
        # ----------------------------------------------------

        self.font = pygame.font.SysFont(
            "Arial",
            18
        )

        self.font_small = pygame.font.SysFont(
            "Arial",
            14
        )

        self.font_title = pygame.font.SysFont(
            "Arial",
            22,
            bold=True
        )

        self.font_matrix = pygame.font.SysFont(
            "Arial",
            18,
            bold=True
        )

        self.font_file = pygame.font.SysFont(
            "Arial",
            17
        )

        # ----------------------------------------------------
        # Image panels
        # ----------------------------------------------------

        self.panel_w = 420
        self.panel_h = 420

        self.image_size = (
            self.panel_w - 20,
            self.panel_h - 55
        )

        # ----------------------------------------------------
        # Load originals
        # ----------------------------------------------------

        self.original1_surface = load_image(
            image1_path,
            self.image_size
        )

        self.original2_surface = load_image(
            image2_path,
            self.image_size
        )

        self.original1 = (
            pygame_surface_to_array(
                self.original1_surface
            )
        )

        self.original2 = (
            pygame_surface_to_array(
                self.original2_surface
            )
        )

        # ----------------------------------------------------
        # Fourier processor
        # ----------------------------------------------------

        self.processor = FourierProcessor(
            self.original1,
            self.original2
        )

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        self.mode = "original"

        self.phase_selection = 1
        self.magnitude_selection = 1

        self.pixel_position = 0.0

        self.matrix_row = 0
        self.matrix_col = 0

        self.fourier_cache = {}

        # ----------------------------------------------------
        # Adjustments
        # ----------------------------------------------------

        self.luminance = 1.0
        self.vibrance = 0.0
        self.saturation = 1.0

        # ----------------------------------------------------
        # LOAD state
        # ----------------------------------------------------

        self.load_mode = False
        self.load_target = None

        self.png_files = []
        self.file_rects = []

        # ----------------------------------------------------
        # Layout
        # ----------------------------------------------------

        self.calculate_layout()

        # ----------------------------------------------------
        # Sliders
        # ----------------------------------------------------

        slider_x = self.panel_x1
        slider_width = self.panel_w - 20

        self.slider_luminance = Slider(
            (
                slider_x,
                78,
                slider_width,
                10
            ),
            "Luminance",
            0.5,
            1.5,
            1.0
        )

        self.slider_vibrance = Slider(
            (
                slider_x,
                125,
                slider_width,
                10
            ),
            "Vibrance",
            0.0,
            2.0,
            0.0
        )

        self.slider_saturation = Slider(
            (
                slider_x,
                172,
                slider_width,
                10
            ),
            "Saturation",
            0.0,
            2.0,
            1.0
        )

        self.active_slider = None

        # ----------------------------------------------------
        # Source selectors
        # ----------------------------------------------------

        self.group_w = 155
        self.group_h = 96

        self.group1_rect = pygame.Rect(
            self.panel_x3,
            55,
            self.group_w,
            self.group_h
        )

        self.group2_rect = pygame.Rect(
            self.panel_x3
            + self.group_w
            + 12,
            55,
            self.group_w,
            self.group_h
        )

        self.phase1_button = pygame.Rect(
            self.group1_rect.left + 8,
            self.group1_rect.top + 31,
            self.group_w - 16,
            25
        )

        self.mag1_button = pygame.Rect(
            self.group1_rect.left + 8,
            self.group1_rect.top + 62,
            self.group_w - 16,
            25
        )

        self.phase2_button = pygame.Rect(
            self.group2_rect.left + 8,
            self.group2_rect.top + 31,
            self.group_w - 16,
            25
        )

        self.mag2_button = pygame.Rect(
            self.group2_rect.left + 8,
            self.group2_rect.top + 62,
            self.group_w - 16,
            25
        )

        # ----------------------------------------------------
        # Action buttons
        # ----------------------------------------------------

        button_y = (
            self.panel_y
            + self.panel_h
            + 10
        )

        button_h = 34

        self.load_button = pygame.Rect(
            self.panel_x1,
            button_y,
            105,
            button_h
        )

        self.pixel_button = pygame.Rect(
            self.panel_x1 + 115,
            button_y,
            130,
            button_h
        )

        self.fourier_button = pygame.Rect(
            self.panel_x1 + 255,
            button_y,
            105,
            button_h
        )

        self.save_button = pygame.Rect(
            self.panel_x1 + 370,
            button_y,
            100,
            button_h
        )

        # ----------------------------------------------------
        # Matrix
        # ----------------------------------------------------

        self.matrix_x = (
            WIDTH
            - self.MATRIX_SIZE
        ) // 2

        self.matrix_y = 45

        self.matrix_cell = (
            self.MATRIX_SIZE
            / self.MATRIX_N
        )

        # ----------------------------------------------------
        # Initial result
        # ----------------------------------------------------

        self.current_result = (
            self.original1.copy()
        )

        self.display_surface = array_to_surface(
            self.current_result
        )

    # ========================================================
    # Layout
    # ========================================================

    def calculate_layout(self):

        gap = 20

        total_width = (
            self.panel_w * 3
            + gap * 2
        )

        start_x = (
            WIDTH
            - total_width
        ) // 2

        self.panel_x1 = start_x

        self.panel_x2 = (
            self.panel_x1
            + self.panel_w
            + gap
        )

        self.panel_x3 = (
            self.panel_x2
            + self.panel_w
            + gap
        )

        self.panel_y = self.IMAGE_Y

    # ========================================================
    # Adjustment
    # ========================================================

    def refresh_adjusted_result(self):

        adjusted = adjust_rgb(
            self.current_result,
            self.luminance,
            self.vibrance,
            self.saturation
        )

        self.display_surface = array_to_surface(
            adjusted
        )

    # ========================================================
    # Source combination
    # ========================================================

    def show_selected_source_combination(self):

        phase = self.phase_selection
        magnitude = self.magnitude_selection

        if phase == 1 and magnitude == 1:

            self.mode = "source_combination"

            self.current_result = (
                self.original1.copy()
            )

            self.refresh_adjusted_result()

            return

        if phase == 2 and magnitude == 2:

            self.mode = "source_combination"

            self.current_result = (
                self.original2.copy()
            )

            self.refresh_adjusted_result()

            return

        if phase == 1 and magnitude == 2:

            result = self.processor.reconstruct(
                magnitude_source=2,
                phase_source=1,
                magnitude_fraction=0.0,
                phase_fraction=0.0
            )

            self.current_result = result

            self.refresh_adjusted_result()

            return

        if phase == 2 and magnitude == 1:

            result = self.processor.reconstruct(
                magnitude_source=1,
                phase_source=2,
                magnitude_fraction=0.0,
                phase_fraction=0.0
            )

            self.current_result = result

            self.refresh_adjusted_result()

    # ========================================================
    # Fourier cell
    # ========================================================

    def calculate_fourier_cell(
        self,
        row,
        col
    ):

        key = (
            row,
            col
        )

        if key in self.fourier_cache:

            self.current_result = (
                self.fourier_cache[key].copy()
            )

            self.refresh_adjusted_result()

            return

        amplitude_fraction = (
            row / 8.0
        )

        phase_fraction = (
            col / 8.0
        )

        result = self.processor.reconstruct(
            magnitude_source=1,
            phase_source=1,
            magnitude_fraction=amplitude_fraction,
            phase_fraction=phase_fraction
        )

        self.fourier_cache[key] = (
            result.copy()
        )

        filename = (
            f"fourier_a{row + 1}"
            f"_p{col + 1}.png"
        )

        path = os.path.join(
            RENDER_DIR,
            filename
        )

        if not os.path.exists(path):

            pygame.image.save(
                array_to_surface(result),
                path
            )

        self.current_result = result.copy()

        self.refresh_adjusted_result()

    # ========================================================
    # Fourier selection
    # ========================================================

    def show_fourier_selection(self):

        row = self.matrix_row
        col = self.matrix_col

        if row == 0 and col == 0:

            self.current_result = (
                self.original1.copy()
            )

            self.refresh_adjusted_result()

            return

        if row == 8 and col == 8:

            self.current_result = (
                self.original2.copy()
            )

            self.refresh_adjusted_result()

            return

        self.calculate_fourier_cell(
            row,
            col
        )

    # ========================================================
    # Activate Fourier mode
    # ========================================================

    def activate_fourier(self):

        self.mode = "fourier_blend"

        self.matrix_row = 0
        self.matrix_col = 0

        # Top row.
        for col in range(1, 9):

            self.calculate_fourier_cell(
                0,
                col
            )

        # Right column.
        for row in range(1, 8):

            self.calculate_fourier_cell(
                row,
                8
            )

        self.matrix_row = 0
        self.matrix_col = 0

        self.current_result = (
            self.original1.copy()
        )

        self.refresh_adjusted_result()

    # ========================================================
    # Pixel blend
    # ========================================================

    def show_pixel_blend(self):

        a = self.pixel_position

        result = (
            (1.0 - a)
            * self.original1
            + a
            * self.original2
        )

        self.current_result = np.clip(
            result,
            0,
            255
        ).astype(np.float32)

        self.refresh_adjusted_result()

    # ========================================================
    # Save
    # ========================================================

    def save_result(self):

        filename = (
            "fourier_result_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )
            + ".png"
        )

        path = os.path.join(
            RENDER_DIR,
            filename
        )

        pygame.image.save(
            self.display_surface,
            path
        )

        print(
            f"Saved: {path}"
        )

    # ========================================================
    # LOAD
    # ========================================================

    def begin_load(self):

        self.load_mode = True
        self.load_target = None

        self.png_files = sorted(
            [
                filename
                for filename in os.listdir(".")
                if (
                    os.path.isfile(filename)
                    and filename.lower().endswith(".png")
                )
            ],
            key=str.lower
        )

        self.file_rects = []

        print(
            "LOAD: click Original 1 or Original 2"
        )

    # ========================================================
    # Start PNG selection
    # ========================================================

    def open_png_selector(
        self,
        original_number
    ):

        self.load_target = original_number

        self.png_files = sorted(
            [
                filename
                for filename in os.listdir(".")
                if (
                    os.path.isfile(filename)
                    and filename.lower().endswith(".png")
                )
            ],
            key=str.lower
        )

        self.file_rects = []

    # ========================================================
    # Draw PNG selector
    # ========================================================

    def draw_file_selector(self):

        # Full-screen overlay.
        overlay = pygame.Surface(
            (
                WIDTH,
                HEIGHT
            ),
            pygame.SRCALPHA
        )

        overlay.fill(
            (
                0,
                0,
                0,
                235
            )
        )

        self.screen.blit(
            overlay,
            (
                0,
                0
            )
        )

        title = self.font_title.render(
            "SELECT PNG FILE",
            True,
            TEXT
        )

        self.screen.blit(
            title,
            title.get_rect(
                centerx=WIDTH // 2,
                top=35
            )
        )

        directory = self.font_small.render(
            f"Current directory: {os.getcwd()}",
            True,
            TEXT_DIM
        )

        self.screen.blit(
            directory,
            directory.get_rect(
                centerx=WIDTH // 2,
                top=68
            )
        )

        instruction = self.font_small.render(
            "Click a file to load it    •    ESC to cancel",
            True,
            TEXT_DIM
        )

        self.screen.blit(
            instruction,
            instruction.get_rect(
                centerx=WIDTH // 2,
                top=92
            )
        )

        # ----------------------------------------------------
        # File list
        # ----------------------------------------------------

        list_x = 250
        list_y = 135

        row_height = 30

        max_rows = (
            HEIGHT
            - list_y
            - 30
        ) // row_height

        visible_files = self.png_files[
            :max_rows
        ]

        self.file_rects = []

        for index, filename in enumerate(
            visible_files
        ):

            rect = pygame.Rect(
                list_x,
                list_y
                + index * row_height,
                WIDTH - 500,
                row_height - 3
            )

            self.file_rects.append(
                (
                    rect,
                    filename
                )
            )

            if rect.collidepoint(
                pygame.mouse.get_pos()
            ):

                background = ORANGE
                foreground = (20, 20, 20)

            else:

                background = PANEL
                foreground = TEXT

            pygame.draw.rect(
                self.screen,
                background,
                rect,
                border_radius=4
            )

            text = self.font_file.render(
                filename,
                True,
                foreground
            )

            self.screen.blit(
                text,
                (
                    rect.left + 10,
                    rect.centery
                    - text.get_height() // 2
                )
            )

        if not self.png_files:

            empty = self.font.render(
                "No PNG files found.",
                True,
                TEXT_DIM
            )

            self.screen.blit(
                empty,
                empty.get_rect(
                    centerx=WIDTH // 2,
                    top=150
                )
            )

        elif len(self.png_files) > max_rows:

            more = self.font_small.render(
                f"{len(self.png_files) - max_rows} more files...",
                True,
                TEXT_DIM
            )

            self.screen.blit(
                more,
                more.get_rect(
                    centerx=WIDTH // 2,
                    bottom=25
                )
            )

    # ========================================================
    # Load selected PNG
    # ========================================================

    def load_png_file(
        self,
        filename
    ):

        if self.load_target is None:

            return

        try:

            surface = load_image(
                filename,
                self.image_size
            )

            image = pygame_surface_to_array(
                surface
            )

        except Exception as exc:

            print(
                f"Could not load image "
                f"{filename}: {exc}"
            )

            return

        if self.load_target == 1:

            self.original1_surface = surface
            self.original1 = image

            print(
                f"Original 1 replaced by {filename}"
            )

        else:

            self.original2_surface = surface
            self.original2 = image

            print(
                f"Original 2 replaced by {filename}"
            )

        self.reset_gui_after_load()

    # ========================================================
    # Reset after LOAD
    # ========================================================

    def reset_gui_after_load(self):

        self.processor = FourierProcessor(
            self.original1,
            self.original2
        )

        self.fourier_cache = {}

        self.mode = "original"

        self.phase_selection = 1
        self.magnitude_selection = 1

        self.pixel_position = 0.0

        self.matrix_row = 0
        self.matrix_col = 0

        self.current_result = (
            self.original1.copy()
        )

        self.refresh_adjusted_result()

        self.load_mode = False
        self.load_target = None
        self.png_files = []
        self.file_rects = []

    # ========================================================
    # Draw image panel
    # ========================================================

    def draw_image_panel(
        self,
        x,
        title,
        image_surface
    ):

        panel_rect = pygame.Rect(
            x,
            self.panel_y,
            self.panel_w,
            self.panel_h
        )

        pygame.draw.rect(
            self.screen,
            PANEL,
            panel_rect,
            border_radius=8
        )

        pygame.draw.rect(
            self.screen,
            BORDER,
            panel_rect,
            1,
            border_radius=8
        )

        title_surface = self.font_title.render(
            title,
            True,
            TEXT
        )

        title_rect = title_surface.get_rect(
            centerx=panel_rect.centerx,
            top=panel_rect.top + 8
        )

        self.screen.blit(
            title_surface,
            title_rect
        )

        image_rect = image_surface.get_rect(
            center=(
                panel_rect.centerx,
                panel_rect.top
                + 55
                + (
                    self.panel_h - 55
                ) // 2
            )
        )

        self.screen.blit(
            image_surface,
            image_rect
        )

    # ========================================================
    # Draw source group
    # ========================================================

    def draw_source_group(
        self,
        group_rect,
        title,
        phase_rect,
        phase_text,
        mag_rect,
        mag_text,
        phase_active,
        mag_active
    ):

        pygame.draw.rect(
            self.screen,
            PANEL,
            group_rect,
            border_radius=7
        )

        pygame.draw.rect(
            self.screen,
            BORDER,
            group_rect,
            1,
            border_radius=7
        )

        title_surface = self.font_small.render(
            title,
            True,
            TEXT
        )

        self.screen.blit(
            title_surface,
            title_surface.get_rect(
                centerx=group_rect.centerx,
                top=group_rect.top + 6
            )
        )

        # Phase.
        pygame.draw.rect(
            self.screen,
            ORANGE if phase_active else PANEL_DARK,
            phase_rect,
            border_radius=5
        )

        pygame.draw.rect(
            self.screen,
            BORDER,
            phase_rect,
            1,
            border_radius=5
        )

        text_surface = self.font_small.render(
            phase_text,
            True,
            (20, 20, 20)
            if phase_active
            else TEXT
        )

        self.screen.blit(
            text_surface,
            text_surface.get_rect(
                center=phase_rect.center
            )
        )

        # Magnitude.
        pygame.draw.rect(
            self.screen,
            BLUE if mag_active else PANEL_DARK,
            mag_rect,
            border_radius=5
        )

        pygame.draw.rect(
            self.screen,
            BORDER,
            mag_rect,
            1,
            border_radius=5
        )

        text_surface = self.font_small.render(
            mag_text,
            True,
            (20, 20, 20)
            if mag_active
            else TEXT
        )

        self.screen.blit(
            text_surface,
            text_surface.get_rect(
                center=mag_rect.center
            )
        )

    # ========================================================
    # Draw source groups
    # ========================================================

    def draw_source_groups(self):

        self.draw_source_group(
            self.group1_rect,
            "ORIGINAL 1",
            self.phase1_button,
            "PHASE 1",
            self.mag1_button,
            "MAG 1",
            self.phase_selection == 1,
            self.magnitude_selection == 1
        )

        self.draw_source_group(
            self.group2_rect,
            "ORIGINAL 2",
            self.phase2_button,
            "PHASE 2",
            self.mag2_button,
            "MAG 2",
            self.phase_selection == 2,
            self.magnitude_selection == 2
        )

    # ========================================================
    # Draw Fourier matrix
    # ========================================================

    def draw_matrix(self):

        x = self.matrix_x
        y = self.matrix_y

        size = self.MATRIX_SIZE
        cell = self.matrix_cell

        title = self.font.render(
            "FOURIER MATRIX",
            True,
            TEXT
        )

        self.screen.blit(
            title,
            title.get_rect(
                centerx=x + size / 2,
                bottom=y - 8
            )
        )

        # PHASE arrow.
        arrow_y = y - 27

        pygame.draw.line(
            self.screen,
            TEXT_DIM,
            (
                x,
                arrow_y
            ),
            (
                x + size,
                arrow_y
            ),
            2
        )

        pygame.draw.polygon(
            self.screen,
            TEXT_DIM,
            [
                (
                    x + size,
                    arrow_y
                ),
                (
                    x + size - 8,
                    arrow_y - 4
                ),
                (
                    x + size - 8,
                    arrow_y + 4
                )
            ]
        )

        phase_label = self.font_small.render(
            "PHASE",
            True,
            TEXT
        )

        self.screen.blit(
            phase_label,
            phase_label.get_rect(
                center=(
                    x + size / 2,
                    arrow_y - 10
                )
            )
        )

        # AMPLITUDE arrow.
        arrow_x = x - 28

        pygame.draw.line(
            self.screen,
            TEXT_DIM,
            (
                arrow_x,
                y
            ),
            (
                arrow_x,
                y + size
            ),
            2
        )

        pygame.draw.polygon(
            self.screen,
            TEXT_DIM,
            [
                (
                    arrow_x,
                    y + size
                ),
                (
                    arrow_x - 4,
                    y + size - 8
                ),
                (
                    arrow_x + 4,
                    y + size - 8
                )
            ]
        )

        amp_label = self.font_small.render(
            "AMPLITUDE",
            True,
            TEXT
        )

        amp_label = pygame.transform.rotate(
            amp_label,
            90
        )

        self.screen.blit(
            amp_label,
            amp_label.get_rect(
                center=(
                    arrow_x - 10,
                    y + size / 2
                )
            )
        )

        # 9 x 9 cells.
        for row in range(
            self.MATRIX_N
        ):

            for col in range(
                self.MATRIX_N
            ):

                left = round(
                    x + col * cell
                )

                top = round(
                    y + row * cell
                )

                right = round(
                    x + (col + 1) * cell
                )

                bottom = round(
                    y + (row + 1) * cell
                )

                rect = pygame.Rect(
                    left,
                    top,
                    right - left,
                    bottom - top
                )

                selected = (
                    row == self.matrix_row
                    and col == self.matrix_col
                )

                if selected:

                    fill = ORANGE

                else:

                    fill = (
                        (58, 58, 66)
                        if (
                            row + col
                        ) % 2 == 0
                        else (48, 48, 55)
                    )

                pygame.draw.rect(
                    self.screen,
                    fill,
                    rect
                )

                pygame.draw.rect(
                    self.screen,
                    BORDER,
                    rect,
                    1
                )

                # Only 1 and 2 appear inside cells.
                if row == 0 and col == 0:

                    text_surface = (
                        self.font_matrix.render(
                            "1",
                            True,
                            (20, 20, 20)
                            if selected
                            else TEXT
                        )
                    )

                    self.screen.blit(
                        text_surface,
                        text_surface.get_rect(
                            center=rect.center
                        )
                    )

                elif row == 8 and col == 8:

                    text_surface = (
                        self.font_matrix.render(
                            "2",
                            True,
                            (20, 20, 20)
                            if selected
                            else TEXT
                        )
                    )

                    self.screen.blit(
                        text_surface,
                        text_surface.get_rect(
                            center=rect.center
                        )
                    )

    # ========================================================
    # Draw button
    # ========================================================

    def draw_button(
        self,
        rect,
        label,
        active=False
    ):

        pygame.draw.rect(
            self.screen,
            ORANGE if active else PANEL,
            rect,
            border_radius=6
        )

        pygame.draw.rect(
            self.screen,
            BORDER,
            rect,
            1,
            border_radius=6
        )

        surface = self.font.render(
            label,
            True,
            (20, 20, 20)
            if active
            else TEXT
        )

        self.screen.blit(
            surface,
            surface.get_rect(
                center=rect.center
            )
        )

    # ========================================================
    # Draw
    # ========================================================

    def draw(self):

        self.screen.fill(
            BG
        )

        # ----------------------------------------------------
        # Normal GUI
        # ----------------------------------------------------

        title = self.font_title.render(
            "RGB Fourier Image Viewer",
            True,
            TEXT
        )

        self.screen.blit(
            title,
            (
                20,
                18
            )
        )

        self.slider_luminance.draw(
            self.screen,
            self.font_small
        )

        self.slider_vibrance.draw(
            self.screen,
            self.font_small
        )

        self.slider_saturation.draw(
            self.screen,
            self.font_small
        )

        if self.mode == "fourier_blend":

            self.draw_matrix()

        self.draw_source_groups()

        self.draw_image_panel(
            self.panel_x1,
            "ORIGINAL 1",
            self.original1_surface
        )

        self.draw_image_panel(
            self.panel_x2,
            "ORIGINAL 2",
            self.original2_surface
        )

        self.draw_image_panel(
            self.panel_x3,
            "RESULT",
            self.display_surface
        )

        self.draw_button(
            self.load_button,
            "LOAD"
        )

        self.draw_button(
            self.pixel_button,
            "PIXEL BLEND",
            self.mode == "pixel_blend"
        )

        self.draw_button(
            self.fourier_button,
            "F-BLEND",
            self.mode == "fourier_blend"
        )

        self.draw_button(
            self.save_button,
            "SAVE"
        )

        # ----------------------------------------------------
        # LOAD instruction
        # ----------------------------------------------------

        if (
            self.load_mode
            and self.load_target is None
        ):

            message = self.font.render(
                "click on image to replace it",
                True,
                ORANGE
            )

            self.screen.blit(
                message,
                message.get_rect(
                    centerx=WIDTH // 2,
                    top=210
                )
            )

        # ----------------------------------------------------
        # PNG file selector
        # ----------------------------------------------------

        if (
            self.load_mode
            and self.load_target is not None
        ):

            self.draw_file_selector()

        pygame.display.flip()

    # ========================================================
    # Source click
    # ========================================================

    def handle_source_click(
        self,
        pos
    ):

        if self.phase1_button.collidepoint(
            pos
        ):

            self.phase_selection = 1

            self.show_selected_source_combination()

            return True

        if self.mag1_button.collidepoint(
            pos
        ):

            self.magnitude_selection = 1

            self.show_selected_source_combination()

            return True

        if self.phase2_button.collidepoint(
            pos
        ):

            self.phase_selection = 2

            self.show_selected_source_combination()

            return True

        if self.mag2_button.collidepoint(
            pos
        ):

            self.magnitude_selection = 2

            self.show_selected_source_combination()

            return True

        return False

    # ========================================================
    # Matrix click
    # ========================================================

    def handle_matrix_click(
        self,
        pos
    ):

        matrix_rect = pygame.Rect(
            self.matrix_x,
            self.matrix_y,
            self.MATRIX_SIZE,
            self.MATRIX_SIZE
        )

        if not matrix_rect.collidepoint(
            pos
        ):

            return False

        col = int(
            (
                pos[0]
                - self.matrix_x
            )
            / self.matrix_cell
        )

        row = int(
            (
                pos[1]
                - self.matrix_y
            )
            / self.matrix_cell
        )

        col = int(
            clamp(
                col,
                0,
                self.MATRIX_N - 1
            )
        )

        row = int(
            clamp(
                row,
                0,
                self.MATRIX_N - 1
            )
        )

        self.matrix_col = col
        self.matrix_row = row

        self.show_fourier_selection()

        return True

    # ========================================================
    # Mouse down
    # ========================================================

    def handle_mouse_down(
        self,
        pos
    ):

        # ----------------------------------------------------
        # LOAD file selector.
        # ----------------------------------------------------

        if (
            self.load_mode
            and self.load_target is not None
        ):

            for rect, filename in self.file_rects:

                if rect.collidepoint(pos):

                    self.load_png_file(
                        filename
                    )

                    return

            return

        # ----------------------------------------------------
        # Fourier matrix.
        # ----------------------------------------------------

        if self.mode == "fourier_blend":

            if self.handle_matrix_click(
                pos
            ):

                return

        # ----------------------------------------------------
        # Sliders.
        # ----------------------------------------------------

        if self.slider_luminance.hit(
            pos
        ):

            self.active_slider = (
                self.slider_luminance
            )

            self.active_slider.set_from_mouse(
                pos[0]
            )

            self.luminance = (
                self.active_slider.value
            )

            self.refresh_adjusted_result()

            return

        if self.slider_vibrance.hit(
            pos
        ):

            self.active_slider = (
                self.slider_vibrance
            )

            self.active_slider.set_from_mouse(
                pos[0]
            )

            self.vibrance = (
                self.active_slider.value
            )

            self.refresh_adjusted_result()

            return

        if self.slider_saturation.hit(
            pos
        ):

            self.active_slider = (
                self.slider_saturation
            )

            self.active_slider.set_from_mouse(
                pos[0]
            )

            self.saturation = (
                self.active_slider.value
            )

            self.refresh_adjusted_result()

            return

        # ----------------------------------------------------
        # LOAD mode: choose Original 1 or Original 2.
        # ----------------------------------------------------

        if self.load_mode:

            original1_rect = pygame.Rect(
                self.panel_x1,
                self.panel_y,
                self.panel_w,
                self.panel_h
            )

            original2_rect = pygame.Rect(
                self.panel_x2,
                self.panel_y,
                self.panel_w,
                self.panel_h
            )

            if original1_rect.collidepoint(
                pos
            ):

                self.open_png_selector(1)

                return

            if original2_rect.collidepoint(
                pos
            ):

                self.open_png_selector(2)

                return

            return

        # ----------------------------------------------------
        # Source selectors.
        # ----------------------------------------------------

        if self.handle_source_click(
            pos
        ):

            return

        # ----------------------------------------------------
        # LOAD.
        # ----------------------------------------------------

        if self.load_button.collidepoint(
            pos
        ):

            self.begin_load()

            return

        # ----------------------------------------------------
        # PIXEL BLEND.
        # ----------------------------------------------------

        if self.pixel_button.collidepoint(
            pos
        ):

            self.mode = "pixel_blend"

            self.pixel_position = 0.0

            self.show_pixel_blend()

            return

        # ----------------------------------------------------
        # F-BLEND.
        # ----------------------------------------------------

        if self.fourier_button.collidepoint(
            pos
        ):

            self.activate_fourier()

            return

        # ----------------------------------------------------
        # SAVE.
        # ----------------------------------------------------

        if self.save_button.collidepoint(
            pos
        ):

            self.save_result()

            return

    # ========================================================
    # Mouse motion
    # ========================================================

    def handle_mouse_motion(
        self,
        pos
    ):

        if self.active_slider is None:

            return

        self.active_slider.set_from_mouse(
            pos[0]
        )

        if (
            self.active_slider
            is self.slider_luminance
        ):

            self.luminance = (
                self.active_slider.value
            )

        elif (
            self.active_slider
            is self.slider_vibrance
        ):

            self.vibrance = (
                self.active_slider.value
            )

        elif (
            self.active_slider
            is self.slider_saturation
        ):

            self.saturation = (
                self.active_slider.value
            )

        self.refresh_adjusted_result()

    # ========================================================
    # Keyboard
    # ========================================================

    def handle_key(
        self,
        key
    ):

        # ----------------------------------------------------
        # LOAD selector.
        # ----------------------------------------------------

        if self.load_mode:

            if key == pygame.K_ESCAPE:

                self.load_mode = False
                self.load_target = None
                self.png_files = []
                self.file_rects = []

            return

        # ----------------------------------------------------
        # Fourier matrix.
        # ----------------------------------------------------

        if self.mode == "fourier_blend":

            if key == pygame.K_LEFT:

                self.matrix_col = max(
                    0,
                    self.matrix_col - 1
                )

                self.show_fourier_selection()

            elif key == pygame.K_RIGHT:

                self.matrix_col = min(
                    self.MATRIX_N - 1,
                    self.matrix_col + 1
                )

                self.show_fourier_selection()

            elif key == pygame.K_UP:

                self.matrix_row = max(
                    0,
                    self.matrix_row - 1
                )

                self.show_fourier_selection()

            elif key == pygame.K_DOWN:

                self.matrix_row = min(
                    self.MATRIX_N - 1,
                    self.matrix_row + 1
                )

                self.show_fourier_selection()

            return

        # ----------------------------------------------------
        # Pixel blend.
        # ----------------------------------------------------

        if self.mode == "pixel_blend":

            if key == pygame.K_LEFT:

                self.pixel_position = clamp(
                    self.pixel_position - 0.05,
                    0.0,
                    1.0
                )

                self.show_pixel_blend()

            elif key == pygame.K_RIGHT:

                self.pixel_position = clamp(
                    self.pixel_position + 0.05,
                    0.0,
                    1.0
                )

                self.show_pixel_blend()

    # ========================================================
    # Main loop
    # ========================================================

    def run(self):

        running = True

        while running:

            for event in pygame.event.get():

                if event.type == pygame.QUIT:

                    running = False

                elif event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_ESCAPE:

                        if self.load_mode:

                            self.load_mode = False
                            self.load_target = None
                            self.png_files = []
                            self.file_rects = []

                        else:

                            running = False

                    else:

                        self.handle_key(
                            event.key
                        )

                elif event.type == pygame.MOUSEBUTTONDOWN:

                    if event.button == 1:

                        self.handle_mouse_down(
                            event.pos
                        )

                elif event.type == pygame.MOUSEMOTION:

                    if self.active_slider is not None:

                        self.handle_mouse_motion(
                            event.pos
                        )

                elif event.type == pygame.MOUSEBUTTONUP:

                    if event.button == 1:

                        self.active_slider = None

            self.draw()

            self.clock.tick(60)

        pygame.quit()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    IMAGE1 = "original1.png"
    IMAGE2 = "original2.png"

    viewer = FourierViewer(
        IMAGE1,
        IMAGE2
    )

    viewer.run()
