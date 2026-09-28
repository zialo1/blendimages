"""
Interactive Fourier image viewer using Pygame.

Features
--------
- Two input images.
- Fourier magnitude/phase recombination.
- RGB/color processing with --color.
- Four Fourier combinations:
    magnitude 1 + phase 1
    magnitude 1 + phase 2
    magnitude 2 + phase 1
    magnitude 2 + phase 2
- 12-step pixel-space blending.
- Left/right cursor keys navigate results.
- Three image adjustments:
    luminance  -30% .. +30%
    vibrance   -30% .. +30%
    saturation -30% .. +30%
- ESC terminates the application.
"""

import argparse
import sys

import numpy as np
import pygame
from PIL import Image


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive Fourier image viewer using Pygame."
    )

    parser.add_argument(
        "image1",
        help="first input image"
    )

    parser.add_argument(
        "image2",
        help="second input image"
    )

    parser.add_argument(
        "--color",
        action="store_true",
        help="process RGB color images instead of grayscale"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1400,
        help="Pygame window width (default: 1400)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Pygame window height (default: 800)"
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Fourier image
# ----------------------------------------------------------------------

class FourierImage:
    """Store an image and its Fourier magnitude and phase."""

    def __init__(self, image):
        """
        Initialize the Fourier representation.

        Parameters
        ----------
        image : numpy.ndarray
            Grayscale image of shape (H, W), or RGB image
            of shape (H, W, 3).
        """
        self.image = image.astype(np.float32)

        self.fft = self._calculate_fft()
        self.magnitude = np.abs(self.fft)
        self.phase = np.angle(self.fft)

    def _calculate_fft(self):
        """Calculate the two-dimensional FFT for all image channels."""
        if self.image.ndim == 2:
            return np.fft.fft2(self.image)

        channels = []

        for channel in range(self.image.shape[2]):
            channels.append(
                np.fft.fft2(self.image[..., channel])
            )

        return np.stack(channels, axis=-1)

    def reconstruct(self, magnitude, phase):
        """
        Reconstruct an image from a magnitude and phase array.

        Parameters
        ----------
        magnitude : numpy.ndarray
            Fourier magnitude.
        phase : numpy.ndarray
            Fourier phase.

        Returns
        -------
        numpy.ndarray
            Reconstructed image.
        """
        spectrum = magnitude * np.exp(1j * phase)

        if spectrum.ndim == 2:
            result = np.fft.ifft2(spectrum).real
        else:
            channels = []

            for channel in range(spectrum.shape[2]):
                channels.append(
                    np.fft.ifft2(
                        spectrum[..., channel]
                    ).real
                )

            result = np.stack(channels, axis=-1)

        return result


# ----------------------------------------------------------------------
# Image processing
# ----------------------------------------------------------------------

class ImageProcessor:
    """Load images and generate Fourier and blend results."""

    def __init__(self, image1_path, image2_path, color=False):
        """
        Initialize the image processor.

        Parameters
        ----------
        image1_path : str
            Path to the first image.
        image2_path : str
            Path to the second image.
        color : bool
            If True, process RGB channels independently.
        """
        self.color = color

        self.image1 = self._load(image1_path)
        self.image2 = self._load(image2_path)

        self._check_dimensions()

        self.fourier1 = FourierImage(self.image1)
        self.fourier2 = FourierImage(self.image2)

    def _load(self, path):
        """Load an image and convert it to the requested representation."""
        try:
            image = Image.open(path)

            if self.color:
                image = image.convert("RGB")
            else:
                image = image.convert("L")

            return np.asarray(image, dtype=np.float32)

        except Exception as exc:
            raise RuntimeError(
                f"Could not load image '{path}': {exc}"
            ) from exc

    def _check_dimensions(self):
        """Ensure that both images have identical dimensions."""
        if self.image1.shape != self.image2.shape:
            raise ValueError(
                "Images must have identical dimensions. "
                f"Image 1: {self.image1.shape}, "
                f"Image 2: {self.image2.shape}"
            )

    @staticmethod
    def _clip(image):
        """Clip image values to the valid 8-bit range."""
        return np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def _normalize(image):
        """
        Normalize a reconstructed image to 0..255.

        Each channel is normalized independently for RGB images.
        """
        image = np.asarray(image, dtype=np.float32)

        if image.ndim == 2:
            minimum = image.min()
            maximum = image.max()

            if maximum - minimum < 1e-12:
                return np.zeros_like(image, dtype=np.uint8)

            normalized = (
                (image - minimum)
                / (maximum - minimum)
                * 255.0
            )

            return np.clip(
                normalized, 0, 255
            ).astype(np.uint8)

        result = np.empty_like(image)

        for channel in range(image.shape[2]):
            data = image[..., channel]

            minimum = data.min()
            maximum = data.max()

            if maximum - minimum < 1e-12:
                result[..., channel] = 0
            else:
                result[..., channel] = (
                    (data - minimum)
                    / (maximum - minimum)
                    * 255.0
                )

        return np.clip(result, 0, 255).astype(np.uint8)

    def reconstruct(self, magnitude_source, phase_source):
        """
        Reconstruct an image from selected Fourier sources.

        Parameters
        ----------
        magnitude_source : int
            1 or 2.
        phase_source : int
            1 or 2.

        Returns
        -------
        numpy.ndarray
            Reconstructed uint8 image.
        """
        if magnitude_source == 1:
            magnitude = self.fourier1.magnitude
        else:
            magnitude = self.fourier2.magnitude

        if phase_source == 1:
            phase = self.fourier1.phase
        else:
            phase = self.fourier2.phase

        result = self.fourier1.reconstruct(
            magnitude,
            phase
        )

        return self._normalize(result)

    def blend(self, alpha):
        """
        Blend the two original images in image space.

        Parameters
        ----------
        alpha : float
            Blend amount from 0.0 to 1.0.

        Returns
        -------
        numpy.ndarray
            Blended uint8 image.
        """
        alpha = float(np.clip(alpha, 0.0, 1.0))

        result = (
            (1.0 - alpha) * self.image1
            + alpha * self.image2
        )

        return self._clip(result)

    def create_gradient(self, steps=12):
        """
        Create a sequence of linearly blended images.

        Parameters
        ----------
        steps : int
            Number of generated images.

        Returns
        -------
        list
            List of uint8 images.
        """
        if steps < 2:
            steps = 2

        return [
            self.blend(alpha)
            for alpha in np.linspace(0.0, 1.0, steps)
        ]


# ----------------------------------------------------------------------
# Image adjustments
# ----------------------------------------------------------------------

class ImageAdjustments:
    """Apply luminance, vibrance and saturation adjustments."""

    @staticmethod
    def apply(
        image,
        luminance=0.0,
        vibrance=0.0,
        saturation=0.0
    ):
        """
        Apply luminance, vibrance and saturation.

        Parameters
        ----------
        image : numpy.ndarray
            RGB or grayscale uint8 image.
        luminance : float
            Adjustment from -0.30 to +0.30.
        vibrance : float
            Adjustment from -0.30 to +0.30.
        saturation : float
            Adjustment from -0.30 to +0.30.

        Returns
        -------
        numpy.ndarray
            Adjusted uint8 image.
        """
        if image.ndim == 2:
            image = np.stack(
                [image, image, image],
                axis=-1
            )

        rgb = image.astype(np.float32) / 255.0

        hsv = ImageAdjustments._rgb_to_hsv(rgb)

        h = hsv[..., 0]
        s = hsv[..., 1]
        v = hsv[..., 2]

        # --------------------------------------------------------------
        # Saturation
        # --------------------------------------------------------------

        if saturation >= 0.0:
            s += saturation * (1.0 - s)
        else:
            s *= 1.0 + saturation

        # --------------------------------------------------------------
        # Vibrance
        #
        # Positive vibrance affects weakly saturated colours more.
        # Negative vibrance reduces saturation without destroying
        # already weak colours as aggressively as saturation does.
        # --------------------------------------------------------------

        if vibrance >= 0.0:
            s += (
                vibrance
                * (1.0 - s)
                * (1.0 - s)
            )
        else:
            s *= (
                1.0
                + vibrance * (0.5 + 0.5 * s)
            )

        s = np.clip(s, 0.0, 1.0)

        # --------------------------------------------------------------
        # Luminance / brightness
        # --------------------------------------------------------------

        if luminance >= 0.0:
            v += luminance * (1.0 - v)
        else:
            v *= 1.0 + luminance

        v = np.clip(v, 0.0, 1.0)

        hsv[..., 1] = s
        hsv[..., 2] = v

        result = ImageAdjustments._hsv_to_rgb(hsv)

        return np.clip(
            result * 255.0,
            0,
            255
        ).astype(np.uint8)

    @staticmethod
    def _rgb_to_hsv(rgb):
        """Convert an RGB image in the range 0..1 to HSV."""
        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]

        maximum = np.max(rgb, axis=-1)
        minimum = np.min(rgb, axis=-1)
        delta = maximum - minimum

        h = np.zeros_like(maximum)
        s = np.zeros_like(maximum)
        v = maximum

        nonzero = maximum > 0

        s[nonzero] = (
            delta[nonzero]
            / maximum[nonzero]
        )

        mask = delta > 1e-10

        rmask = mask & (maximum == r)
        gmask = mask & (maximum == g)
        bmask = mask & (maximum == b)

        h[rmask] = (
            (g[rmask] - b[rmask])
            / delta[rmask]
        ) % 6.0

        h[gmask] = (
            (b[gmask] - r[gmask])
            / delta[gmask]
        ) + 2.0

        h[bmask] = (
            (r[bmask] - g[bmask])
            / delta[bmask]
        ) + 4.0

        h /= 6.0

        return np.stack(
            (h, s, v),
            axis=-1
        )

    @staticmethod
    def _hsv_to_rgb(hsv):
        """Convert an HSV image to RGB in the range 0..1."""
        h = hsv[..., 0] * 6.0
        s = hsv[..., 1]
        v = hsv[..., 2]

        i = np.floor(h).astype(np.int32)
        f = h - i

        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (
            1.0 - s * (1.0 - f)
        )

        i %= 6

        result = np.empty_like(hsv)

        masks = [
            i == 0,
            i == 1,
            i == 2,
            i == 3,
            i == 4,
            i == 5
        ]

        result[masks[0]] = np.stack(
            (
                v[masks[0]],
                t[masks[0]],
                p[masks[0]]
            ),
            axis=-1
        )

        result[masks[1]] = np.stack(
            (
                q[masks[1]],
                v[masks[1]],
                p[masks[1]]
            ),
            axis=-1
        )

        result[masks[2]] = np.stack(
            (
                p[masks[2]],
                v[masks[2]],
                t[masks[2]]
            ),
            axis=-1
        )

        result[masks[3]] = np.stack(
            (
                p[masks[3]],
                q[masks[3]],
                v[masks[3]]
            ),
            axis=-1
        )

        result[masks[4]] = np.stack(
            (
                t[masks[4]],
                p[masks[4]],
                v[masks[4]]
            ),
            axis=-1
        )

        result[masks[5]] = np.stack(
            (
                v[masks[5]],
                p[masks[5]],
                q[masks[5]]
            ),
            axis=-1
        )

        return result


# ----------------------------------------------------------------------
# Pygame button
# ----------------------------------------------------------------------

class PygameButton:
    """Clickable Pygame button."""

    def __init__(
        self,
        x,
        y,
        width,
        height,
        text,
        callback
    ):
        """
        Initialize a button.

        Parameters
        ----------
        x, y : int
            Position.
        width, height : int
            Button dimensions.
        text : str
            Button label.
        callback : callable
            Function called when the button is clicked.
        """
        self.rect = pygame.Rect(
            x,
            y,
            width,
            height
        )

        self.text = text
        self.callback = callback

        self.active = False

        self.font = pygame.font.SysFont(
            None,
            20
        )

    def draw(self, screen):
        """Draw the button."""
        if self.active:
            background = (60, 110, 180)
        else:
            background = (65, 65, 65)

        pygame.draw.rect(
            screen,
            background,
            self.rect,
            border_radius=6
        )

        pygame.draw.rect(
            screen,
            (130, 130, 130),
            self.rect,
            width=1,
            border_radius=6
        )

        text_surface = self.font.render(
            self.text,
            True,
            (240, 240, 240)
        )

        text_rect = text_surface.get_rect(
            center=self.rect.center
        )

        screen.blit(
            text_surface,
            text_rect
        )

    def handle_event(self, event):
        """Process mouse events."""
        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        ):
            self.callback()
            return True

        return False


# ----------------------------------------------------------------------
# Pygame slider
# ----------------------------------------------------------------------

class PygameSlider:
    """Interactive horizontal slider."""

    def __init__(
        self,
        x,
        y,
        width,
        label,
        callback
    ):
        """
        Initialize a slider.

        Parameters
        ----------
        x, y : int
            Position of the slider track.
        width : int
            Track width.
        label : str
            Slider label.
        callback : callable
            Called with a value in the range -0.30..+0.30.
        """
        self.x = x
        self.y = y
        self.width = width
        self.label = label
        self.callback = callback

        self.minimum = -30
        self.maximum = 30
        self.value = 0

        self.height = 8
        self.knob_radius = 9

        self.dragging = False

        self.font = pygame.font.SysFont(
            None,
            20
        )

    def _value_from_x(self, x):
        """Convert a screen x-position to a slider value."""
        x = max(
            self.x,
            min(
                self.x + self.width,
                x
            )
        )

        fraction = (
            (x - self.x)
            / self.width
        )

        return int(round(
            self.minimum
            + fraction
            * (
                self.maximum
                - self.minimum
            )
        ))

    def _x_from_value(self):
        """Convert the current value to a screen x-position."""
        fraction = (
            self.value - self.minimum
        ) / (
            self.maximum
            - self.minimum
        )

        return (
            self.x
            + fraction * self.width
        )

    def set_value(self, value):
        """Set the slider value and call the callback."""
        self.value = max(
            self.minimum,
            min(
                self.maximum,
                int(value)
            )
        )

        self.callback(
            self.value / 100.0
        )

    def draw(self, screen):
        """Draw the slider and its current value."""
        label = self.font.render(
            f"{self.label}: {self.value:+d}%",
            True,
            (225, 225, 225)
        )

        screen.blit(
            label,
            (
                self.x,
                self.y - 25
            )
        )

        # Track
        pygame.draw.rect(
            screen,
            (75, 75, 75),
            (
                self.x,
                self.y,
                self.width,
                self.height
            ),
            border_radius=4
        )

        # Zero marker
        center_x = (
            self.x
            + self.width / 2
        )

        pygame.draw.line(
            screen,
            (150, 150, 150),
            (
                int(center_x),
                self.y - 5
            ),
            (
                int(center_x),
                self.y + self.height + 5
            ),
            1
        )

        # Slider knob
        knob_x = self._x_from_value()

        pygame.draw.circle(
            screen,
            (225, 225, 225),
            (
                int(knob_x),
                self.y + self.height // 2
            ),
            self.knob_radius
        )

    def handle_event(self, event):
        """Handle mouse interaction."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:

                rect = pygame.Rect(
                    self.x - 10,
                    self.y - 15,
                    self.width + 20,
                    self.height + 30
                )

                if rect.collidepoint(event.pos):
                    self.dragging = True

                    self.set_value(
                        self._value_from_x(
                            event.pos[0]
                        )
                    )

                    return True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False

        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self.set_value(
                    self._value_from_x(
                        event.pos[0]
                    )
                )

                return True

        return False


# ----------------------------------------------------------------------
# Main viewer
# ----------------------------------------------------------------------

class FourierViewer:
    """Interactive Pygame viewer for Fourier image combinations."""

    def __init__(
        self,
        processor,
        width=1400,
        height=800
    ):
        """
        Initialize the viewer.

        Parameters
        ----------
        processor : ImageProcessor
            Image processor.
        width : int
            Window width.
        height : int
            Window height.
        """
        pygame.init()

        self.processor = processor

        self.width = width
        self.height = height

        self.screen = pygame.display.set_mode(
            (width, height)
        )

        pygame.display.set_caption(
            "Fourier Image Viewer"
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(
            None,
            22
        )

        self.small_font = pygame.font.SysFont(
            None,
            18
        )

        self.results = []
        self.result_index = 0

        self.mode = "fourier"

        # Default:
        # magnitude = image 1
        # phase = image 1
        self.magnitude_source = 1
        self.phase_source = 1

        # Image adjustments
        self.luminance = 0.0
        self.vibrance = 0.0
        self.saturation = 0.0

        self.buttons = []

        self._create_buttons()
        self._create_sliders()

        self.create_fourier_result()

    # ------------------------------------------------------------------

    def _create_buttons(self):
        """Create the Pygame control buttons."""
        button_y = self.height - 165

        button_width = 130
        button_height = 38
        gap = 10

        x = self.width - (
            6 * button_width
            + 5 * gap
            + 30
        )

        self.buttons = [
            PygameButton(
                x,
                button_y,
                button_width,
                button_height,
                "PHASE 1",
                lambda: self.set_phase_source(1)
            ),

            PygameButton(
                x + (button_width + gap),
                button_y,
                button_width,
                button_height,
                "PHASE 2",
                lambda: self.set_phase_source(2)
            ),

            PygameButton(
                x + 2 * (button_width + gap),
                button_y,
                button_width,
                button_height,
                "MAG 1",
                lambda: self.set_magnitude_source(1)
            ),

            PygameButton(
                x + 3 * (button_width + gap),
                button_y,
                button_width,
                button_height,
                "MAG 2",
                lambda: self.set_magnitude_source(2)
            ),

            PygameButton(
                x + 4 * (button_width + gap),
                button_y,
                button_width,
                button_height,
                "CREATE",
                self.create_fourier_result
            ),

            PygameButton(
                x + 5 * (button_width + gap),
                button_y,
                button_width,
                button_height,
                "BLEND",
                self.create_blend_results
            )
        ]

        self._update_button_states()

    # ------------------------------------------------------------------

    def _create_sliders(self):
        """Create luminance, vibrance and saturation sliders."""
        x = 40
        width = 280

        base_y = self.height - 130

        self.sliders = [
            PygameSlider(
                x,
                base_y,
                width,
                "Luminance",
                self.set_luminance
            ),

            PygameSlider(
                x,
                base_y + 45,
                width,
                "Vibrance",
                self.set_vibrance
            ),

            PygameSlider(
                x,
                base_y + 90,
                width,
                "Saturation",
                self.set_saturation
            )
        ]

    # ------------------------------------------------------------------

    def _update_button_states(self):
        """Update the active state of Fourier source buttons."""
        if len(self.buttons) < 6:
            return

        self.buttons[0].active = (
            self.phase_source == 1
        )

        self.buttons[1].active = (
            self.phase_source == 2
        )

        self.buttons[2].active = (
            self.magnitude_source == 1
        )

        self.buttons[3].active = (
            self.magnitude_source == 2
        )

        self.buttons[4].active = (
            self.mode == "fourier"
        )

        self.buttons[5].active = (
            self.mode == "blend"
        )

    # ------------------------------------------------------------------

    def set_phase_source(self, source):
        """Select the Fourier phase source."""
        self.phase_source = source
        self.mode = "fourier"

        self._update_button_states()

    # ------------------------------------------------------------------

    def set_magnitude_source(self, source):
        """Select the Fourier magnitude source."""
        self.magnitude_source = source
        self.mode = "fourier"

        self._update_button_states()

    # ------------------------------------------------------------------

    def set_luminance(self, value):
        """Set luminance adjustment."""
        self.luminance = value
        self._refresh_adjusted_result()

    # ------------------------------------------------------------------

    def set_vibrance(self, value):
        """Set vibrance adjustment."""
        self.vibrance = value
        self._refresh_adjusted_result()

    # ------------------------------------------------------------------

    def set_saturation(self, value):
        """Set saturation adjustment."""
        self.saturation = value
        self._refresh_adjusted_result()

    # ------------------------------------------------------------------

    def _refresh_adjusted_result(self):
        """Apply current image adjustments to the displayed result."""
        if not self.results:
            return

        source = self.results[
            self.result_index
        ]

        self.display_result = ImageAdjustments.apply(
            source,
            luminance=self.luminance,
            vibrance=self.vibrance,
            saturation=self.saturation
        )

    # ------------------------------------------------------------------

    def create_fourier_result(self):
        """Create the currently selected Fourier combination."""
        result = self.processor.reconstruct(
            self.magnitude_source,
            self.phase_source
        )

        self.results = [result]
        self.result_index = 0

        self.mode = "fourier"

        self._update_button_states()
        self._refresh_adjusted_result()

    # ------------------------------------------------------------------

    def create_blend_results(self):
        """Create the 12-step image-space blend sequence."""
        self.results = (
            self.processor.create_gradient(
                steps=12
            )
        )

        self.result_index = 0
        self.mode = "blend"

        self._update_button_states()
        self._refresh_adjusted_result()

    # ------------------------------------------------------------------

    def next_result(self):
        """Move to the next generated result."""
        if not self.results:
            return

        self.result_index = (
            self.result_index + 1
        ) % len(self.results)

        self._refresh_adjusted_result()

    # ------------------------------------------------------------------

    def previous_result(self):
        """Move to the previous generated result."""
        if not self.results:
            return

        self.result_index = (
            self.result_index - 1
        ) % len(self.results)

        self._refresh_adjusted_result()

    # ------------------------------------------------------------------

    def _fit_image(self, image, area):
        """Scale an image to fit inside a rectangular display area."""
        if image.ndim == 2:
            image = np.stack(
                [image, image, image],
                axis=-1
            )

        height, width = image.shape[:2]

        area_width = area.width
        area_height = area.height

        scale = min(
            area_width / width,
            area_height / height
        )

        new_width = max(
            1,
            int(width * scale)
        )

        new_height = max(
            1,
            int(height * scale)
        )

        pil_image = Image.fromarray(
            image.astype(np.uint8)
        )

        pil_image = pil_image.resize(
            (
                new_width,
                new_height
            ),
            Image.Resampling.LANCZOS
        )

        return np.asarray(
            pil_image
        )

    # ------------------------------------------------------------------

    def _draw_image(
        self,
        screen,
        image,
        area,
        title
    ):
        """Draw an image centered inside a display area."""
        pygame.draw.rect(
            screen,
            (25, 25, 25),
            area
        )

        fitted = self._fit_image(
            image,
            area
        )

        if fitted.ndim == 2:
            fitted = np.stack(
                [fitted, fitted, fitted],
                axis=-1
            )

        surface = pygame.surfarray.make_surface(
            np.transpose(
                fitted,
                (1, 0, 2)
            )
        )

        x = (
            area.x
            + (area.width - surface.get_width())
            // 2
        )

        y = (
            area.y
            + (area.height - surface.get_height())
            // 2
        )

        screen.blit(
            surface,
            (x, y)
        )

        pygame.draw.rect(
            screen,
            (100, 100, 100),
            area,
            width=1
        )

        title_surface = self.font.render(
            title,
            True,
            (235, 235, 235)
        )

        screen.blit(
            title_surface,
            (
                area.x + 8,
                area.y + 8
            )
        )

    # ------------------------------------------------------------------

    def _draw_status(self, screen):
        """Draw current Fourier and adjustment information."""
        if self.mode == "fourier":
            text = (
                f"Magnitude {self.magnitude_source}  |  "
                f"Phase {self.phase_source}"
            )
        else:
            alpha = 0.0

            if len(self.results) > 1:
                alpha = (
                    self.result_index
                    / (len(self.results) - 1)
                )

            text = (
                f"BLEND  "
                f"{self.result_index + 1}/"
                f"{len(self.results)}  "
                f"alpha={alpha:.2f}"
            )

        surface = self.small_font.render(
            text,
            True,
            (210, 210, 210)
        )

        screen.blit(
            surface,
            (20, 15)
        )

        adjustment = (
            f"Lum {self.luminance * 100:+.0f}%   "
            f"Vib {self.vibrance * 100:+.0f}%   "
            f"Sat {self.saturation * 100:+.0f}%"
        )

        surface = self.small_font.render(
            adjustment,
            True,
            (180, 180, 180)
        )

        screen.blit(
            surface,
            (20, 38)
        )

    # ------------------------------------------------------------------

    def _draw_navigation(self, screen):
        """Draw navigation instructions."""
        text = (
            "← / →  navigate    |    ESC  quit"
        )

        surface = self.small_font.render(
            text,
            True,
            (170, 170, 170)
        )

        rect = surface.get_rect()

        rect.topright = (
            self.width - 20,
            15
        )

        screen.blit(
            surface,
            rect
        )

    # ------------------------------------------------------------------

    def draw(self):
        """Draw the complete application window."""
        self.screen.fill(
            (15, 15, 15)
        )

        margin = 20
        top = 70

        bottom = self.height - 190

        panel_height = (
            bottom - top
        )

        gap = 15

        panel_width = (
            self.width
            - 2 * margin
            - 2 * gap
        ) // 3

        area1 = pygame.Rect(
            margin,
            top,
            panel_width,
            panel_height
        )

        area2 = pygame.Rect(
            margin + panel_width + gap,
            top,
            panel_width,
            panel_height
        )

        area3 = pygame.Rect(
            margin
            + 2 * (panel_width + gap),
            top,
            panel_width,
            panel_height
        )

        self._draw_image(
            self.screen,
            self.processor.image1,
            area1,
            "ORIGINAL 1"
        )

        self._draw_image(
            self.screen,
            self.processor.image2,
            area2,
            "ORIGINAL 2"
        )

        self._draw_image(
            self.screen,
            self.display_result,
            area3,
            "RESULT"
        )

        self._draw_status(
            self.screen
        )

        self._draw_navigation(
            self.screen
        )

        # Sliders
        for slider in self.sliders:
            slider.draw(self.screen)

        # Buttons
        for button in self.buttons:
            button.draw(self.screen)

        pygame.display.flip()

    # ------------------------------------------------------------------

    def handle_event(self, event):
        """Handle keyboard, mouse and application events."""
        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:
                return False

            if event.key == pygame.K_RIGHT:
                self.next_result()

            elif event.key == pygame.K_LEFT:
                self.previous_result()

        for slider in self.sliders:
            if slider.handle_event(event):
                return True

        for button in self.buttons:
            if button.handle_event(event):
                return True

        return True

    # ------------------------------------------------------------------

    def run(self):
        """Run the Pygame event loop."""
        running = True

        while running:

            for event in pygame.event.get():
                running = self.handle_event(
                    event
                )

                if not running:
                    break

            self.draw()

            self.clock.tick(60)

        pygame.quit()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    """Create the application and start the viewer."""
    args = parse_args()

    try:
        processor = ImageProcessor(
            args.image1,
            args.image2,
            color=args.color
        )

        viewer = FourierViewer(
            processor,
            width=args.width,
            height=args.height
        )

        viewer.run()

    except Exception as exc:
        pygame.quit()

        print(
            f"Error: {exc}",
            file=sys.stderr
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
