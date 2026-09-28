#!/usr/bin/env python3

"""
Fourier-domain image analysis.

The program can:

* calculate 2-D FFTs of two images
* exchange magnitude and phase between images
* process grayscale or RGB images
* create ordinary image blends
* create a 12-step gradient
* display image histograms
* display Fourier magnitude and phase
* report dominant spatial frequencies

Example:

    python fft_images.py image1.jpg image2.jpg

Color:

    python fft_images.py image1.jpg image2.jpg --color

Gradient:

    python fft_images.py image1.jpg image2.jpg --blend --gradient

Histogram:

    python fft_images.py image1.jpg image2.jpg --hist
"""


import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# Command line
# ============================================================

def parse_args():
    """
    Parse and validate command-line arguments.

    Returns
    -------
    argparse.Namespace
        Namespace containing all command-line options.

    Notes
    -----
    The positional arguments are the two input images.
    The ``--color`` option controls RGB versus grayscale
    processing. The remaining options control which analyses
    and output operations are performed.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Fourier image analysis, magnitude/phase "
            "exchange and image blending."
        )
    )

    parser.add_argument(
        "original1",
        help="path to the first input image"
    )

    parser.add_argument(
        "original2",
        help="path to the second input image"
    )

    parser.add_argument(
        "--color",
        action="store_true",
        help=(
            "process images as RGB; without this option "
            "images are converted to grayscale"
        )
    )

    parser.add_argument(
        "--blend",
        action="store_true",
        help=(
            "enable ordinary linear image blending; "
            "without --gradient a single 50%% blend is created"
        )
    )

    parser.add_argument(
        "--gradient",
        action="store_true",
        help=(
            "create 12 images interpolating from original1 "
            "to original2; requires --blend"
        )
    )

    parser.add_argument(
        "--hist",
        action="store_true",
        help=(
            "display histograms of original1 and original2"
        )
    )

    parser.add_argument(
        "--fft",
        action="store_true",
        help=(
            "display the original images together with "
            "their Fourier magnitude and phase"
        )
    )

    parser.add_argument(
        "--frequencies",
        type=int,
        default=20,
        metavar="N",
        help=(
            "number of dominant spatial frequencies to "
            "report; default: 20"
        )
    )

    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help=(
            "do not create the two magnitude/phase "
            "hybrid images"
        )
    )

    args = parser.parse_args()

    if args.gradient and not args.blend:
        parser.error(
            "--gradient requires --blend"
        )

    if args.frequencies < 1:
        parser.error(
            "--frequencies must be greater than zero"
        )

    return args


# ============================================================
# FourierImage
# ============================================================

class FourierImage:
    """
    Represent an image in the two-dimensional Fourier domain.

    The class calculates and stores the complex FFT together
    with its magnitude and phase. Grayscale images are treated
    as one channel. RGB images are transformed independently
    in each of their three color channels.

    Parameters
    ----------
    image : numpy.ndarray
        Input image as a two-dimensional grayscale array or
        a three-dimensional RGB array.

    Attributes
    ----------
    image : numpy.ndarray
        Original image converted to floating-point values.
    channels : int
        Number of image channels, either 1 or 3.
    fft : numpy.ndarray
        Shifted complex Fourier transform.
    magnitude : numpy.ndarray
        Absolute value of the Fourier transform.
    phase : numpy.ndarray
        Phase angle of the Fourier transform.
    """

    def __init__(self, image):
        """
        Initialize a Fourier representation of an image.

        Parameters
        ----------
        image : numpy.ndarray
            Grayscale or RGB image.
        """

        self.image = image.astype(np.float64)

        if self.image.ndim == 2:
            self.channels = 1

        elif (
            self.image.ndim == 3
            and self.image.shape[2] == 3
        ):
            self.channels = 3

        else:
            raise ValueError(
                "Image must be grayscale or RGB."
            )

        self.fft = self._calculate_fft()

        self.magnitude = np.abs(self.fft)
        self.phase = np.angle(self.fft)

    def _calculate_fft(self):
        """
        Calculate the centered two-dimensional FFT.

        Returns
        -------
        numpy.ndarray
            Complex Fourier transform with the zero-frequency
            component shifted to the center.

        Notes
        -----
        For RGB images the FFT is calculated independently
        for the red, green and blue channels.
        """

        if self.channels == 1:

            return np.fft.fftshift(
                np.fft.fft2(self.image)
            )

        result = np.empty(
            self.image.shape,
            dtype=np.complex128
        )

        for channel in range(3):

            result[:, :, channel] = np.fft.fftshift(
                np.fft.fft2(
                    self.image[:, :, channel]
                )
            )

        return result

    def reconstruct(self, magnitude, phase):
        """
        Reconstruct an image from Fourier magnitude and phase.

        Parameters
        ----------
        magnitude : numpy.ndarray
            Fourier magnitude.
        phase : numpy.ndarray
            Fourier phase in radians.

        Returns
        -------
        numpy.ndarray
            Real-valued reconstructed image.

        Notes
        -----
        The inverse FFT is performed independently for every
        RGB channel when processing a color image.
        """

        spectrum = (
            magnitude
            * np.exp(1j * phase)
        )

        if self.channels == 1:

            result = np.fft.ifft2(
                np.fft.ifftshift(spectrum)
            )

        else:

            result = np.empty(
                self.image.shape,
                dtype=np.complex128
            )

            for channel in range(3):

                result[:, :, channel] = np.fft.ifft2(
                    np.fft.ifftshift(
                        spectrum[:, :, channel]
                    )
                )

        return np.real(result)


# ============================================================
# ImageProcessor
# ============================================================

class ImageProcessor:
    """
    Process and compare two images using Fourier analysis.

    The class loads two images, constructs their Fourier
    representations and provides operations for:

    * magnitude/phase exchange
    * image blending
    * gradient generation
    * histogram display
    * Fourier visualization
    * dominant-frequency analysis

    Parameters
    ----------
    filename1 : str or pathlib.Path
        Path to the first image.
    filename2 : str or pathlib.Path
        Path to the second image.
    color : bool, optional
        If True, images are loaded as RGB. Otherwise they are
        converted to grayscale.

    Attributes
    ----------
    image1 : numpy.ndarray
        First image.
    image2 : numpy.ndarray
        Second image.
    fft1 : FourierImage
        Fourier representation of the first image.
    fft2 : FourierImage
        Fourier representation of the second image.
    """

    def __init__(
        self,
        filename1,
        filename2,
        color=False
    ):
        """
        Load and initialize the two input images.

        Parameters
        ----------
        filename1 : str or pathlib.Path
            Path to the first image.
        filename2 : str or pathlib.Path
            Path to the second image.
        color : bool, optional
            Load images as RGB when True, otherwise grayscale.
        """

        self.filename1 = Path(filename1)
        self.filename2 = Path(filename2)
        self.color = color

        self.image1 = self._load(
            self.filename1
        )

        self.image2 = self._load(
            self.filename2
        )

        self._check_dimensions()

        self.fft1 = FourierImage(
            self.image1
        )

        self.fft2 = FourierImage(
            self.image2
        )

    # --------------------------------------------------------
    # Image loading
    # --------------------------------------------------------

    def _load(self, filename):
        """
        Load an image from disk.

        Parameters
        ----------
        filename : pathlib.Path
            Path to the image file.

        Returns
        -------
        numpy.ndarray
            Image as a floating-point NumPy array.

        Notes
        -----
        Images are converted to grayscale unless ``color`` was
        requested. RGB images are represented as three channels.
        """

        image = Image.open(filename)

        if self.color:
            image = image.convert("RGB")
        else:
            image = image.convert("L")

        return np.asarray(
            image,
            dtype=np.float64
        )

    def _check_dimensions(self):
        """
        Verify that both images have identical dimensions.

        Raises
        ------
        ValueError
            If the two images do not have identical shapes.

        Notes
        -----
        Identical dimensions are required because Fourier
        magnitude and phase are exchanged pixel-for-pixel.
        """

        if self.image1.shape != self.image2.shape:

            raise ValueError(
                "Images must have identical dimensions:\n"
                f"  image 1: {self.image1.shape}\n"
                f"  image 2: {self.image2.shape}"
            )

    # --------------------------------------------------------
    # Image conversion
    # --------------------------------------------------------

    @staticmethod
    def _clip(image):
        """
        Clip image values to the valid 8-bit range.

        Parameters
        ----------
        image : numpy.ndarray
            Image data.

        Returns
        -------
        numpy.ndarray
            Unsigned 8-bit image with values between 0 and 255.
        """

        return np.clip(
            image,
            0,
            255
        ).astype(np.uint8)

    @staticmethod
    def _normalize(image):
        """
        Normalize an image independently to the range 0...255.

        Parameters
        ----------
        image : numpy.ndarray
            Image data, potentially containing negative values
            or values outside the 8-bit range.

        Returns
        -------
        numpy.ndarray
            Normalized unsigned 8-bit image.

        Notes
        -----
        For RGB images each color channel is normalized
        independently.
        """

        image = np.real(image)

        if image.ndim == 2:

            minimum = image.min()
            maximum = image.max()

            if maximum > minimum:

                image = (
                    (image - minimum)
                    / (maximum - minimum)
                    * 255
                )

            else:

                image = np.zeros_like(image)

        else:

            result = np.empty_like(image)

            for channel in range(
                image.shape[2]
            ):

                data = image[:, :, channel]

                minimum = data.min()
                maximum = data.max()

                if maximum > minimum:

                    result[:, :, channel] = (
                        (data - minimum)
                        / (maximum - minimum)
                        * 255
                    )

                else:

                    result[:, :, channel] = 0

            image = result

        return np.clip(
            image,
            0,
            255
        ).astype(np.uint8)

    def save(
        self,
        image,
        filename,
        normalize=False
    ):
        """
        Save an image to disk.

        Parameters
        ----------
        image : numpy.ndarray
            Image to save.
        filename : str or pathlib.Path
            Output filename.
        normalize : bool, optional
            If True, normalize the image independently to
            0...255. If False, simply clip to that range.
        """

        if normalize:
            image = self._normalize(image)
        else:
            image = self._clip(image)

        Image.fromarray(image).save(filename)

        print(f"written: {filename}")

    # --------------------------------------------------------
    # Magnitude / phase exchange
    # --------------------------------------------------------

    def hybrid_images(self):
        """
        Create the two possible magnitude/phase combinations.

        Returns
        -------
        tuple of numpy.ndarray
            Two reconstructed images:

            1. magnitude from image 1 + phase from image 2
            2. magnitude from image 2 + phase from image 1

        Notes
        -----
        This operation is performed independently for every
        RGB channel when color processing is enabled.
        """

        magnitude1_phase2 = self.fft1.reconstruct(
            self.fft1.magnitude,
            self.fft2.phase
        )

        magnitude2_phase1 = self.fft1.reconstruct(
            self.fft2.magnitude,
            self.fft1.phase
        )

        return (
            magnitude1_phase2,
            magnitude2_phase1
        )

    def save_hybrids(self):
        """
        Create and save both magnitude/phase hybrid images.

        The generated files are:

        * ``magnitude1_phase2.png``
        * ``magnitude2_phase1.png``
        """

        image12, image21 = (
            self.hybrid_images()
        )

        self.save(
            image12,
            "magnitude1_phase2.png",
            normalize=True
        )

        self.save(
            image21,
            "magnitude2_phase1.png",
            normalize=True
        )

    # --------------------------------------------------------
    # Image blending
    # --------------------------------------------------------

    def blend(self, alpha):
        """
        Linearly interpolate between the two original images.

        Parameters
        ----------
        alpha : float
            Interpolation parameter.

            ``alpha = 0`` gives image 1.
            ``alpha = 1`` gives image 2.
            ``alpha = 0.5`` gives an equal blend.

        Returns
        -------
        numpy.ndarray
            Blended image.
        """

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "alpha must be between 0 and 1"
            )

        return (
            (1.0 - alpha) * self.image1
            + alpha * self.image2
        )

    def gradient(self, steps=12):
        """
        Generate a sequence interpolating between the images.

        Parameters
        ----------
        steps : int, optional
            Number of images in the sequence. The default is 12.

        Returns
        -------
        list of numpy.ndarray
            Images starting with image 1 and ending with image 2.
        """

        if steps < 2:
            raise ValueError(
                "steps must be at least 2"
            )

        alphas = np.linspace(
            0.0,
            1.0,
            steps
        )

        return [
            self.blend(alpha)
            for alpha in alphas
        ]

    def save_gradient(self, steps=12):
        """
        Generate and save an image interpolation sequence.

        Parameters
        ----------
        steps : int, optional
            Number of images to generate. Default is 12.

        Notes
        -----
        The files are named:

            gradient_01_of_12.png
            gradient_02_of_12.png
            ...
            gradient_12_of_12.png
        """

        images = self.gradient(
            steps
        )

        for index, image in enumerate(
            images,
            start=1
        ):

            filename = (
                f"gradient_{index:02d}"
                f"_of_{steps:02d}.png"
            )

            self.save(
                image,
                filename
            )

    # --------------------------------------------------------
    # Histogram
    # --------------------------------------------------------

    def show_histograms(self):
        """
        Display pixel-value histograms for both images.

        For grayscale images a single histogram contains the
        pixel distributions of both images.

        For RGB images three histograms are displayed, one each
        for red, green and blue.
        """

        if not self.color:

            plt.figure(
                figsize=(10, 5)
            )

            plt.hist(
                self.image1.ravel(),
                bins=256,
                alpha=0.5,
                label="Original 1"
            )

            plt.hist(
                self.image2.ravel(),
                bins=256,
                alpha=0.5,
                label="Original 2"
            )

            plt.xlabel("Pixel value")
            plt.ylabel("Pixels")
            plt.title("Image histograms")
            plt.legend()
            plt.tight_layout()
            plt.show()

            return

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(10, 9)
        )

        names = (
            "Red",
            "Green",
            "Blue"
        )

        for channel, (
            ax,
            name
        ) in enumerate(
            zip(axes, names)
        ):

            ax.hist(
                self.image1[
                    :, :, channel
                ].ravel(),
                bins=256,
                alpha=0.5,
                label=(
                    f"Original 1 - {name}"
                )
            )

            ax.hist(
                self.image2[
                    :, :, channel
                ].ravel(),
                bins=256,
                alpha=0.5,
                label=(
                    f"Original 2 - {name}"
                )
            )

            ax.set_xlabel(
                "Pixel value"
            )

            ax.set_ylabel(
                "Pixels"
            )

            ax.legend()

        fig.suptitle(
            "RGB image histograms"
        )

        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # FFT visualization
    # --------------------------------------------------------

    @staticmethod
    def _log_magnitude(magnitude):
        """
        Convert Fourier magnitude to a displayable logarithmic scale.

        Parameters
        ----------
        magnitude : numpy.ndarray
            Fourier magnitude.

        Returns
        -------
        numpy.ndarray
            Logarithmically scaled magnitude suitable for display.

        Notes
        -----
        A logarithmic scale is used because Fourier magnitudes
        normally span several orders of magnitude.
        """

        if magnitude.ndim == 2:
            return np.log1p(
                magnitude
            )

        return np.log1p(
            np.sqrt(
                np.sum(
                    magnitude ** 2,
                    axis=2
                )
            )
        )

    @staticmethod
    def _phase_display(phase):
        """
        Convert phase data to a two-dimensional display image.

        Parameters
        ----------
        phase : numpy.ndarray
            Fourier phase.

        Returns
        -------
        numpy.ndarray
            Two-dimensional phase representation.

        Notes
        -----
        For RGB images the three channel phases are averaged
        only for visualization. The actual Fourier processing
        always retains all three phase channels separately.
        """

        if phase.ndim == 2:
            return phase

        return np.mean(
            phase,
            axis=2
        )

    def show_fft(self):
        """
        Display original images and their Fourier components.

        The figure contains, for each image:

        * original image
        * logarithmic Fourier magnitude
        * Fourier phase

        For RGB images the magnitude and phase displays combine
        the three channels for visualization only.
        """

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(13, 8)
        )

        axes[0, 0].imshow(
            self.image1,
            cmap=(
                None
                if self.color
                else "gray"
            )
        )

        axes[0, 0].set_title(
            "Original 1"
        )

        axes[0, 1].imshow(
            self._log_magnitude(
                self.fft1.magnitude
            ),
            cmap="gray"
        )

        axes[0, 1].set_title(
            "Magnitude 1"
        )

        axes[0, 2].imshow(
            self._phase_display(
                self.fft1.phase
            ),
            cmap="gray"
        )

        axes[0, 2].set_title(
            "Phase 1"
        )

        axes[1, 0].imshow(
            self.image2,
            cmap=(
                None
                if self.color
                else "gray"
            )
        )

        axes[1, 0].set_title(
            "Original 2"
        )

        axes[1, 1].imshow(
            self._log_magnitude(
                self.fft2.magnitude
            ),
            cmap="gray"
        )

        axes[1, 1].set_title(
            "Magnitude 2"
        )

        axes[1, 2].imshow(
            self._phase_display(
                self.fft2.phase
            ),
            cmap="gray"
        )

        axes[1, 2].set_title(
            "Phase 2"
        )

        for ax in axes.flat:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # Dominant frequencies
    # --------------------------------------------------------

    @staticmethod
    def _frequency_table(
        fourier,
        number
    ):
        """
        Calculate the strongest spatial Fourier frequencies.

        Parameters
        ----------
        fourier : FourierImage
            Fourier representation of an image.
        number : int
            Number of frequencies to return.

        Returns
        -------
        list of tuple
            Each tuple contains:

            ``fx``
                Horizontal frequency in cycles/pixel.

            ``fy``
                Vertical frequency in cycles/pixel.

            ``frequency``
                Radial spatial frequency.

            ``wavelength``
                Corresponding wavelength in pixels.

            ``angle``
                Frequency orientation in degrees.

            ``magnitude``
                Fourier magnitude.
        """

        magnitude = fourier.magnitude

        if magnitude.ndim == 3:

            magnitude = np.sqrt(
                np.sum(
                    magnitude ** 2,
                    axis=2
                )
            )

        height, width = (
            magnitude.shape
        )

        fx = np.fft.fftshift(
            np.fft.fftfreq(width)
        )

        fy = np.fft.fftshift(
            np.fft.fftfreq(height)
        )

        FX, FY = np.meshgrid(
            fx,
            fy
        )

        magnitude = magnitude.copy()

        # Remove the DC component.
        magnitude[
            height // 2,
            width // 2
        ] = 0

        indices = np.argsort(
            magnitude.ravel()
        )[::-1]

        result = []

        for index in indices:

            y, x = np.unravel_index(
                index,
                magnitude.shape
            )

            f_x = FX[y, x]
            f_y = FY[y, x]

            frequency = np.hypot(
                f_x,
                f_y
            )

            if frequency == 0:
                continue

            wavelength = (
                1.0 / frequency
            )

            angle = np.degrees(
                np.arctan2(
                    f_y,
                    f_x
                )
            )

            result.append(
                (
                    f_x,
                    f_y,
                    frequency,
                    wavelength,
                    angle,
                    magnitude[y, x]
                )
            )

            if len(result) >= number:
                break

        return result

    def print_dominant_frequencies(
        self,
        number=20
    ):
        """
        Print the strongest spatial frequencies of both images.

        Parameters
        ----------
        number : int, optional
            Number of frequencies to print for each image.
            Default is 20.

        Notes
        -----
        Frequencies are expressed in cycles per pixel.
        The corresponding wavelength is expressed in pixels.
        """

        if number < 1:
            raise ValueError(
                "number must be greater than zero"
            )

        for name, fourier in (
            ("ORIGINAL 1", self.fft1),
            ("ORIGINAL 2", self.fft2),
        ):

            print()
            print("=" * 88)
            print(
                f"DOMINANT FREQUENCIES — {name}"
            )
            print("=" * 88)

            print(
                f"{'#':>3} "
                f"{'fx':>10} "
                f"{'fy':>10} "
                f"{'frequency':>12} "
                f"{'wavelength':>14} "
                f"{'angle':>10} "
                f"{'magnitude':>15}"
            )

            print("-" * 88)

            frequencies = (
                self._frequency_table(
                    fourier,
                    number
                )
            )

            for index, (
                fx,
                fy,
                frequency,
                wavelength,
                angle,
                magnitude
            ) in enumerate(
                frequencies,
                start=1
            ):

                print(
                    f"{index:3d} "
                    f"{fx:10.5f} "
                    f"{fy:10.5f} "
                    f"{frequency:12.5f} "
                    f"{wavelength:14.2f} "
                    f"{angle:10.2f} "
                    f"{magnitude:15.2f}"
                )


# ============================================================
# Main
# ============================================================

def main():
    """
    Run the Fourier image analysis application.

    Command-line options determine which processing operations
    are performed.
    """

    args = parse_args()

    processor = ImageProcessor(
        args.original1,
        args.original2,
        color=args.color
    )

    if not args.no_hybrid:
        processor.save_hybrids()

    if args.fft:
        processor.show_fft()

    if args.hist:
        processor.show_histograms()

    if args.blend:

        if args.gradient:

            processor.save_gradient(
                steps=12
            )

        else:

            processor.save(
                processor.blend(0.5),
                "blend.png"
            )

    processor.print_dominant_frequencies(
        args.frequencies
    )


if __name__ == "__main__":
    main()
