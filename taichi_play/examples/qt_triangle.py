"""PyQt6 window with a Taichi-rendered triangle with rotatable vertex colors."""

import sys
import numpy as np
import taichi as ti
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap


@ti.data_oriented
class TriangleRenderer:
    """Handles all Taichi rendering logic for the colored triangle."""

    def __init__(self, width=800, height=600):
        """Initialize the Taichi renderer.

        Args:
            width: Window width in pixels
            height: Window height in pixels
        """
        ti.init(arch=ti.gpu)

        self.width = width
        self.height = height

        # Field to store the rendered image (RGBA)
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

        # Triangle vertices (normalized coordinates: -1 to 1)
        self.vertices = ti.Vector.field(2, dtype=ti.f32, shape=3)
        self.vertices[0] = [0.0, 0.6]   # Top
        self.vertices[1] = [-0.5, -0.3]  # Bottom left
        self.vertices[2] = [0.5, -0.3]   # Bottom right

        # Vertex colors (RGB)
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=3)
        self.colors[0] = [1.0, 0.0, 0.0]  # Red
        self.colors[1] = [0.0, 1.0, 0.0]  # Green
        self.colors[2] = [0.0, 0.0, 1.0]  # Blue

    @ti.func
    def barycentric(self, p, v0, v1, v2):
        """Calculate barycentric coordinates for point p in triangle (v0, v1, v2).

        Args:
            p: Point to test
            v0, v1, v2: Triangle vertices

        Returns:
            vec3 of barycentric coordinates (w0, w1, w2)
        """
        e0 = v1 - v0
        e1 = v2 - v0
        e2 = p - v0

        d00 = e0.dot(e0)
        d01 = e0.dot(e1)
        d11 = e1.dot(e1)
        d20 = e2.dot(e0)
        d21 = e2.dot(e1)

        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        return ti.Vector([u, v, w])

    @ti.kernel
    def render(self):
        """Render the triangle with interpolated vertex colors."""
        # Clear background to white
        for i, j in self.pixels:
            self.pixels[i, j] = ti.Vector([1.0, 1.0, 1.0])

        # Get vertices
        v0 = self.vertices[0]
        v1 = self.vertices[1]
        v2 = self.vertices[2]

        # Render triangle
        for i, j in self.pixels:
            # Convert pixel coordinates to normalized device coordinates (-1 to 1)
            x = (i / self.width) * 2.0 - 1.0
            y = (j / self.height) * 2.0 - 1.0
            p = ti.Vector([x, y])

            # Calculate barycentric coordinates
            bary = self.barycentric(p, v0, v1, v2)

            # Check if point is inside triangle
            if bary[0] >= 0.0 and bary[1] >= 0.0 and bary[2] >= 0.0:
                # Interpolate color using barycentric coordinates
                color = (
                    bary[0] * self.colors[0] +
                    bary[1] * self.colors[1] +
                    bary[2] * self.colors[2]
                )
                self.pixels[i, j] = color

    def rotate_colors(self):
        """Rotate the vertex colors clockwise (0→1, 1→2, 2→0)."""
        # Store colors in temporary array
        temp_colors = [
            [self.colors[0][0], self.colors[0][1], self.colors[0][2]],
            [self.colors[1][0], self.colors[1][1], self.colors[1][2]],
            [self.colors[2][0], self.colors[2][1], self.colors[2][2]],
        ]

        # Rotate: color[0] <- color[2], color[1] <- color[0], color[2] <- color[1]
        self.colors[0] = temp_colors[2]
        self.colors[1] = temp_colors[0]
        self.colors[2] = temp_colors[1]

    def get_frame(self):
        """Get the current rendered frame as a numpy array.

        Returns:
            numpy array of shape (height, width, 3) with uint8 values
        """
        # Convert Taichi field to numpy array
        img = self.pixels.to_numpy()

        # Convert from float32 [0,1] to uint8 [0,255]
        img = (img * 255).astype(np.uint8)

        # Flip vertically (Taichi uses bottom-left origin, Qt uses top-left)
        img = np.flip(img, axis=1)

        return img


class MainWindow(QMainWindow):
    """PyQt6 main window for displaying the triangle."""

    def __init__(self):
        """Initialize the Qt window and UI components."""
        super().__init__()

        self.setWindowTitle("Taichi + PyQt6: Colored Triangle")

        # Create the Taichi renderer
        self.renderer = TriangleRenderer(width=800, height=600)

        # Set up UI
        self.init_ui()

        # Set up timer for rendering updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)  # ~60 FPS

    def init_ui(self):
        """Initialize the user interface components."""
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Image display label
        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        # Rotate button
        self.rotate_button = QPushButton("Rotate Colors")
        self.rotate_button.clicked.connect(self.on_rotate_clicked)
        layout.addWidget(self.rotate_button)

        # Render initial frame
        self.update_frame()

    def on_rotate_clicked(self):
        """Handle rotate button click."""
        self.renderer.rotate_colors()

    def update_frame(self):
        """Update the displayed frame from the renderer."""
        # Render the current frame
        self.renderer.render()

        # Get the frame as numpy array
        img_array = self.renderer.get_frame()

        # Convert to QImage
        height, width, channels = img_array.shape
        bytes_per_line = channels * width

        # Ensure array is contiguous and convert to bytes
        img_array = np.ascontiguousarray(img_array)
        q_image = QImage(
            img_array.tobytes(),
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )

        # Display in label
        self.image_label.setPixmap(QPixmap.fromImage(q_image))


def run():
    """Run the PyQt6 triangle application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
