"""PyQt6 window with a Taichi-rendered triangle with rotatable vertex colors."""

import sys
import numpy as np
import taichi as ti
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPainter


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

        # Field to store the rendered image (RGB)
        # Shape is (height, width) following image convention: pixels[row, col]
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(height, width))

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

    @ti.func
    def edge_distance(self, p, v0, v1, v2):
        """Calculate minimum distance to triangle edges for anti-aliasing.

        Args:
            p: Point to test
            v0, v1, v2: Triangle vertices

        Returns:
            Minimum distance to any edge (negative inside, positive outside)
        """
        # Distance to each edge
        def line_distance(p, a, b):
            pa = p - a
            ba = b - a
            h = ti.math.clamp(pa.dot(ba) / ba.dot(ba), 0.0, 1.0)
            return (pa - ba * h).norm()

        d0 = line_distance(p, v0, v1)
        d1 = line_distance(p, v1, v2)
        d2 = line_distance(p, v2, v0)

        return ti.min(ti.min(d0, d1), d2)

    @ti.kernel
    def render(self):
        """Render the triangle with interpolated vertex colors and anti-aliasing."""
        # Get vertices
        v0 = self.vertices[0]
        v1 = self.vertices[1]
        v2 = self.vertices[2]

        # Background color
        bg_color = ti.Vector([1.0, 1.0, 1.0])

        # Render triangle with anti-aliasing
        for i, j in self.pixels:
            # Convert pixel coordinates to normalized device coordinates (-1 to 1)
            # i is row (vertical), j is column (horizontal)
            x = (j / self.width) * 2.0 - 1.0
            y = 1.0 - (i / self.height) * 2.0  # Flip y so top is positive
            p = ti.Vector([x, y])

            # Calculate barycentric coordinates
            bary = self.barycentric(p, v0, v1, v2)

            # Minimum barycentric coordinate (for edge distance-based AA)
            min_bary = ti.min(ti.min(bary[0], bary[1]), bary[2])

            # Check if point is near or inside triangle
            if min_bary >= -0.02:  # Render a bit outside for AA
                # Interpolate color using barycentric coordinates
                color = (
                    bary[0] * self.colors[0] +
                    bary[1] * self.colors[1] +
                    bary[2] * self.colors[2]
                )

                # Anti-aliasing: smooth transition at edges
                # Scale factor for AA (controls smoothness)
                aa_scale = ti.max(self.width, self.height) * 0.5
                alpha = ti.math.smoothstep(0.0, 1.0 / aa_scale, min_bary)

                # Blend between triangle color and background
                self.pixels[i, j] = color * alpha + bg_color * (1.0 - alpha)
            else:
                # Outside triangle
                self.pixels[i, j] = bg_color

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
        # Convert Taichi field to numpy array (already in correct orientation)
        img = self.pixels.to_numpy()

        # Convert from float32 [0,1] to uint8 [0,255]
        img = (img * 255).astype(np.uint8)

        return img


class TaichiRenderWidget(QWidget):
    """Custom Qt widget for efficiently displaying Taichi-rendered content with scaling."""

    def __init__(self, render_width=800, render_height=600, parent=None):
        """Initialize the render widget.

        Args:
            render_width: Internal render buffer width in pixels
            render_height: Internal render buffer height in pixels
            parent: Parent widget
        """
        super().__init__(parent)
        self.render_width = render_width
        self.render_height = render_height

        # Set minimum size but allow resizing
        self.setMinimumSize(400, 300)

        # Create persistent QImage for efficient buffer updates
        # Allocate buffer once and reuse it
        self.image_buffer = np.zeros((render_height, render_width, 3), dtype=np.uint8)
        self.q_image = QImage(
            self.image_buffer.data,
            render_width,
            render_height,
            render_width * 3,  # bytes per line
            QImage.Format.Format_RGB888
        )

        # Cached scaled image for smooth rendering
        self.scaled_image = None

    def update_image(self, image_data):
        """Update the displayed image efficiently.

        Args:
            image_data: numpy array of shape (height, width, 3) with uint8 values
        """
        # Copy new data into persistent buffer (fast memcpy)
        np.copyto(self.image_buffer, image_data)

        # Invalidate scaled cache
        self.scaled_image = None

        # Request repaint
        self.update()

    def resizeEvent(self, event):
        """Handle resize events by invalidating the scaled image cache.

        Args:
            event: QResizeEvent
        """
        self.scaled_image = None
        super().resizeEvent(event)

    def paintEvent(self, event):
        """Handle paint events by drawing the scaled image buffer.

        Args:
            event: QPaintEvent
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Get widget size
        widget_width = self.width()
        widget_height = self.height()

        # Calculate scaling to fit while maintaining aspect ratio
        scale_x = widget_width / self.render_width
        scale_y = widget_height / self.render_height
        scale = min(scale_x, scale_y)

        # Calculate centered position
        scaled_width = int(self.render_width * scale)
        scaled_height = int(self.render_height * scale)
        x_offset = (widget_width - scaled_width) // 2
        y_offset = (widget_height - scaled_height) // 2

        # Draw scaled image with smooth interpolation
        target_rect = painter.viewport()
        target_rect.setRect(x_offset, y_offset, scaled_width, scaled_height)
        source_rect = self.q_image.rect()

        painter.drawImage(target_rect, self.q_image, source_rect)


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

        # Custom render widget for efficient display with resizing support
        self.render_widget = TaichiRenderWidget(render_width=800, render_height=600)
        layout.addWidget(self.render_widget)

        # Rotate button
        self.rotate_button = QPushButton("Rotate Colors")
        self.rotate_button.clicked.connect(self.on_rotate_clicked)
        layout.addWidget(self.rotate_button)

        # Set initial window size (resizable)
        self.resize(820, 680)

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

        # Update the widget efficiently (no QImage recreation)
        self.render_widget.update_image(img_array)


def run():
    """Run the PyQt6 triangle application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
