"""Simple colored triangle using Taichi GGUI integrated with Qt - like classic OpenGL hello world."""

import sys
import taichi as ti
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PyQt6.QtCore import QTimer


def run():
    """Run the Qt + Taichi GGUI triangle example."""
    ti.init(arch=ti.gpu)

    # Create window with GGUI
    window = ti.ui.Window("Taichi GGUI: Colored Triangle (Qt-controlled)", (800, 600))
    canvas = window.get_canvas()

    # Define triangle vertices (x, y, z) - 3 vertices
    # GGUI canvas uses [0, 1] coordinates, not [-1, 1] like OpenGL
    # (0, 0) = bottom-left, (1, 1) = top-right
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=3)
    vertices[0] = [0.5, 0.8, 0.0]   # Top (center-top)
    vertices[1] = [0.25, 0.35, 0.0] # Bottom left
    vertices[2] = [0.75, 0.35, 0.0] # Bottom right

    # Define vertex colors (r, g, b) - 3 colors
    colors = ti.Vector.field(3, dtype=ti.f32, shape=3)
    colors[0] = [1.0, 0.0, 0.0]  # Red
    colors[1] = [0.0, 1.0, 0.0]  # Green
    colors[2] = [0.0, 0.0, 1.0]  # Blue

    # Triangle indices (defining the triangle)
    indices = ti.field(ti.i32, shape=3)
    indices[0] = 0
    indices[1] = 1
    indices[2] = 2

    print("Controls:")
    print("  ESC or close window to exit")
    print("  Click 'Swap Colors' button to rotate vertex colors")

    # Qt setup
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    main_window.setWindowTitle("Taichi + Qt Control Panel")
    main_window.setGeometry(100, 100, 300, 150)

    # Create central widget and layout
    central_widget = QWidget()
    layout = QVBoxLayout()
    central_widget.setLayout(layout)
    main_window.setCentralWidget(central_widget)

    # Create button to swap colors
    swap_button = QPushButton("Swap Colors")
    layout.addWidget(swap_button)

    def swap_colors():
        """Rotate the vertex colors around."""
        temp = [colors[0][0], colors[0][1], colors[0][2]]
        colors[0] = colors[1]
        colors[1] = colors[2]
        colors[2] = temp
        print("Colors swapped!")

    swap_button.clicked.connect(swap_colors)

    # Render loop driven by Qt timer
    def render_frame():
        """Single frame render - called by Qt timer."""
        if not window.running:
            timer.stop()
            app.quit()
            return

        # Handle escape key
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                timer.stop()
                app.quit()
                return

        # Set background color (white)
        canvas.set_background_color((1.0, 1.0, 1.0))

        # Draw the triangle with vertex colors
        # GGUI handles all rasterization, interpolation, and anti-aliasing!
        canvas.triangles(
            vertices=vertices,
            color=(1.0, 1.0, 1.0),  # Not used when per_vertex_color is provided
            per_vertex_color=colors,
            indices=indices
        )

        # Display the frame
        window.show()

    # Create timer to drive rendering at ~60fps
    timer = QTimer()
    timer.timeout.connect(render_frame)
    timer.start(16)  # ~60fps (16ms per frame)

    # Show Qt window
    main_window.show()

    # Start Qt event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
