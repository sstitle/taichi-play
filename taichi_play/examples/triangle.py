"""Simple colored triangle using Taichi GGUI - like classic OpenGL hello world."""

import taichi as ti


def run():
    """Run the simple GGUI triangle example."""
    ti.init(arch=ti.gpu)

    # Create window with GGUI
    window = ti.ui.Window("Taichi GGUI: Colored Triangle (OpenGL-style)", (800, 600))
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

    # Render loop
    while window.running:
        # Handle escape key
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break

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


if __name__ == "__main__":
    run()
