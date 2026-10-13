"""Regenerate the PNG preview of the JointGV architecture diagram."""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1500, 920


def font(size: int, bold: bool = False):
    candidates = (
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def box(draw, bounds, fill, outline, title, lines=()):
    draw.rounded_rectangle(bounds, radius=16, fill=fill, outline=outline, width=2)
    x0, y0, x1, _ = bounds
    center = (x0 + x1) // 2
    draw.text((center, y0 + 18), title, font=font(17, True), fill="#f8fafc", anchor="ma")
    for index, line in enumerate(lines):
        draw.text((center, y0 + 49 + index * 23), line, font=font(13), fill="#cbd5e1", anchor="ma")


def arrow(draw, points, fill="#94a3b8", width=3):
    draw.line(points, fill=fill, width=width, joint="curve")
    x, y = points[-1]
    previous_x, previous_y = points[-2]
    if abs(x - previous_x) >= abs(y - previous_y):
        sign = 1 if x > previous_x else -1
        triangle = [(x, y), (x - 11 * sign, y - 6), (x - 11 * sign, y + 6)]
    else:
        sign = 1 if y > previous_y else -1
        triangle = [(x, y), (x - 6, y - 11 * sign), (x + 6, y - 11 * sign)]
    draw.polygon(triangle, fill=fill)


def main() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#0f172a")
    draw = ImageDraw.Draw(image)
    draw.text((55, 32), "Joint Graph–Video Latent Dynamics World Model", font=font(30, True), fill="#e2e8f0")
    draw.text((55, 73), "Explicit graph perception, recursive Graph Dynamics, and delta-G / delta-Z co-dynamics", font=font(15), fill="#94a3b8")
    box(draw, (55, 135, 225, 235), "#1e293b", "#64748b", "RGB observation", ("current + history",))
    box(draw, (55, 660, 225, 742), "#1e293b", "#64748b", "Robot action", ("a_t",))

    draw.rounded_rectangle((285, 115, 1445, 345), radius=22, fill="#172554", outline="#3b82f6", width=2)
    draw.text((315, 132), "Backbone-native visual world model", font=font(17, True), fill="#f8fafc")
    box(draw, (315, 180, 550, 285), "#1e3a8a", "#3b82f6", "Visual encoder", ("VAE / Dreamer CNN / DINOv2", "native latent"))
    box(draw, (625, 180, 875, 285), "#1e3a8a", "#3b82f6", "Latent transition", ("MDN-LSTM / RSSM / DINO-WM", "base delta-Z_t"))
    box(draw, (1190, 180, 1410, 285), "#1e3a8a", "#3b82f6", "Conditioned prior", ("predicted Z_(t+1)", "recursive rollout"))

    draw.rounded_rectangle((285, 390, 975, 620), radius=22, fill="#052e2b", outline="#14b8a6", width=2)
    draw.text((315, 407), "Pretrained Explicit Graph Observer (RGB-only inference)", font=font(17, True), fill="#f8fafc")
    box(draw, (315, 455, 495, 570), "#115e59", "#14b8a6", "Official RelTR", ("semantic queries +", "dense layer3 map"))
    box(draw, (520, 455, 710, 570), "#115e59", "#14b8a6", "Role grounding", ("dense soft-argmax", "bounded residual"))
    box(draw, (735, 455, 945, 570), "#115e59", "#14b8a6", "Explicit relations", ("VTransE head", "1-round IMP"))
    box(draw, (1015, 420, 1275, 570), "#4c1d95", "#a78bfa", "Graph Dynamics", ("GN(G_t, a_t)", "delta-G_t -> G_(t+1)", "bounded recursion"))
    box(draw, (1015, 620, 1410, 755), "#7c2d12", "#fb923c", "Graph–world co-dynamics", ("tokens: Sinkhorn delta-G <-> delta-Z", "global: delta-s queries graph nodes", "bounded/gated correction"))
    box(draw, (1015, 805, 1410, 895), "#1e293b", "#64748b", "Reward / Actor / Critic", ("Dreamer: unchanged RSSM feature", "DINO-WM: prediction only"))
    box(draw, (300, 675, 850, 830), "#312e81", "#818cf8", "MuJoCo teacher (training only)", ("one geom: 3D + 2D coordinates", "grounding first, then temporal", "never an inference input"))

    arrow(draw, [(225, 185), (315, 185)])
    arrow(draw, [(550, 232), (625, 232)])
    arrow(draw, [(875, 232), (1190, 232)])
    arrow(draw, [(140, 235), (140, 512), (315, 512)])
    arrow(draw, [(495, 512), (520, 512)])
    arrow(draw, [(710, 512), (735, 512)])
    arrow(draw, [(945, 512), (1015, 512)])
    # The same robot action conditions both the native visual transition and
    # Graph Dynamics. Keep this branch explicit so the diagram does not imply
    # that DreamerV3/MDN-LSTM are action-free.
    arrow(draw, [(225, 684), (260, 684), (260, 320), (750, 320), (750, 285)])
    draw.text((690, 300), "a_t", font=font(13, True), fill="#94a3b8")
    arrow(draw, [(225, 701), (930, 701), (930, 540), (1015, 540)])
    arrow(draw, [(1145, 570), (1145, 620)])
    arrow(draw, [(875, 260), (970, 260), (970, 687), (1015, 687)])
    arrow(draw, [(1410, 687), (1460, 687), (1460, 232), (1410, 232)])
    arrow(draw, [(1300, 285), (1300, 805)])
    image.save(
        Path(__file__).resolve().parents[3]
        / "docs" / "figures" / "jointgv_architecture.png"
    )


if __name__ == "__main__":
    main()
