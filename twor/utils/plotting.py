from twor.geom.objects import get_glyph_bounds


def set_axis_glyph_bounds(ax, glyphs):
    x0, y0, x1, y1 = get_glyph_bounds(glyphs)
    xy0 = min(x0, y0)
    xy1 = max(x1, y1)
    ax.set_xlim([xy0, xy1])
    ax.set_ylim([xy0, xy1])
    return xy0, xy1
