"""Large report heading + technical caption for ML figures."""

from __future__ import annotations

from matplotlib.figure import Figure


def apply_report_titles(
    fig: Figure | None,
    heading: str | None,
    caption: str | None = None,
    *,
    steal_existing: bool = True,
) -> Figure | None:
    """Place an 18px heading and 11px technical line above the axes.

    If ``caption`` is omitted and ``steal_existing`` is True, reuse the figure
    suptitle, or a single axes title (cleared so it is not duplicated). Panel
    titles on multi-axes figures (for example validation / test) are kept.
    """
    if fig is None or not heading:
        return fig
    if getattr(fig, "_gummybear_report_titles", False):
        return fig

    stolen: str | None = None
    if steal_existing:
        if fig._suptitle is not None:
            text = str(fig._suptitle.get_text() or "").strip()
            if text:
                stolen = text
        if stolen is None:
            titled = [ax for ax in fig.axes if str(ax.get_title() or "").strip()]
            if len(titled) == 1:
                stolen = str(titled[0].get_title()).strip() or None

    cap = caption if caption is not None else stolen
    if fig._suptitle is not None:
        fig.suptitle("")
    titled = [ax for ax in fig.axes if str(ax.get_title() or "").strip()]
    if len(titled) == 1:
        ax_title = str(titled[0].get_title()).strip()
        if caption is not None or (stolen and ax_title == stolen):
            titled[0].set_title("")

    try:
        fig.set_layout_engine(None)
    except Exception:
        pass
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    fig.text(
        0.5,
        0.98,
        heading,
        ha="center",
        va="top",
        fontsize=18,
        fontweight="semibold",
    )
    if cap:
        fig.text(
            0.5,
            0.91,
            cap,
            ha="center",
            va="top",
            fontsize=11,
        )
    fig._gummybear_report_titles = True
    return fig
