"""Homogeneous optical material parameters for transport and diffusion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpticalMaterialConfig:
    """Homogeneous bulk optical properties for refraction, attenuation, and diffusion.

    ``n_refractive`` sets Snell refraction at entry and exit faces.
    ``mu_absorption`` and ``mu_scatter`` are linear attenuation coefficients in
    inverse mesh-length units; their sum drives Beer–Lambert loss along chords.
    ``g`` is the anisotropy factor used to derive the diffusion coefficient
    ``D = 1 / (3 * (mu_a + (1 - g) * mu_s))``; use ``g=0`` for isotropic scatter.

    Parameters
    ----------
    n_refractive:
        Relative refractive index of the phantom medium (dimensionless).
    mu_absorption:
        Absorption coefficient ``mu_a`` (1 / mesh units).
    mu_scatter:
        Scattering coefficient ``mu_s`` (1 / mesh units).
    g:
        Anisotropy factor in ``[-1, 1]``; ``0`` is isotropic.
    """

    n_refractive: float = 1.33
    mu_absorption: float = 0.1
    mu_scatter: float = 0.0
    g: float = 0.0

    @property
    def mu_a(self) -> float:
        """Absorption coefficient ``mu_a`` alias for ``mu_absorption``."""
        return self.mu_absorption

    @property
    def mu_s(self) -> float:
        """Scattering coefficient ``mu_s`` alias for ``mu_scatter``."""
        return self.mu_scatter

    @property
    def mu_total(self) -> float:
        """Total attenuation coefficient ``mu_a + mu_s`` (absorption plus scatter)."""
        return self.mu_absorption + self.mu_scatter

    @property
    def diffusion_coefficient(self) -> float:
        """Apparent diffusion coefficient from absorption coefficient ``mu_a``, scattering coefficient ``mu_s``, and ``g``.

        Raises
        ------
        ValueError
            When ``mu_total <= 0`` (diffusion undefined).
        """
        if self.mu_total <= 0:
            raise ValueError("Diffusion coefficient undefined for mu_total <= 0")
        return 1.0 / (3.0 * (self.mu_absorption + (1.0 - self.g) * self.mu_scatter))
