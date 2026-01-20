# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython-accelerated frame generation for simulated EMCCD in photon-counting mode.

Realistic simulation of:
- Poisson photon arrivals
- Stochastic EM gain (exponential distribution per electron)
- Clock-induced charge (CIC)
- Read noise
"""

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, cos, sin, pow

np.import_array()

ctypedef np.float32_t FLOAT32
ctypedef np.float64_t FLOAT64
ctypedef np.uint16_t UINT16
ctypedef np.int32_t INT32


def compute_star_image(list stars, dict galaxy,
                       int img_width, int img_height,
                       int hbin, int hstart, int vstart):
    """
    Compute the star and galaxy pattern image.
    Returns a float32 array with expected photon rate per pixel (photons/sec).
    """
    cdef np.ndarray[FLOAT32, ndim=2] image = np.zeros((img_height, img_width), dtype=np.float32)
    cdef int i, x_min, x_max, y_min, y_max, xx, yy, radius
    cdef double star_x, star_y, sigma, flux, fwhm
    cdef double dist_sq, psf_val, psf_sum
    cdef double gal_x, gal_y, r_eff, ellipticity, position_angle, sersic_n, gal_flux
    cdef double pa_rad, cos_pa, sin_pa, dx, dy, x_rot, y_rot, q, r, b_n, profile_val, profile_sum

    # Render stars
    for i in range(len(stars)):
        star = stars[i]
        star_x = (star['x'] - hstart + 1) / hbin
        star_y = (star['y'] - vstart + 1) / hbin
        fwhm = star['fwhm']
        flux = star['flux']

        if star_x < 0 or star_x >= img_width or star_y < 0 or star_y >= img_height:
            continue

        sigma = fwhm / (2.355 * hbin)
        radius = max(1, int(sigma * 4))

        x_min = max(0, int(star_x) - radius)
        x_max = min(img_width, int(star_x) + radius + 1)
        y_min = max(0, int(star_y) - radius)
        y_max = min(img_height, int(star_y) + radius + 1)

        # Compute PSF sum first
        psf_sum = 0.0
        for yy in range(y_min, y_max):
            for xx in range(x_min, x_max):
                dist_sq = (xx - star_x) * (xx - star_x) + (yy - star_y) * (yy - star_y)
                psf_sum += exp(-dist_sq / (2 * sigma * sigma))

        # Add normalized PSF
        if psf_sum > 1e-10:
            for yy in range(y_min, y_max):
                for xx in range(x_min, x_max):
                    dist_sq = (xx - star_x) * (xx - star_x) + (yy - star_y) * (yy - star_y)
                    psf_val = exp(-dist_sq / (2 * sigma * sigma))
                    image[yy, xx] += <FLOAT32>(flux * psf_val / psf_sum)

    # Render galaxy (Sersic profile)
    gal_x = (galaxy['x'] - hstart + 1) / hbin
    gal_y = (galaxy['y'] - vstart + 1) / hbin
    r_eff = galaxy['r_eff'] / hbin
    ellipticity = galaxy['ellipticity']
    position_angle = galaxy['position_angle']
    sersic_n = galaxy['sersic_n']
    gal_flux = galaxy['flux']

    if 0 <= gal_x < img_width and 0 <= gal_y < img_height:
        radius = max(1, int(r_eff * 5))
        x_min = max(0, int(gal_x) - radius)
        x_max = min(img_width, int(gal_x) + radius + 1)
        y_min = max(0, int(gal_y) - radius)
        y_max = min(img_height, int(gal_y) + radius + 1)

        pa_rad = position_angle * 3.14159265358979 / 180.0
        cos_pa = cos(pa_rad)
        sin_pa = sin(pa_rad)
        q = 1.0 - ellipticity
        b_n = 2 * sersic_n - 1.0 / 3 + 0.009876 / sersic_n

        # Compute profile sum first
        profile_sum = 0.0
        for yy in range(y_min, y_max):
            for xx in range(x_min, x_max):
                dx = xx - gal_x
                dy = yy - gal_y
                x_rot = dx * cos_pa + dy * sin_pa
                y_rot = -dx * sin_pa + dy * cos_pa
                r = sqrt(x_rot * x_rot + (y_rot / q) * (y_rot / q))
                if r < 0.1:
                    r = 0.1
                profile_sum += exp(-b_n * (pow(r / r_eff, 1.0 / sersic_n) - 1))

        # Add normalized profile
        if profile_sum > 1e-10:
            for yy in range(y_min, y_max):
                for xx in range(x_min, x_max):
                    dx = xx - gal_x
                    dy = yy - gal_y
                    x_rot = dx * cos_pa + dy * sin_pa
                    y_rot = -dx * sin_pa + dy * cos_pa
                    r = sqrt(x_rot * x_rot + (y_rot / q) * (y_rot / q))
                    if r < 0.1:
                        r = 0.1
                    profile_val = exp(-b_n * (pow(r / r_eff, 1.0 / sersic_n) - 1))
                    image[yy, xx] += <FLOAT32>(gal_flux * profile_val / profile_sum)

    return image


def generate_frame_fast(np.ndarray[FLOAT32, ndim=2] flux_map,
                        int img_width, int img_height,
                        float exposure_time, float em_gain,
                        float bias_level, float cic_rate,
                        object rng):
    """
    Generate a realistic EMCCD frame in photon-counting mode.

    Uses vectorized operations for speed while maintaining physical accuracy:
    - Poisson photon arrivals
    - Stochastic EM gain via gamma distribution
    - CIC as single-electron events
    """
    cdef int num_cic, i
    cdef int max_photons

    # Expected photon counts per pixel
    expected = flux_map * exposure_time

    # Generate Poisson photon arrivals
    photon_counts = rng.poisson(expected)

    # Start with bias
    frame = np.full((img_height, img_width), bias_level, dtype=np.float32)

    # Find pixels with photons and apply stochastic EM gain
    # For efficiency, process by photon count level
    max_photons = int(photon_counts.max())

    for n in range(1, min(max_photons + 1, 20)):  # Cap at 20 for speed
        mask = (photon_counts == n)
        count = mask.sum()
        if count > 0:
            # Gamma(n, em_gain) gives sum of n exponential(em_gain) RVs
            amplified = rng.gamma(n, em_gain, size=count).astype(np.float32)
            frame[mask] += amplified

    # Handle high photon counts (rare) with approximation
    high_mask = (photon_counts >= 20)
    if high_mask.any():
        n_vals = photon_counts[high_mask]
        # For large n, gamma approaches normal: mean=n*gain, var=n*gain^2
        amplified = rng.gamma(n_vals, em_gain).astype(np.float32)
        frame[high_mask] += amplified

    # Add CIC events (single electrons with exponential EM gain)
    num_cic = rng.poisson(cic_rate * img_width * img_height)
    if num_cic > 0:
        cic_y = rng.integers(0, img_height, num_cic)
        cic_x = rng.integers(0, img_width, num_cic)
        cic_vals = rng.exponential(em_gain, num_cic).astype(np.float32)
        for i in range(num_cic):
            frame[cic_y[i], cic_x[i]] += cic_vals[i]

    # Add small read noise
    frame += rng.normal(0, 3.0, size=(img_height, img_width)).astype(np.float32)

    return np.clip(frame, 0, 65535).astype(np.uint16)
