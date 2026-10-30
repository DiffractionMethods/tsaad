# detector_stats

Calibration-free estimation of detector response statistics—absorbed quantum gain ($G$), cascade factor ($\gamma$), and pixel noise factor ($\psi$)—directly from diffraction rotation-scan datasets.

## Overview

Following the detector response model of Waterman & Evans (2010), *J. Appl. Cryst.* **43**, 1356–1371, the signal variance on an integrating or charge-sharing detector is governed by:

$$\sigma_p^2 = \bar{p} \, G \gamma + N \psi$$

Local single-frame dispersion estimators (such as `dials.estimate_gain`) underestimate the true response product $G\gamma$ whenever a point-spread function (PSF) or charge-sharing kernel $K(\mathbf{u})$ introduces spatial correlations between neighboring pixels, damping single-pixel variance by the zero-lag PSF autocorrelation factor $S_2 = \alpha(\mathbf{0}) = \sum_{\mathbf{v}} K(\mathbf{v})^2 < 1$.

`detector_stats` treats a rotation scan as an ensemble of coupled pixel time series to separate smooth spatiotemporal background trends and transient outliers (Bragg reflections, cosmic-ray zingers) from stochastic detector fluctuations, recovering $(G, \gamma, \psi)$ and the spatial autocorrelation kernel $\alpha(\mathbf{u})$ via lag-summed spatial cross-covariance across intensity regimes.

## Running Tests

```bash
pytest
```
