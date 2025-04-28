# Temporal-disaggregation-with-Ordinary-Differential-Equation-ODE-

Time series disaggregation is not merely about missing data imputation or interpolation. It also involves disaggregation (or anti-aggregation), which refers to decomposing low-frequency time series into high-frequency components.
Two scenarios:
(1) When both low-frequency and high-frequency units share the same meaning (interpolation scenario).
(2) When low-frequency data is aggregated from high-frequency data (anti-aggregation scenario, modeled by methods like Chow-Lin and Denton).
Despite the rapid development of deep learning, time series disaggregation has rarely been a primary research focus, often treated as a form of imputation. I am particularly interested in Neural ODEs, and subsequent ODE-based models have primarily focused on irregular sampling, but no major research has treated anti-aggregation as the main direction.

I was trying to use Neural SDEs [1,2,3,4] to do temporal disaggregation with auxialiary data. But I failed. In anti-aggregation scenarios, there are no explicit labels, and all unsurpervised learning natrually is clustering. Maybe current DL was not enough to do that.


[1] Xuechen Li, Ting-Kam Leonard Wong, Ricky T. Q. Chen, David Duvenaud. "Scalable Gradients for Stochastic Differential Equations". International Conference on Artificial Intelligence and Statistics. 2020. (https://arxiv.org/pdf/2001.01328)

[2] Patrick Kidger, James Foster, Xuechen Li, Harald Oberhauser, Terry Lyons. "Neural SDEs as Infinite-Dimensional GANs". International Conference on Machine Learning 2021. (https://proceedings.mlr.press/v139/kidger21b.html)
