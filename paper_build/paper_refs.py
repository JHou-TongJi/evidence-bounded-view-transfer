#!/usr/bin/env python
"""Bibliography for the IEEE T-ITS manuscript.

Ordered by first citation in the text (IEEE convention). `REFS` maps a stable key
to an IEEE-formatted entry string. `cite(*keys)` returns the bracketed label and
records usage so `unused()` can flag entries that never appear in the text --
the automated equivalent of the bib-cleaning step.
"""
from collections import OrderedDict

REFS = OrderedDict([

    # --- radiance fields and Gaussian splatting ---------------------------
    ("nerf",
     "B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and "
     "R. Ng, “NeRF: Representing scenes as neural radiance fields for view "
     "synthesis,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2020, pp. 405–421."),
    ("mipnerf360",
     "J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, "
     "“Mip-NeRF 360: Unbounded anti-aliased neural radiance fields,” in Proc. "
     "IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 5470–5479."),
    ("instantngp",
     "T. Müller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics "
     "primitives with a multiresolution hash encoding,” ACM Trans. Graph., vol. 41, "
     "no. 4, art. no. 102, pp. 1–15, Jul. 2022."),
    ("gs3d",
     "B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis, “3D Gaussian "
     "splatting for real-time radiance field rendering,” ACM Trans. Graph., vol. 42, "
     "no. 4, art. no. 139, pp. 1–14, Jul. 2023."),
    ("gs2d",
     "B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2D Gaussian splatting for "
     "geometrically accurate radiance fields,” in Proc. ACM SIGGRAPH Conf. Papers, "
     "2024, art. no. 32, pp. 1–11."),
    ("sugar",
     "A. Guédon and V. Lepetit, “SuGaR: Surface-aligned Gaussian splatting for "
     "efficient 3D mesh reconstruction and high-quality mesh rendering,” in Proc. "
     "IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 5354–5363."),
    ("gof",
     "Z. Yu, T. Sattler, and A. Geiger, “Gaussian opacity fields: Efficient adaptive "
     "surface reconstruction in unbounded scenes,” ACM Trans. Graph., vol. 43, no. 6, "
     "art. no. 271, pp. 1–13, Dec. 2024."),
    ("scaffoldgs",
     "T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai, “Scaffold-GS: "
     "Structured 3D Gaussians for view-adaptive rendering,” in Proc. IEEE/CVF Conf. "
     "Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 20654–20664."),
    ("mipsplat",
     "Z. Yu, A. Chen, B. Huang, T. Sattler, and A. Geiger, “Mip-Splatting: "
     "Alias-free 3D Gaussian splatting,” in Proc. IEEE/CVF Conf. Comput. Vis. "
     "Pattern Recognit. (CVPR), 2024, pp. 19447–19456."),

    # --- driving-scene reconstruction -------------------------------------
    ("blocknerf",
     "M. Tancik, V. Casser, X. Yan, S. Pradhan, B. Mildenhall, P. P. Srinivasan, "
     "J. T. Barron, and H. Kretzschmar, “Block-NeRF: Scalable large scene neural "
     "view synthesis,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. "
     "(CVPR), 2022, pp. 8248–8258."),
    ("snerf",
     "Z. Xie, J. Zhang, W. Li, F. Zhang, and L. Zhang, “S-NeRF: Neural radiance "
     "fields for street views,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2023."),
    ("emernerf",
     "J. Yang, B. Ivanovic, O. Litany, X. Weng, S. W. Kim, B. Li, T. Che, D. Xu, "
     "S. Fidler, M. Pavone, and Y. Wang, “EmerNeRF: Emergent spatial-temporal scene "
     "decomposition via self-supervision,” in Proc. Int. Conf. Learn. Represent. "
     "(ICLR), 2024."),
    ("mars",
     "Z. Wu, T. Liu, L. Luo, Z. Zhong, J. Chen, H. Xiao, C. Hou, H. Lou, Y. Chen, "
     "R. Yang, Y. Huang, X. Ye, Z. Yan, Y. Shi, Y. Liao, and H. Zhao, “MARS: An "
     "instance-aware, modular and realistic simulator for autonomous driving,” in "
     "Artificial Intelligence (CICAI 2023), Lecture Notes in Computer Science, "
     "vol. 14473, Singapore: Springer, 2024, pp. 3–15."),
    ("streetsurf",
     "J. Guo, N. Deng, X. Li, Y. Bai, B. Shi, C. Wang, C. Ding, D. Wang, and Y. Li, "
     "“StreetSurf: Extending multi-view implicit surface reconstruction to street "
     "views,” arXiv:2306.04988, 2023."),
    ("streetgs",
     "Y. Yan, H. Lin, C. Zhou, W. Wang, H. Sun, K. Zhan, X. Lang, X. Zhou, and "
     "S. Peng, “Street Gaussians: Modeling dynamic urban scenes with Gaussian "
     "splatting,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024."),
    ("drivinggs",
     "X. Zhou, Z. Lin, X. Shan, Y. Wang, D. Sun, and M.-H. Yang, “DrivingGaussian: "
     "Composite Gaussian splatting for surrounding dynamic autonomous driving "
     "scenes,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, "
     "pp. 21634–21643."),
    ("pvg",
     "Y. Chen, C. Gu, J. Jiang, X. Zhu, and L. Zhang, “Periodic vibration Gaussian: "
     "Dynamic urban scene reconstruction and real-time rendering,” "
     "arXiv:2311.18561, 2023."),
    ("gs4d",
     "G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, "
     "“4D Gaussian splatting for real-time dynamic scene rendering,” in Proc. "
     "IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 20310–20320."),
    ("omnire",
     "Z. Chen, J. Yang, J. Huang, R. de Lutio, J. M. Esturo, B. Ivanovic, O. Litany, "
     "Z. Gojcic, S. Fidler, M. Pavone, L. Song, and Y. Wang, “OmniRe: Omni urban "
     "scene reconstruction,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2025."),
    ("hugs",
     "H. Zhou, J. Shao, L. Xu, D. Bai, W. Qiu, B. Liu, Y. Wang, A. Geiger, and "
     "Y. Liao, “HUGS: Holistic urban 3D scene understanding via Gaussian "
     "splatting,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), "
     "2024, pp. 21336–21345."),

    # --- sensor simulation and data generation ----------------------------
    ("carla",
     "A. Dosovitskiy, G. Ros, F. Codevilla, A. López, and V. Koltun, “CARLA: An "
     "open urban driving simulator,” in Proc. Conf. Robot Learn. (CoRL), 2017, "
     "pp. 1–16."),
    ("lidarsim",
     "S. Manivasagam, S. Wang, K. Wong, W. Zeng, M. Sazanovich, S. Tan, B. Yang, "
     "W.-C. Ma, and R. Urtasun, “LiDARsim: Realistic LiDAR simulation by leveraging "
     "the real world,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. "
     "(CVPR), 2020, pp. 11167–11176."),
    ("geosim",
     "Y. Chen, F. Rong, S. Duggal, S. Wang, X. Yan, S. Manivasagam, S. Xue, "
     "E. Yumer, and R. Urtasun, “GeoSim: Realistic video simulation via geometry-"
     "aware composition for self-driving,” in Proc. IEEE/CVF Conf. Comput. Vis. "
     "Pattern Recognit. (CVPR), 2021, pp. 7230–7240."),
    ("surfelgan",
     "Z. Yang, Y. Chai, D. Anguelov, Y. Zhou, P. Sun, D. Erhan, S. Rafferty, and "
     "H. Kretzschmar, “SurfelGAN: Synthesizing realistic sensor data for autonomous "
     "driving,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, "
     "pp. 11118–11127."),
    ("unisim",
     "Z. Yang, Y. Chen, J. Wang, S. Manivasagam, W.-C. Ma, A. J. Yang, and "
     "R. Urtasun, “UniSim: A neural closed-loop sensor simulator,” in Proc. "
     "IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, "
     "pp. 1389–1399."),
    ("neurad",
     "A. Tonderski, C. Lindström, G. Hess, W. Ljungbergh, L. Svensson, and "
     "C. Petersson, “NeuRAD: Neural rendering for autonomous driving,” in Proc. "
     "IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, "
     "pp. 14895–14904."),
    ("splatad",
     "G. Hess, C. Lindström, M. Fatemi, C. Petersson, and L. Svensson, “SplatAD: "
     "Real-time LiDAR and camera rendering with 3D Gaussian splatting for autonomous "
     "driving,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, "
     "pp. 11982–11992."),
    ("magicdrive",
     "R. Gao, K. Chen, E. Xie, L. Hong, Z. Li, D.-Y. Yeung, and Q. Xu, "
     "“MagicDrive: Street view generation with diverse 3D geometry control,” in "
     "Proc. Int. Conf. Learn. Represent. (ICLR), 2024."),
    ("simrealgap",
     "G. Hess, W. Ljungbergh, A. Tonderski, C. Petersson, and L. Svensson, "
     "“Splatting the sim-to-real gap: Evaluating neural rendering for autonomous "
     "driving perception,” arXiv:2501.06719, 2025."),

    # --- datasets ---------------------------------------------------------
    ("kitti",
     "A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for autonomous driving? The "
     "KITTI vision benchmark suite,” in Proc. IEEE Conf. Comput. Vis. Pattern "
     "Recognit. (CVPR), 2012, pp. 3354–3361."),
    ("nuscenes",
     "H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, "
     "Y. Pan, G. Baldan, and O. Beijbom, “nuScenes: A multimodal dataset for "
     "autonomous driving,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. "
     "(CVPR), 2020, pp. 11621–11631."),
    ("waymo",
     "P. Sun et al., “Scalability in perception for autonomous driving: Waymo Open "
     "Dataset,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, "
     "pp. 2446–2454."),
    ("argoverse2",
     "B. Wilson et al., “Argoverse 2: Next generation datasets for self-driving "
     "perception and forecasting,” in Proc. NeurIPS Datasets and Benchmarks Track, "
     "2021."),
    ("pandaset",
     "P. Xiao, Z. Shao, S. Hao, Z. Zhang, X. Chai, J. Jiao, Z. Li, J. Wu, K. Sun, "
     "K. Jiang, Y. Wang, and D. Yang, “PandaSet: Advanced sensor suite dataset for "
     "autonomous driving,” in Proc. IEEE Int. Intell. Transp. Syst. Conf. (ITSC), "
     "2021, pp. 3095–3101."),
    ("ncore",
     "NVIDIA, “PhysicalAI Autonomous Vehicles – NCore,” Hugging Face, 2026. "
     "[Online]. Available: "
     "https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore "
     "(accessed Sep. 1, 2026)."),

    # --- camera models, geometry, components ------------------------------
    ("colmap",
     "J. L. Schönberger and J.-M. Frahm, “Structure-from-motion revisited,” in "
     "Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 4104–4113."),
    ("kannala",
     "J. Kannala and S. S. Brandt, “A generic camera model and calibration method "
     "for conventional, wide-angle, and fish-eye lenses,” IEEE Trans. Pattern Anal. "
     "Mach. Intell., vol. 28, no. 8, pp. 1335–1340, Aug. 2006."),
    ("scaramuzza",
     "D. Scaramuzza, A. Martinelli, and R. Siegwart, “A toolbox for easily "
     "calibrating omnidirectional cameras,” in Proc. IEEE/RSJ Int. Conf. Intell. "
     "Robots Syst. (IROS), 2006, pp. 5695–5701."),
    ("rollingshutter",
     "J. Hedborg, P.-E. Forssén, M. Felsberg, and E. Ringaby, “Rolling shutter "
     "bundle adjustment,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), "
     "2012, pp. 1434–1441."),
    ("rsnerf",
     "M. Li, P. Wang, L. Zhao, B. Liao, and P. Liu, “USB-NeRF: Unrolling shutter "
     "bundle adjusted neural radiance fields,” in Proc. Int. Conf. Learn. Represent. "
     "(ICLR), 2024."),
    ("segformer",
     "E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “SegFormer: "
     "Simple and efficient design for semantic segmentation with transformers,” in "
     "Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2021, pp. 12077–12090."),
    ("depthanything2",
     "L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth "
     "Anything V2,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2024."),
    ("sam2",
     "N. Ravi et al., “SAM 2: Segment anything in images and videos,” in Proc. "
     "Int. Conf. Learn. Represent. (ICLR), 2025."),
    ("roma",
     "J. Edstedt, Q. Sun, G. Bokman, M. Wadenbäck, and M. Felsberg, “RoMa: Robust "
     "dense feature matching,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern "
     "Recognit. (CVPR), 2024, pp. 19790–19800."),
    ("charbonnier",
     "P. Charbonnier, L. Blanc-Féraud, G. Aubert, and M. Barlaud, "
     "“Deterministic edge-preserving regularization in computed imaging,” IEEE "
     "Trans. Image Process., vol. 6, no. 2, pp. 298–311, Feb. 1997."),

    # --- evaluation, uncertainty, abstention ------------------------------
    ("lpips",
     "R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The "
     "unreasonable effectiveness of deep features as a perceptual metric,” in Proc. "
     "IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 586–595."),
    ("ssim",
     "Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality "
     "assessment: From error visibility to structural similarity,” IEEE Trans. "
     "Image Process., vol. 13, no. 4, pp. 600–612, Apr. 2004."),
    ("niqe",
     "A. Mittal, R. Soundararajan, and A. C. Bovik, “Making a “completely blind” "
     "image quality analyzer,” IEEE Signal Process. Lett., vol. 20, no. 3, "
     "pp. 209–212, Mar. 2013."),
    ("bayesrays",
     "L. Goli, C. Reading, S. Sellán, A. Jacobson, and A. Tagliasacchi, "
     "“Bayes' Rays: Uncertainty quantification for neural radiance fields,” in "
     "Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, "
     "pp. 20061–20070."),
    ("cfnerf",
     "J. Shen, A. Agudo, F. Moreno-Noguer, and A. Ruiz, “Conditional-flow NeRF: "
     "Accurate 3D modelling with reliable uncertainty quantification,” in Proc. Eur. "
     "Conf. Comput. Vis. (ECCV), 2022, pp. 540–557."),
    ("chow",
     "C. K. Chow, “On optimum recognition error and reject tradeoff,” IEEE Trans. "
     "Inf. Theory, vol. 16, no. 1, pp. 41–46, Jan. 1970."),
    ("selectivenet",
     "Y. Geifman and R. El-Yaniv, “SelectiveNet: A deep neural network with an "
     "integrated reject option,” in Proc. Int. Conf. Mach. Learn. (ICML), 2019, "
     "pp. 2151–2159."),
    ("elyaniv",
     "R. El-Yaniv and Y. Wiener, “On the foundations of noise-free selective "
     "classification,” J. Mach. Learn. Res., vol. 11, pp. 1605–1641, May 2010."),
    ("conformal",
     "A. N. Angelopoulos and S. Bates, “Conformal prediction: A gentle "
     "introduction,” Found. Trends Mach. Learn., vol. 16, no. 4, pp. 494–591, "
     "2023."),
    ("mcdropout",
     "Y. Gal and Z. Ghahramani, “Dropout as a Bayesian approximation: Representing "
     "model uncertainty in deep learning,” in Proc. Int. Conf. Mach. Learn. (ICML), "
     "2016, pp. 1050–1059."),
    ("kendallgal",
     "A. Kendall and Y. Gal, “What uncertainties do we need in Bayesian deep "
     "learning for computer vision?,” in Proc. Adv. Neural Inf. Process. Syst. "
     "(NeurIPS), 2017, pp. 5574–5584."),
    ("iso21448",
     "Road Vehicles – Safety of the Intended Functionality, 1st ed., ISO Standard "
     "21448:2022, International Organization for Standardization, Geneva, "
     "Switzerland, Jun. 2022."),
    ("iso26262",
     "Road Vehicles – Functional Safety – Part 1: Vocabulary, 2nd ed., ISO "
     "Standard 26262-1:2018, International Organization for Standardization, "
     "Geneva, Switzerland, Dec. 2018."),
])

_USED = []


def cite(*keys):
    """Return the IEEE bracket label for one or more keys, e.g. [4], [4], [7]-[9]."""
    idx = []
    for k in keys:
        if k not in REFS:
            raise KeyError("unknown reference key: " + k)
        if k not in _USED:
            _USED.append(k)
        # IEEE numbers citations in the order they first appear in the text, not
        # in the order the bibliography happens to be declared. Numbering off the
        # REFS declaration order made the body open with "[51], [52]" and put ten
        # descending steps in the first-appearance sequence. `_USED` is appended
        # to exactly once per key, on first citation, so its index is already the
        # correct final number at the moment cite() is called.
        idx.append(_USED.index(k) + 1)
    idx.sort()
    # collapse runs of three or more
    out, i = [], 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        if j - i >= 2:
            out.append("%d]–[%d" % (idx[i], idx[j]))
        else:
            out.extend(str(n) for n in idx[i:j + 1])
        i = j + 1
    return "[" + "], [".join(out) + "]"


def unused():
    return [k for k in REFS if k not in _USED]


def entries():
    """Bibliography in first-citation order, matching the numbers cite() emits.

    Any key never cited would be invisible here; `unused()` is a build gate, so
    that case fails the build before it can silently drop an entry.
    """
    missing = [k for k in REFS if k not in _USED]
    if missing:
        raise ValueError("uncited references would be dropped: %s" % missing)
    return [REFS[k] for k in _USED]
