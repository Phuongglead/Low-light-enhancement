from typing import Dict
import numpy as np

from module_A.analyze import analyze_image
from module_A.traditional import (
    apply_clahe,
    gray_world,
    retinex
)


class AdaptiveRestorationAgent:
    """
    Unified decision-making agent for adaptive image restoration.
    """

    def __init__(
        self,
        low_light_thresh: float = 25.0,
        low_contrast_thresh: float = 18.0,
        color_cast_thresh: float = 10.0,
        high_noise_thresh: float = None,
    ):
        """
        Args:
            low_light_thresh:
                Mean L-channel threshold below which the image is considered severely low-light.
            low_contrast_thresh:
                L-channel std threshold for low contrast.
            color_cast_thresh:
                Absolute a* or b* mean threshold for color cast detection.
            high_noise_thresh:
                Noise threshold derived from eval data (e.g., LOL eval).
                MUST be set before using the agent.
        """
        self.low_light_thresh = low_light_thresh
        self.low_contrast_thresh = low_contrast_thresh
        self.color_cast_thresh = color_cast_thresh
        self.high_noise_thresh = high_noise_thresh

    # PUBLIC API
    def run(self, img: np.ndarray) -> Dict:
        """
        Run the full adaptive restoration pipeline.

        Returns:
            Dict with:
                - output (np.ndarray)
                - analysis (Dict)
                - decision (Dict)
        """

        # 1. Perception (Module A)
        analysis = analyze_image(img)

        # 2. Pre-DL Gate
        dl_decision = self._should_use_dl(analysis)

        # 3. Deep Learning branch
        if dl_decision["use_dl"]:
            # Placeholder for Zero-DCE
            output = img.copy()

            decision = {
                "stage": "deep_learning",
                "method": "Zero-DCE",
                "reason": dl_decision["reason"],
                "evidence": dl_decision["evidence"],
            }

            return {
                "output": output,
                "analysis": analysis,
                "decision": decision,
            }

        # 4. Traditional branch
        output, method_decision = self._apply_traditional(img, analysis)

        decision = {
            "stage": "traditional",
            **method_decision,
        }

        return {
            "output": output,
            "analysis": analysis,
            "decision": decision,
        }

    # INTERNAL LOGIC – PRE-DL GATE
    def _should_use_dl(self, analysis: Dict) -> Dict:
        """
        Decide whether Deep Learning should be used.
        """
        brightness = analysis["brightness_mean"]
        noise = analysis["noise_score"]
        a_mean = abs(analysis["a_mean"])
        b_mean = abs(analysis["b_mean"])

        # Rule 1: Not severely low-light → avoid DL
        if brightness >= self.low_light_thresh:
            return {
                "use_dl": False,
                "reason": "Brightness is within recoverable range for traditional methods",
                "evidence": {
                    "brightness_mean": brightness,
                    "threshold": self.low_light_thresh,
                },
            }

        # Rule 2: Strong color cast → prefer white balance
        if a_mean > self.color_cast_thresh or b_mean > self.color_cast_thresh:
            return {
                "use_dl": False,
                "reason": "Strong color cast detected; white-balance is preferred",
                "evidence": {
                    "a_mean": a_mean,
                    "b_mean": b_mean,
                    "threshold": self.color_cast_thresh,
                },
            }

        # Rule 3: High noise → DL may amplify artifacts
        if noise >= self.high_noise_thresh:
            return {
                "use_dl": False,
                "reason": "High noise level; deep learning may amplify noise",
                "evidence": {
                    "noise_score": noise,
                    "noise_threshold": self.high_noise_thresh,
                },
            }

        # Otherwise: DL is expected to help
        return {
            "use_dl": True,
            "reason": "Severely low illumination with acceptable noise level",
            "evidence": {
                "brightness_mean": brightness,
                "noise_score": noise,
            },
        }

    # INTERNAL LOGIC – TRADITIONAL SELECTOR
    def _apply_traditional(self, img: np.ndarray, analysis: Dict):
        """
        Select and apply the most suitable traditional method.
        """

        brightness = analysis["brightness_mean"]
        contrast = analysis["contrast"]
        a_mean = abs(analysis["a_mean"])
        b_mean = abs(analysis["b_mean"])

        # Priority 1: Color cast
        if a_mean > self.color_cast_thresh or b_mean > self.color_cast_thresh:
            output, _ = gray_world(img)
            return output, {
                "method": "GRAY_WORLD",
                "reason": "Color cast detected in LAB color space",
                "evidence": {
                    "a_mean": a_mean,
                    "b_mean": b_mean,
                },
            }

        # Priority 2: Low contrast
        if contrast < self.low_contrast_thresh:
            output, _ = apply_clahe(img)
            return output, {
                "method": "CLAHE",
                "reason": "Low contrast under moderate illumination",
                "evidence": {
                    "contrast": contrast,
                    "threshold": self.low_contrast_thresh,
                },
            }

        # Priority 3: Mild illumination issue
        output, _ = retinex(img)
        return output, {
            "method": "RETINEX",
            "reason": "Mild illumination degradation",
            "evidence": {
                "brightness_mean": brightness,
            },
        }