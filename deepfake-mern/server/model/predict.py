import base64
import contextlib
import io
import json
import os
import sys

import cv2
import numpy as np


def _suppress_stdout():
    return contextlib.redirect_stdout(io.StringIO())


def _to_base64_png(rgb_image: np.ndarray) -> str:
    if rgb_image is None:
        return ''

    image = np.asarray(rgb_image)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode('.png', bgr)
    if not ok:
        return ''

    return base64.b64encode(buffer).decode('ascii')


def _ensure_cvcp_on_path() -> str:
    # Repo layout assumption:
    #   F:\CV_CP\deepfake-mern\server\model\predict.py
    #   F:\CV_CP\src\predict.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cvcp_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    if cvcp_root not in sys.path:
        sys.path.insert(0, cvcp_root)
    return cvcp_root


def main() -> None:
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError('Image path not provided or file does not exist')

    cvcp_root = _ensure_cvcp_on_path()

    from src.predict import DeepfakePredictor

    model_path = os.getenv('MODEL_PATH') or os.path.join(cvcp_root, 'models', 'best_model.pth')
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'Model file not found at: {model_path}. '
            'Set MODEL_PATH to an absolute path to your trained .pth file.'
        )
    device = os.getenv('DEVICE')  # optional: 'cpu' or 'cuda'
    image_size = int(os.getenv('IMAGE_SIZE') or '224')
    use_ela = (os.getenv('USE_ELA') or 'true').strip().lower() in ('1', 'true', 'yes', 'y')

    with _suppress_stdout():
        predictor = DeepfakePredictor(
            model_path=model_path,
            device=device if device else None,
            use_ela=use_ela,
            image_size=image_size,
        )

        prediction = predictor.predict(image_path)
        features = predictor.get_feature_visualizations(image_path)
        gradcam = predictor.get_gradcam(image_path)

    prob = prediction.get('probability')
    if prob is None:
        # Fallback: infer from label + confidence
        confidence_pct = float(prediction.get('confidence') or 0)
        if str(prediction.get('label', '')).upper() == 'FAKE':
            prob = confidence_pct / 100.0
        else:
            prob = 1.0 - (confidence_pct / 100.0)

    prob = float(prob)
    fake_pct = prob * 100.0
    real_pct = (1.0 - prob) * 100.0

    response = {
        'label': prediction.get('label', 'UNKNOWN'),
        # DeepfakePredictor returns confidence in percent already (0-100)
        'confidence': prediction.get('confidence', 0),
        'inference_time': prediction.get('inference_time_ms', 0),
        'real_pct': round(real_pct, 2),
        'fake_pct': round(fake_pct, 2),
        'ela': _to_base64_png(features.get('ela')),
        'fft': _to_base64_png(features.get('fft')),
        'edges': _to_base64_png(features.get('edges')),
        'gradcam': _to_base64_png(gradcam),
    }

    # IMPORTANT: stdout must be JSON only (Express parses this).
    print(json.dumps(response))


if __name__ == '__main__':
    main()
