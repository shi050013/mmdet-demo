from __future__ import annotations

import os

import huggingface_hub
import numpy as np
import torch
import torch.nn as nn
import yaml
from mmdet.apis import inference_detector, init_detector


def _load_model_dict(path: str) -> dict[str, dict[str, str]]:
    with open(path) as f:
        dic = yaml.safe_load(f)
    _update_config_path(dic)
    _update_model_dict_if_hf_token_is_given(dic)
    return dic


def _update_config_path(model_dict: dict[str, dict[str, str]]) -> None:
    for dic in model_dict.values():
        dic['config'] = dic['config'].replace(
            'https://github.com/open-mmlab/mmdetection/tree/master',
            'mmdet_configs')


def _update_model_dict_if_hf_token_is_given(
        model_dict: dict[str, dict[str, str]]) -> None:
    token = os.getenv('HF_TOKEN')
    if token is None:
        return

    for dic in model_dict.values():
        ckpt_path = dic['model']
        name = ckpt_path.split('/')[-1]
        ckpt_path = huggingface_hub.hf_hub_download('hysts/mmdetection',
                                                    f'models/{name}',
                                                    use_auth_token=token)
        dic['model'] = ckpt_path


class Model:
    DETECTION_MODEL_DICT = _load_model_dict('model_dict/detection.yaml')
    INSTANCE_SEGMENTATION_MODEL_DICT = _load_model_dict(
        'model_dict/instance_segmentation.yaml')
    PANOPTIC_SEGMENTATION_MODEL_DICT = _load_model_dict(
        'model_dict/panoptic_segmentation.yaml')
    MODEL_DICT = DETECTION_MODEL_DICT | INSTANCE_SEGMENTATION_MODEL_DICT | PANOPTIC_SEGMENTATION_MODEL_DICT

    def __init__(self, model_name: str, device: str | torch.device):
        self.device = torch.device(device)
        self._load_all_models_once()
        self.model_name = model_name
        self.model = self._load_model(model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        return init_detector(dic['config'], dic['model'], device=self.device)

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def detect_and_visualize(
        self, image: np.ndarray, score_threshold: float
    ) -> tuple[list[np.ndarray] | tuple[list[np.ndarray],
                                        list[list[np.ndarray]]]
               | dict[str, np.ndarray], np.ndarray]:
        out = self.detect(image)
        vis = self.visualize_detection_results(image, out, score_threshold)
        return out, vis

    def detect(
        self, image: np.ndarray
    ) -> list[np.ndarray] | tuple[
            list[np.ndarray], list[list[np.ndarray]]] | dict[str, np.ndarray]:
        image = image[:, :, ::-1]  # RGB -> BGR
        out = inference_detector(self.model, image)
        return out

    def visualize_detection_results(
            self,
            image: np.ndarray,
            detection_results: list[np.ndarray]
        | tuple[list[np.ndarray], list[list[np.ndarray]]]
        | dict[str, np.ndarray],
            score_threshold: float = 0.3) -> np.ndarray:
        image = image[:, :, ::-1]  # RGB -> BGR
        vis = self.model.show_result(image,
                                     detection_results,
                                     score_thr=score_threshold,
                                     bbox_color=None,
                                     text_color=(200, 200, 200),
                                     mask_color=None)
        return vis[:, :, ::-1]  # BGR -> RGB


class AppModel(Model):
    def run(
        self, model_name: str, image: np.ndarray, score_threshold: float
    ) -> tuple[list[np.ndarray] | tuple[list[np.ndarray],
                                        list[list[np.ndarray]]]
               | dict[str, np.ndarray], np.ndarray]:
        self.set_model(model_name)
        return self.detect_and_visualize(image, score_threshold)
