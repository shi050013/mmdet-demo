#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import tarfile

if os.getenv('SYSTEM') == 'spaces':
    import mim

    mim.uninstall('mmcv-full', confirm_yes=True)
    mim.install('mmcv-full==1.5.2', is_yes=True)

    subprocess.call('pip uninstall -y opencv-python'.split())
    subprocess.call('pip uninstall -y opencv-python-headless'.split())
    subprocess.call('pip install opencv-python-headless'.split())

import gradio as gr

from model import Model

DEFAULT_MODEL_TYPE = 'detection'
DEFAULT_MODEL_NAMES = {
    'detection': 'YOLOX-l',
    'instance_segmentation': 'QueryInst (R-50-FPN)',
    'panoptic_segmentation': 'MaskFormer (R-50)',
}
DEFAULT_MODEL_NAME = DEFAULT_MODEL_NAMES[DEFAULT_MODEL_TYPE]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    return parser.parse_args()


def extract_tar() -> None:
    if pathlib.Path('mmdet_configs/configs').exists():
        return
    with tarfile.open('mmdet_configs/configs.tar') as f:
        f.extractall('mmdet_configs')


def update_model_name(model_type: str) -> dict:
    model_dict = getattr(Model, f'{model_type.upper()}_MODEL_DICT')
    model_names = list(model_dict.keys())
    model_name = DEFAULT_MODEL_NAMES[model_type]
    return gr.Dropdown.update(choices=model_names, value=model_name)


def update_visualization_score_threshold(model_type: str) -> dict:
    return gr.Slider.update(visible=model_type != 'panoptic_segmentation')


def update_redraw_button(model_type: str) -> dict:
    return gr.Button.update(visible=model_type != 'panoptic_segmentation')


def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])


def main():
    args = parse_args()

    extract_tar()
    model = Model(DEFAULT_MODEL_NAME, args.device)

    css = '''
h1#title {
  text-align: center;
}
img#overview {
  max-width: 1000px;
  max-height: 600px;
}
'''

    with gr.Blocks(theme=args.theme, css=css) as demo:
        gr.Markdown('''<h1 id="title">MMDetection</h1>

This is an unofficial demo for [https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection).
<center><img id="overview" alt="overview" src="https://user-images.githubusercontent.com/12907710/137271636-56ba1cd2-b110-4812-8221-b4c120320aa9.png" /></center>
''')

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(label='Input Image', type='numpy')
                with gr.Group():
                    with gr.Row():
                        model_type = gr.Radio(list(DEFAULT_MODEL_NAMES.keys()),
                                              value=DEFAULT_MODEL_TYPE,
                                              label='Model Type')
                    with gr.Row():
                        model_name = gr.Dropdown(list(
                            model.DETECTION_MODEL_DICT.keys()),
                                                 value=DEFAULT_MODEL_NAME,
                                                 label='Model')
                with gr.Row():
                    run_button = gr.Button(value='Run')
                    prediction_results = gr.Variable()
            with gr.Column():
                with gr.Row():
                    visualization = gr.Image(label='Result', type='numpy')
                with gr.Row():
                    visualization_score_threshold = gr.Slider(
                        0,
                        1,
                        step=0.05,
                        value=0.3,
                        label='Visualization Score Threshold')
                with gr.Row():
                    redraw_button = gr.Button(value='Redraw')

        with gr.Row():
            paths = sorted(pathlib.Path('images').rglob('*.jpg'))
            example_images = gr.Dataset(components=[input_image],
                                        samples=[[path.as_posix()]
                                                 for path in paths])

        gr.Markdown(
            '<center><img src="https://visitor-badge.glitch.me/badge?page_id=hysts.mmdetection" alt="visitor badge"/></center>'
        )

        model_type.change(fn=update_model_name,
                          inputs=[model_type],
                          outputs=[model_name])
        model_type.change(fn=update_visualization_score_threshold,
                          inputs=[model_type],
                          outputs=[visualization_score_threshold])
        model_type.change(fn=update_redraw_button,
                          inputs=[model_type],
                          outputs=[redraw_button])

        model_name.change(fn=model.set_model,
                          inputs=[model_name],
                          outputs=None)
        run_button.click(fn=model.detect_and_visualize,
                         inputs=[
                             input_image,
                             visualization_score_threshold,
                         ],
                         outputs=[
                             prediction_results,
                             visualization,
                         ])
        redraw_button.click(fn=model.visualize_detection_results,
                            inputs=[
                                input_image,
                                prediction_results,
                                visualization_score_threshold,
                            ],
                            outputs=[visualization])
        example_images.click(fn=set_example_image,
                             inputs=[example_images],
                             outputs=[input_image])

    demo.launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
