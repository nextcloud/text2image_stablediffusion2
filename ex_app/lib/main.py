import asyncio
import io
import os
import threading
from contextlib import asynccontextmanager
from threading import Event
from time import sleep, perf_counter
import logging

import torch
import PIL.Image
from fastapi import FastAPI
from nc_py_api import NextcloudApp
from nc_py_api.ex_app import AppAPIAuthMiddleware, LogLvl, run_app, set_handlers, get_computation_device
from diffusers import AutoPipelineForText2Image
from nc_py_api.ex_app.providers.task_processing import TaskProcessingProvider, ShapeDescriptor, ShapeType
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def log(nc, level, content):
    logger.log((level+1)*10, content)
    if level < LogLvl.WARNING:
        return
    try:
        asyncio.run(nc.log(level, content))
    except:
        pass

TASKPROCESSING_PROVIDER_ID = 'text2image_stablediffusion2:sdxl_turbo'

def load_model():
    if get_computation_device().lower() == 'cuda':
        pipe = AutoPipelineForText2Image.from_pretrained("Nextcloud-AI/sdxl-turbo", torch_dtype=torch.float16,
                                                         variant="fp16")
        pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
    else:
        # Cpu does not support fp16
        pipe = AutoPipelineForText2Image.from_pretrained("Nextcloud-AI/sdxl-turbo", torch_dtype=torch.float32,
                                                         variant="fp16")
        pipe.to("cpu")
    return pipe

app_enabled = Event()
@asynccontextmanager
async def lifespan(app: FastAPI):
    set_handlers(app, enabled_handler)
    t = BackgroundProcessTask()
    t.start()
    nc = NextcloudApp()
    if nc.enabled_state:
        app_enabled.set()
    yield


APP = FastAPI(lifespan=lifespan)
APP.add_middleware(AppAPIAuthMiddleware)  # set global AppAPI authentication middleware


class BackgroundProcessTask(threading.Thread):
    def run(self, *args, **kwargs):  # pylint: disable=unused-argument
        nc = NextcloudApp()
        while not app_enabled.is_set():
            sleep(5)

        pipe = load_model()

        while True:
            if not app_enabled.is_set() or pipe is None:
                sleep(30)
                continue
            try:
                next = nc.providers.task_processing.next_task([TASKPROCESSING_PROVIDER_ID], ['core:text2image'])
                if not 'task' in next or next is None:
                    sleep(5)
                    continue
                task = next.get('task')
            except Exception as e:
                print(str(e))
                log(nc, LogLvl.ERROR, str(e))
                sleep(30)
                continue
            try:
                log(nc, LogLvl.INFO, f"Next task: {task['id']}")

                log(nc, LogLvl.INFO, "generating image")
                time_start = perf_counter()
                prompt = task.get("input").get('input')
                size = task.get('input').get('size') or '512x512'
                width, height = size.split('x')
                width = int(width)
                height = int(height)
                images: List[PIL.Image.Image] = pipe(width=width, height=height, prompt=prompt, num_inference_steps=int(os.getenv('NUM_INFERENCE_STEPS', 4)), guidance_scale=0.0, num_images_per_prompt=task.get("input").get('numberOfImages')).images
                log(nc, LogLvl.INFO, f"image generated: {perf_counter() - time_start}s")

                img_ids = []
                for image in images:
                    png_stream = io.BytesIO()
                    image.save(png_stream, format="PNG")
                    img_ids.append(nc.providers.task_processing.upload_result_file(task.get('id'), png_stream))

                NextcloudApp().providers.task_processing.report_result(
                    task["id"],
                    {'images': img_ids},
                )
            except Exception as e:  # noqa
                print(str(e))
                try:
                    log(nc, LogLvl.ERROR, str(e))
                    nc.providers.task_processing.report_result(task["id"], None, str(e))
                except:
                    pass
                sleep(30)



async def enabled_handler(enabled: bool, nc: NextcloudApp) -> str:
    # This will be called each time application is `enabled` or `disabled`
    # NOTE: `user` is unavailable on this step, so all NC API calls that require it will fail as unauthorized.
    print(f"enabled={enabled}")
    if enabled:
        nc.log(LogLvl.WARNING, f"Enabled: {nc.app_cfg.app_name}")
        await nc.providers.task_processing.register(TaskProcessingProvider(
            id=TASKPROCESSING_PROVIDER_ID,
            name='Nextcloud Local Image Generation: Stable Diffusion',
            task_type='core:text2image',
            expected_runtime=120,
            optional_input_shape=[
                ShapeDescriptor(name='size', description='Optional. The size of the generated images. Must be in 512x512 format. Default is 512x512', shape_type=ShapeType.TEXT),
            ],
            input_shape_defaults={'size': '512x512', "numberOfImages": 1},
        ))
        app_enabled.set()
    else:
        await nc.providers.task_processing.unregister('text2image_stablediffusion2:sdxl_turbo', True)
        nc.log(LogLvl.WARNING, f"Disabled {nc.app_cfg.app_name}")
        app_enabled.clear()
    # In case of an error, a non-empty short string should be returned, which will be shown to the NC administrator.
    return ""


if __name__ == "__main__":
    # Wrapper around `uvicorn.run`.
    # You are free to call it directly, with just using the `APP_HOST` and `APP_PORT` variables from the environment.
    run_app("main:APP", log_level="trace")
