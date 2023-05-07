import os
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils.download_util import load_file_from_url

class esrgan:
    def __init__(self, model_name: str = "realesr-general-x4v3", gpu_id: int = None) -> None:
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        model_path = os.path.join('model_weights/esrgan', model_name + '.pth')
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'model_weights/esrgan'), progress=True, file_name=None)

        
        self.gan_model = RealESRGANer(
            model_path=model_path,
            model=model,
            scale=2,
            dni_weight=None,
            gpu_id=gpu_id
                )
        
    def enhance(self, image):
        output, _ = self.gan_model.enhance(image, outscale=2)
        return output
