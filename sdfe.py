import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        if not opt.skip_save:
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_checked_image_torch)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

class guiMain(tk.Frame):
    def __init__(self, master=None):
        super().__init__()
        self.master.title(u'StableDiffusionFrontEnd')
        self.master.geometry('600x300')
        self.paramsEntry = {}
        self.create_widgets()
        self.pack()
    def create_widgets(self):
        self.outputpdf_textbox = guiTextEntry(master=self, label=u'出力ファイル名', inittext=self.params.output_pdf, boxwidth=40)
        self.imgfolder_dirdiag = guiDirDiag(master=self, label=u'出力フォルダ', initdir=self._folder)
        self.imgfolder_dirdiag.pack(anchor=tk.W)
        self.outfolder_dirdiag.pack(anchor=tk.W)
        self.outputpdf_textbox.pack(anchor=tk.W)

        frame_crop = tk.Frame(master=self)
        self.crop_textbox = guiTextEntry(master=frame_crop, label=u'左右切り取りマージン', boxwidth=10, inittext=self.params.margin)
        self.crop_textbox.grid(row=0, column=0, sticky=tk.W)
        frame_crop.pack(anchor=tk.W)

        frame_recursive = tk.Frame(master=self)
        self.suffix_textbox = guiTextEntry(master=frame_recursive, label=u'添字', boxwidth=10, inittext=self.params.suffix)
        self.initcount_textbox = guiTextEntry(master=frame_recursive, label=u'始めの数字', boxwidth=5, inittext=self.params.initcount)
        self.recursive_check = guiRadioButton(master=frame_recursive, label=u'入力フォルダ以下の各フォルダで変換', initcond=self.params.recursive,
            slave_widget=None)
        self.suffix_textbox.grid(row=0, column=1)
        self.initcount_textbox.grid(row=0, column=2)
        self.recursive_check.grid(row=0, column=0, sticky=tk.W)
        frame_recursive.pack(anchor=tk.W)

        frame_split = tk.Frame(master=self)
        frame_split_textbox = tk.Frame(master=frame_split)
        self.splitpages_textbox = guiTextEntry(master=frame_split_textbox, label=u'分割するページ', boxwidth=5, inittext=self.params.splitpage or '')
        self.splitmargin_textbox = guiTextEntry(master=frame_split_textbox, label=u'分割する際の左右マージン', boxwidth=5, inittext=self.params.splitmargin)
        self.splitpages_textbox.grid(row=0, column=0)
        self.splitmargin_textbox.grid(row=0, column=1)
        self.split_check = guiRadioButton(master=frame_split, label=u'ページを分割', initcond=self.params.split,
            slave_widget=None)
        frame_split_textbox.grid(row=0, column=1)
        self.split_check.grid(row=0, column=0, sticky=tk.W)
        frame_split.pack(anchor=tk.W)

        frame_bottom = tk.Frame(master=self)
        button1 = tk.Button(master=frame_bottom, text="実行", command=self.convert)
        button2 = tk.Button(master=frame_bottom, text=("閉じる"), command=quit)
        button1.grid(row=0, column=0)
        button2.grid(row=0, column=1)
        frame_bottom.pack()
    def convert(self):
        self.params.img_folder = self.imgfolder_dirdiag.entry.get()
        self.params.output_dir = self.outfolder_dirdiag.entry.get()
        self.params.output_pdf = self.outputpdf_textbox.entry.get()
        self.params.recursive = self.recursive_check.entry.get()
        self.params.suffix = self.suffix_textbox.entry.get()
        self.params.split = self.split_check.entry.get()
        self.params.splitmargin = int(self.splitmargin_textbox.entry.get())
        self.params.splitpage = self.splitpages_textbox.entry.get()
        self.params.margin = int(self.crop_textbox.entry.get())
        self.params.initcount = int(self.initcount_textbox.entry.get())
        convert(params)

class guiDirDiag(tk.Frame):
    def __init__(self, master=None, label=None, initdir=None, width=100, boxwidth=30):
        super().__init__(master=master)
        self.label = label
        self.initdir = initdir
        self.boxwidth = boxwidth
        self.width = width
        self.create_widgets()
    def create_widgets(self):
        self.entry = tk.StringVar()
        self.entry.set(self.initdir)
        box = tk.Entry(self, textvariable=self.entry, width=self.boxwidth)
        label = tk.Label(self, text=self.label)
        button = tk.Button(self, text=u'フォルダ選択', command=self.dirdialog_clicked)
        label.pack(side=tk.LEFT, anchor=tk.W)
        box.pack(side=tk.LEFT)
        button.pack(side=tk.LEFT)
    def dirdialog_clicked(self):
        #initdir = os.path.abspath(os.path.dirname(self.entry.get()))
        initdir = os.path.abspath(self.entry.get())
        dir = tkf.askdirectory(initialdir = initdir)
        if dir is not None:
            self.entry.set(dir)

class guiTextEntry(tk.Frame):
    def __init__(self, master=None, label=None, inittext=None, boxwidth=30):
        super().__init__(master=master)
        self.label = label
        self.inittext = inittext
        self.boxwidth = boxwidth
        self.create_widgets()
    def create_widgets(self):
        self.entry = tk.StringVar()
        self.entry.set(self.inittext)
        box = tk.Entry(self, textvariable=self.entry, width=self.boxwidth)
        label = tk.Label(self, text=self.label)
        label.pack(side=tk.LEFT, anchor=tk.W)
        box.pack(side=tk.LEFT)

class guiRadioButton(tk.Frame):
    def __init__(self, master=None, label=None, initcond=False, slave_widget=None):
        super().__init__(master=master)
        self.label = label
        self.initcond = initcond
        self.slave_widget = slave_widget
        self.create_widgets()
        if slave_widget is not None:
            self.interactive()
    def create_widgets(self):
        self.entry = tk.BooleanVar()
        self.entry.set(self.initcond)
        if self.slave_widget is not None:
            checkbox = tk.Checkbutton(master=self, text=self.label, variable=self.entry, command=self.interactive)
        else:
            checkbox = tk.Checkbutton(master=self, text=self.label, variable=self.entry)
        checkbox.pack(side=tk.LEFT, anchor=tk.W)
    def interactive(self):
        if self.entry.get():
            self.slave_widget.grid()
        else:
            self.slave_widget.grid_remove()

if __name__ == "__main__":
    main()
