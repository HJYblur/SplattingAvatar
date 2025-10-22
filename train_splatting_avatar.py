import os
from pathlib import Path
from random import randint
from argparse import ArgumentParser
from gaussian_renderer import network_gui
from datetime import datetime
from tqdm import tqdm
from omegaconf import OmegaConf
from model.splatting_avatar_model import SplattingAvatarModel
from model.splatting_avatar_optim import SplattingAvatarOptimizer
from model.loss_base import run_testing
from dataset.dataset_helper import make_frameset_data, make_dataloader
from model import libcore

# Add logging of training time
import time
import logging
import torch

if __name__ == "__main__":
    parser = ArgumentParser(description="SplattingAvatar Training")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--dat_dir", type=str, required=True)
    parser.add_argument(
        "--configs",
        type=lambda s: [i for i in s.split(";")],
        required=True,
        help="path to config file",
    )
    parser.add_argument("--model_path", type=str, default=None)
    args, extras = parser.parse_known_args()

    # output dir
    if args.model_path is None:
        model_path = f"output-splatting/{datetime.now().strftime('@%Y%m%d-%H%M%S')}"
    else:
        model_path = args.model_path

    if not os.path.isabs(model_path):
        model_path = os.path.join(args.dat_dir, model_path)
    os.makedirs(model_path, exist_ok=True)

    # load model and training config
    config = libcore.load_from_config(args.configs, cli_args=extras)
    OmegaConf.save(config, os.path.join(model_path, "config.yaml"))
    libcore.set_seed(config.get("seed", 9061))

    # ---------------- logging setup ----------------
    log_path = os.path.join(model_path, "train.log")
    logger = logging.getLogger("splatting")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="a")
        sh = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)

    timings_csv = os.path.join(model_path, "timings.csv")
    if not os.path.exists(timings_csv):
        with open(timings_csv, "w") as f:
            f.write("iter,iter_ms,render_ms,loss,psnr,num_gauss\n")

    testing_csv = os.path.join(model_path, "testing_timings.csv")
    if not os.path.exists(testing_csv):
        with open(testing_csv, "w") as f:
            f.write("iter,total_seconds\n")
    # ------------------------------------------------

    ##################################################
    config.dataset.dat_dir = args.dat_dir
    config.cache_dir = os.path.join(args.dat_dir, f"cache_{Path(args.configs[0]).stem}")
    frameset_train = make_frameset_data(config.dataset, split="train")
    frameset_test = make_frameset_data(config.dataset, split="test")
    dataloader = make_dataloader(frameset_train, shuffle=True)

    # first frame as canonical
    first_batch = frameset_train.__getitem__(0)
    cano_mesh = first_batch["mesh_info"]

    ##################################################
    pipe = config.pipe
    gs_model = SplattingAvatarModel(config.model, verbose=True)
    gs_model.create_from_canonical(cano_mesh)

    gs_optim = SplattingAvatarOptimizer(gs_model, config.optim)

    ##################################################
    if args.ip != "none":
        network_gui.init(args.ip, args.port)
        verify = args.dat_dir
    else:
        verify = None

    data_iterator = iter(dataloader)

    total_iteration = config.optim.total_iteration
    save_every_iter = config.optim.get("save_every_iter", 10000)
    testing_iterations = config.optim.get("testing_iterations", [total_iteration])

    # Record the training start time
    start_time = datetime.now()
    print(f"Training started at: {start_time}")
    logger.info(f"Training started at: {start_time}")
    train_wall_start = time.perf_counter()
    sum_iter_ms = 0.0
    sum_render_ms = 0.0
    pbar = tqdm(range(1, total_iteration + 1))
    for iteration in pbar:
        gs_optim.update_learning_rate(iteration)
        iter_t0 = time.perf_counter()

        try:
            batches = next(data_iterator)
        except:
            data_iterator = iter(dataloader)
            batches = next(data_iterator)

        batch = batches[0]
        frm_idx = batch["frm_idx"]
        scene_cameras = batch["scene_cameras"]

        # update to current posed mesh
        gs_model.update_to_posed_mesh(batch["mesh_info"])

        # there should be only one camera
        viewpoint_cam = scene_cameras[0].cuda()
        gt_image = viewpoint_cam.original_image

        # send one image to gui (optional)
        if args.ip != "none":
            network_gui.render_to_network(gs_model, pipe, verify, gt_image=gt_image)

        # render (time the pure render)
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        render_t0 = time.perf_counter()
        render_pkg = gs_model.render_to_camera(viewpoint_cam, pipe)
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        render_ms = (time.perf_counter() - render_t0) * 1000.0
        image = render_pkg["render"]
        gt_image = render_pkg["gt_image"]
        gt_alpha_mask = render_pkg["gt_alpha_mask"]

        # ### debug ###
        # from model import libcore
        # libcore.write_tensor_image(os.path.join('e:/dummy/gt_image.jpg'), gt_image, rgb2bgr=True)
        # libcore.write_tensor_image(os.path.join('e:/dummy/render.jpg'), image, rgb2bgr=True)

        # loss
        loss = gs_optim.collect_loss(gt_image, image, gt_alpha_mask=gt_alpha_mask)
        loss["loss"].backward()

        # densify and prune
        gs_optim.adaptive_density_control(render_pkg, iteration)

        gs_optim.step()
        gs_optim.zero_grad(set_to_none=True)

        # end-of-iter timing (include queued GPU work)
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        iter_ms = (time.perf_counter() - iter_t0) * 1000.0

        sum_iter_ms += iter_ms
        sum_render_ms += render_ms

        pbar.set_postfix(
            {
                "#gauss": gs_model.num_gauss,
                "loss": loss["loss"].item(),
                "psnr": loss["psnr_full"],
                "iter_ms": f"{iter_ms:.1f}",
                "render_ms": f"{render_ms:.1f}",
            }
        )

        # periodic logging and CSV row
        if iteration % 50 == 0 or iteration == 1:
            logger.info(
                f"iter {iteration}/{total_iteration} "
                f"loss={loss['loss'].item():.5f} psnr={loss['psnr_full']:.2f} "
                f"iter_ms={iter_ms:.2f} render_ms={render_ms:.2f} #gauss={gs_model.num_gauss}"
            )
        with open(timings_csv, "a") as f:
            f.write(
                f"{iteration},{iter_ms:.4f},{render_ms:.4f},{loss['loss'].item():.6f},{loss['psnr_full']:.4f},{gs_model.num_gauss}\n"
            )

        # walking on triangles
        gs_optim.update_trangle_walk(iteration)

        # report testing
        if iteration in testing_iterations:
            t0 = time.perf_counter()
            run_testing(
                pipe, frameset_test, gs_model, model_path, iteration, verify=verify
            )
            test_secs = time.perf_counter() - t0
            logger.info(f"Testing at iter {iteration} took {test_secs:.2f}s")
            with open(testing_csv, "a") as f:
                f.write(f"{iteration},{test_secs:.4f}\n")

        # save
        if iteration % save_every_iter == 0:
            pc_dir = gs_optim.save_checkpoint(model_path, iteration)
            libcore.write_tensor_image(
                os.path.join(pc_dir, "gt_image.jpg"), gt_image, rgb2bgr=True
            )
            libcore.write_tensor_image(
                os.path.join(pc_dir, "render.jpg"), image, rgb2bgr=True
            )

    ##################################################
    # training summary and inference micro-benchmark
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    total_train_secs = time.perf_counter() - train_wall_start
    avg_iter_ms = (sum_iter_ms / max(1, total_iteration)) if total_iteration else 0.0
    avg_render_ms = (
        (sum_render_ms / max(1, total_iteration)) if total_iteration else 0.0
    )
    logger.info(
        f"Training finished in {total_train_secs/60.0:.2f} min "
        f"(avg iter {avg_iter_ms:.2f} ms, avg render {avg_render_ms:.2f} ms)"
    )

    # Simple inference benchmark on one test view
    try:
        sample = frameset_test.__getitem__(0)
        gs_model.update_to_posed_mesh(sample["mesh_info"])
        cam = sample["scene_cameras"][0].cuda()
        # warmup
        for _ in range(5):
            _ = gs_model.render_to_camera(cam, pipe)
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        reps = 30
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = gs_model.render_to_camera(cam, pipe)
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        avg_ms = (time.perf_counter() - t0) * 1000.0 / reps
        fps = 1000.0 / avg_ms if avg_ms > 0 else float("inf")
        logger.info(
            f"Inference micro-benchmark (1 view): avg {avg_ms:.2f} ms, {fps:.1f} FPS"
        )
    except Exception as e:
        logger.warning(f"Benchmark skipped: {e}")

    ##################################################
    # training finished. hold on
    while network_gui.conn is not None:
        network_gui.render_to_network(gs_model, pipe, args.dat_dir)

    print("[done]")
