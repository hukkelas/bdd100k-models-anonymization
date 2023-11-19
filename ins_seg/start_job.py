import click
import os
from pathlib import Path


def prepare_dataset_bdd100k(dataset_dir):
#    dataset_dir = dataset_dir.joinpath("bdd100k")
    original_dataset_dir = Path("/cluster/work/haakohu/datasets/bdd100k")
    to_symlink = [
        "jsons", "labels",
        "images/10k/test",
        "images/10k/val"
    ]
    for p in to_symlink:
        print(f"Symlinking: {dataset_dir.joinpath(p)} -> {original_dataset_dir.joinpath(p)}")
        if not dataset_dir.joinpath(p).exists():
            dataset_dir.joinpath(p).symlink_to(original_dataset_dir.joinpath(p))



Path("slurm_scripts").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)
slurm_script_path = "start_train.slurm"
NUM_SEED = 3

@click.command()
@click.argument("dataset_path")
@click.option("--eval-only", "-e", default=False, is_flag=True)
def main(dataset_path, eval_only):
    dataset_path = Path(dataset_path)
    if dataset_path.name == "bdd100k":
        name ="original"
    else:
        name = dataset_path.stem
        dataset_path = dataset_path.joinpath('bdd100k')

    
    config_file = "configs/ins_seg/mask_rcnn_r50_fpn_3x_ins_seg_bdd100k.py"
    task = {
        "configs/ins_seg/mask_rcnn_r50_fpn_3x_ins_seg_bdd100k.py": "mrcnn_r50_fpn"
    }[config_file]
    for seed in range(NUM_SEED): 
        output_dir = Path("/cluster/work/haakohu/outputs/mmdet_anonymization/", task, name, f"seed{seed}")
        test_output_dir = output_dir.joinpath("test_results")
        final_model_path = output_dir.joinpath("latest.pth")
        cmds = []
        eval_cmds = [
            f"MMDET_DATASETS={dataset_path}/ python test.py {config_file} --format-only --format-dir {test_output_dir} --cfg-options load_from={final_model_path}",
            f"python -m bdd100k.eval.run -t ins_seg  -g /cluster/work/haakohu/datasets/bdd100k/labels/ins_seg/bitmasks/val/ -r {test_output_dir.joinpath('bitmasks')} --score-file {test_output_dir.joinpath('score.json')}  --out-file {output_dir.joinpath('final_metrics.json')}"
        ]
        if eval_only:
            cmds = eval_cmds
        else:
            cmd = f"MMDET_DATASETS={dataset_path}/  python train.py {config_file} --work-dir {output_dir} --seed {seed}"
            cmds = [cmd] + eval_cmds
        new_slurm_script = f"slurm_scripts/{name}_{task}_{seed}.slurm"
        new_out = f"outputs/{name}_{task}_{seed}.out"

        with open(slurm_script_path, "r") as fp:
            lines = fp.readlines()
        with open(new_slurm_script, "w") as fp:
            for line in lines:
                line = line.strip()
                if "#SBATCH --output=out.slurm" in line:
                    line = f"#SBATCH --output={new_out}"
                if "#SBATCH --job-name=detectron2_keypoints" in line:
                    line = f"#SBATCH --job-name=mmdet_{name}_{task}_{seed}"
                fp.write(line + "\n")
            for cmd in cmds:
                fp.write(cmd + "\n")
        print("saving to:", output_dir)
        print("Output path:", new_out)
        print("Slurm script:", new_slurm_script)
        prepare_dataset_bdd100k(dataset_path)
        os.system(f"sbatch {new_slurm_script}")
        

if __name__ == "__main__":
    main()
