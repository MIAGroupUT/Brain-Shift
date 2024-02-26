
from src.pipeline.segmentation import  infer_segmentation
from src.pipeline.centering import  infer_centered

import wandb

if __name__ == '__main__':


    run_name = "input_for_morph"
    # location = "/home/baris/Documents/brain-morphing"
    location = "/home/imreb/brain-morphing"
    # location = "/home/baris/Documents/work/brain-morphing"

    wandb.init(project="infer",
               entity="barisimre",
               name=run_name)


    # infer_centered(location=location, run_name=run_name, read_location="all_data_centering", do_annotations=False, hdf5_target="all_data.hdf5")


    infer_segmentation(location=location, relative_model_path="/training_with_hdf5_just_in_case/weights/4800.pt", run_name=run_name, make_hdf5=True, do_skull_strip=True, hdf5_location=f'input_for_segmentation.hdf5')
