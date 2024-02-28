
from src.pipeline.segmentation import  infer_segmentation
from src.pipeline.centering import  infer_centered

import wandb

if __name__ == '__main__':


    run_name = "new_script_input"
    # location = "/home/baris/Documents/brain-morphing"
    location = "/home/imreb/brain-morphing"
    # location = "/home/baris/Documents/work/brain-morphing"

    wandb.init(project="infer",
               entity="barisimre",
               name=run_name)


    # infer_centered(location=location, run_name=run_name, read_location="new_data_center_final", do_annotations=False, hdf5_target="all_new_data.hdf5")


    infer_segmentation(location=location, relative_model_path="/training_with_hdf5_just_in_case/weights/4800.pt", run_name=run_name, make_hdf5=True, do_skull_strip=True, hdf5_location=f'new_deg_input.hdf5')
