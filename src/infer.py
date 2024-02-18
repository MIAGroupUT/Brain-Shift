
from src.pipeline.segmentation import  infer_segmentation
from src.pipeline.centering import  infer_centered

import wandb

if __name__ == '__main__':


    run_name = "centers_all"
    # location = "/home/baris/Documents/brain-morphing"
    location = "/home/imreb/brain-morphing"
    # location = "/home/baris/Documents/work/brain-morphing"

    wandb.init(project="centering",
               entity="barisimre",
               name=run_name)

    # train_segmentation(batch_size=2, run_name=run_name, location=location, slice_thickness="large", dims=3)
    # optimize_centers(location=location, run_name=run_name, batch_size=1, num_epochs=75)
    infer_centered(location=location, run_name=run_name, read_location="large_centers", slice_thickness="large", do_annotations=False)
    #
    # infer_segmentation(location=location, relative_model_path="/segmentation_3d_rotatefirst/weights/800.pt", run_name=run_name, slice_thickness="large", use_nifti=True, nifti_location='/data/centered', make_hdf5=True, do_skull_strip=True)
