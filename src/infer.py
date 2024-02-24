
from src.pipeline.segmentation import  infer_segmentation
from src.pipeline.centering import  infer_centered

import wandb

if __name__ == '__main__':


    run_name = "new_full_annotations_centered"
    # location = "/home/baris/Documents/brain-morphing"
    location = "/home/imreb/brain-morphing"
    # location = "/home/baris/Documents/work/brain-morphing"

    wandb.init(project="annotations",
               entity="barisimre",
               name=run_name)

    # train_segmentation(batch_size=2, run_name=run_name, location=location, slice_thickness="large", dims=3)
    # optimize_centers(location=location, run_name=run_name, batch_size=1, num_epochs=75)
    # infer_centered(location=location, run_name=run_name, read_location="new_annotations_centering", do_annotations=True, hdf5_target="new_annotations.hdf5")
    #
    infer_segmentation(location=location, relative_model_path="/training_with_hdf5_just_in_case/weights/2600.pt", run_name=run_name, make_hdf5=True, do_skull_strip=True, hdf5_location=f'{location}/data/new_annotations.hdf5')
