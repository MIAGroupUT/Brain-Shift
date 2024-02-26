from src.trainings.segmentation import train_segmentation
from src.trainings.centering import optimize_centers
from src.pipeline.segmentation import  infer_segmentation
from src.trainings.morph import train_morph
from src.trainings.instance_morph import train_morph_instant
import wandb

if __name__ == '__main__':


    # run_name = "seg_yesflip_norot_5000"
    run_name = 'all_data_center'
    # location = "/home/baris/Documents/brain-morphing"
    location = "/home/imreb/brain-morphing"
    # location = "/home/baris/Documents/work/brain-morphing"

    wandb.init(project="centering",
               entity="barisimre",
               name=run_name,
               notes="Got some new annotations")

#    train_segmentation(batch_size=2, run_name=run_name, location=location, hdf5_name="new_full_annotations_centered.hdf5", dims=3, num_epochs=5000, loader_num_workers=8)
    optimize_centers(location=location, run_name=run_name, batch_size=1, num_epochs=90, hdf5_target="all_data.hdf5")
    # train_morph(run_name=run_name, num_epochs=500, location=location, data_location="/data/final_dataset.hdf5", num_workers=8, log=True, mode='aided')
    # train_morph_instant(run_name=run_name, num_epochs=250, location=location, data_location="/data/final_dataset.hdf5", num_workers=8, log=True, mode='instance_aided', lr=9e-4)

