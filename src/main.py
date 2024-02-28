from src.trainings.segmentation import train_segmentation
from src.trainings.centering import optimize_centers
from src.trainings.morph import train_morph
from src.trainings.instance_morph import train_morph_instant
import wandb

if __name__ == '__main__':


    # run_name = "seg_yesflip_norot_5000"
    run_name = 'new_data_center_final'
    # location = "/home/baris/Documents/brain-morphing"
    location = "/home/imreb/brain-morphing"
    # location = "/home/baris/Documents/work/brain-morphing"

    wandb.init(project="centering_new",
               entity="barisimre",
               name=run_name,)

    # train_segmentation(batch_size=2, run_name=run_name, location=location, hdf5_name="annotated.hdf5", dims=3, num_epochs=5000, loader_num_workers=8)
    optimize_centers(location=location, run_name=run_name, batch_size=1, num_epochs=90, hdf5_target="all_new_data.hdf5")
    # train_morph(run_name=run_name, num_epochs=200, location=location, data_location="data/hdf5/ready_to_morph.hdf5", num_workers=8, log=True, mode='aided')
    # train_morph_instant(run_name=run_name, num_epochs=100, location=location, data_location="data/hdf5/ready_to_morph.hdf5", num_workers=8, log=True, mode='instance_aided', lr=1e-3, model_slow_start=False)

