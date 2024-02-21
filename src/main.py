from src.trainings.segmentation import train_segmentation
from src.trainings.centering import optimize_centers
from src.pipeline.segmentation import  infer_segmentation
from src.trainings.morph import train_morph
import wandb

if __name__ == '__main__':


    # run_name = "seg_yesflip_norot_5000"
    run_name = 'ssim_aggression_more_jacobian'
    # location = "/home/baris/Documents/brain-morphing"
    location = "/home/imreb/brain-morphing"
    # location = "/home/baris/Documents/work/brain-morphing"

    wandb.init(project="morph_new",
               entity="barisimre",
               name=run_name,
               notes="1 times l1, enabled ventricle wrong side jacobain size on z is 3.0 2 times overlap 10 jacobian")

    # train_segmentation(batch_size=2, run_name=run_name, location=location, slice_thickness="large", dims=3, num_epochs=5000, loader_num_workers=8)
    # optimize_centers(location=location, run_name=run_name, batch_size=1, num_epochs=200, slice_thickness='large')
    train_morph(run_name=run_name, num_epochs=500, location=location, data_location="/data/final_dataset.hdf5", num_workers=8, log=True)

