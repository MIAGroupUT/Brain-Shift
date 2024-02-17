from src.trainings.segmentation import train_segmentation
from src.trainings.centering import optimize_centers
from src.pipeline.segmentation import  infer_segmentation
import wandb

if __name__ == '__main__':


    run_name = "seg_yesflip_norot_5000"
    # location = "/home/baris/Documents/brain-morphing"
    location = "/home/imreb/brain-morphing"
    # location = "/home/baris/Documents/work/brain-morphing"

    wandb.init(project="annotations",
               entity="barisimre",
               name=run_name)

    train_segmentation(batch_size=2, run_name=run_name, location=location, slice_thickness="large", dims=3, num_epochs=5000)
    # optimize_centers(location=location, run_name=run_name, batch_size=1, num_epochs=200, slice_thickness='large')

