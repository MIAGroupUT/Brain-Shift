from src.data_loading.datasets import AllBidsDataset, AnnotatedBidsDataset
from monai.data import DataLoader
from src.utils.general import add_result_to_hdf5
from tqdm import tqdm


def all_bids_to_hdf5(bids_location, hdf5_path, slice_thickness, exclude_registered):
    dataset = AllBidsDataset(bids_location, slice_thickness=slice_thickness, caching=False,
                             exclude_registered=exclude_registered)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for item in tqdm(dataloader):
        tqdm.write(item['name'][0])
        n = {}
        for k in item:
            # TODO: make sure of typing as well
            n[k] = item[k]

        add_result_to_hdf5(n, hdf5_path)


def annotated_bids_to_hdf5(bids_location, hdf5_path, slice_thickness, exclude_registered):
    dataset = AnnotatedBidsDataset(bids_location, slice_thickness=slice_thickness, caching=False,
                                   exclude_registered=exclude_registered)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for item in tqdm(dataloader):
        tqdm.write(item['name'][0])

        n = {}
        for k in item:
            # TODO: make sure of typing as well
            n[k] = item[k][0]

        add_result_to_hdf5(n, hdf5_path)


if __name__ == '__main__':
    location = "/home/baris/Documents/brain-morphing"
    # all_bids_to_hdf5(
    #     bids_location=f"{location}/data/bids",
    #     hdf5_path=f"{location}/data/hdf5/all_bids_large.hdf5",
    #     exclude_registered=False,
    #     slice_thickness="large"
    # )

    annotated_bids_to_hdf5(
        bids_location=f"/home/baris/Desktop/xnat/bids",
        hdf5_path=f"{location}/data/hdf5/new_annotations.hdf5",
        exclude_registered=False,
        slice_thickness="large"
    )
