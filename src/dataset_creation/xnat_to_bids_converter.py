import pandas as pd
from os import path, makedirs
import shutil
from monai.transforms import LoadImaged
import nibabel as nib
from glob import glob
from src.dataset_creation.SegmentationImage import SegmentationImage, SegmentationLabelError
from warnings import warn
from datetime import datetime


def convert(bids_path, xnat_path, annotations_path, meta_path):
    # bids_path = '/Users/elina/Documents/bids'  # where the results are stored
    # xnat_path = '/Users/elina/Documents/XNAT_data'  # where XNAT raw data was dumped
    # annotations_path = '/Users/elina/Documents/annotations/2023-12-04_10-39-55'  # where XNAT annotation data was dumped
    # meta_path = '/Users/elina/Documents/raw_meta'  # where raw meta-data is stored (csv files)
    makedirs(bids_path, exist_ok=True)

    file_path = path.realpath(__file__)
    keys_path = path.join(path.dirname(file_path), "keys_description.json")
    shutil.copy(keys_path, path.join(bids_path, "keys_description.json"))

    readme_path = path.join(path.dirname(file_path), "README.md")
    shutil.copy(readme_path, path.join(bids_path, "README.md"))

    # Descriptions in the NifTi file names
    desc_dict = {
        "large_series": "slicethickness-large_registered-false",
        "small_series": "slicethickness-small_registered-false",
        "registered_series": "slicethickness-large_registered-true",
    }

    # List of annotations which should be converted (others will be ignored)
    allowed_names = ["Janne_annotation", "Sanne_annotation"]

    meta_scans_df = pd.read_csv(path.join(meta_path, "scans.csv"))
    meta_scans_df.set_index(["ID", "annotation_session"], inplace=True)

    loader = LoadImaged(keys=["image"], ensure_channel_first=True, image_only=False)

    # Extract images and scans meta-data
    for patient_id, session_id in meta_scans_df.index.values:
        print(patient_id, session_id)

        # Read image data
        xnat_session_path = path.join(xnat_path, f"{patient_id}_CT_{session_id[2::]}")
        bids_session_path = path.join(bids_path, f"sub-{patient_id}", f"ses-{session_id}", "ct")
        makedirs(bids_session_path, exist_ok=True)

        bids_scans_df = pd.DataFrame()

        for series_name in ["small_series", "large_series", "registered_series"]:
            series_nb = meta_scans_df.loc[(patient_id, session_id), series_name]
            xnat_series_path = None if pd.isna(series_nb) else path.join(xnat_session_path, str(int(series_nb)),
                                                                         "DICOM")

            if xnat_series_path is not None and path.exists(xnat_series_path):
                image_dict = loader({"image": xnat_series_path})
                image_nii = nib.Nifti1Image(
                    image_dict["image"][0].numpy(),
                    affine=image_dict["image_meta_dict"]["affine"].numpy()
                )
                desc = desc_dict[series_name]
                image_filename = f'sub-{patient_id}_ses-{session_id}_{desc}_ct.nii.gz'
                nib.save(image_nii, path.join(bids_session_path, image_filename))

                # Add meta-data
                slice_thickness = image_dict["image_meta_dict"]["spacing"][-1]
                row_df = pd.DataFrame(
                    [[path.join("ct", image_filename), series_nb, slice_thickness, "n/a", False]],
                    columns=["filename", "original_series_number", "slice_thickness", "original_seg_filename",
                             "interpolation"]
                )
                bids_scans_df = pd.concat((bids_scans_df, row_df))

        # Read segmentation data
        seg_path_list = glob(
            path.join(annotations_path, f"{patient_id}_CT_{session_id[2::]}", "SEG*", "SEG", "SEG*.dcm"))

        for seg_dcm_path in seg_path_list:
            xnat_id = seg_dcm_path.split(path.sep)[-4]
            seg_id = seg_dcm_path.split(path.sep)[-3]
            series_nb = int(seg_id.split("_")[-1][1::])

            xnat_series_path = path.join(xnat_path, xnat_id, str(series_nb), "DICOM")
            try:
                image_seg = SegmentationImage(seg_dcm_path, xnat_series_path)
                annotation_name = image_seg.annotation_name

                if annotation_name in allowed_names:
                    bids_session_path = path.join(bids_path, f"sub-{patient_id}", f"ses-{session_id}", "annotation")
                    annotator_index = allowed_names.index(annotation_name)
                    makedirs(bids_session_path, exist_ok=True)
                    # Find corresponding series in scans csv to find the correct description
                    series_name = meta_scans_df.apply(
                        lambda row: row[row == series_nb].index, axis=1
                    ).loc[(patient_id, session_id)].item()
                    desc = desc_dict[series_name]
                    annotation_filename = f"sub-{patient_id}_ses-{session_id}_" \
                                          f"annotator-{annotator_index}_{desc}_annotation.nii.gz"
                    image_seg.save(path.join(bids_session_path, annotation_filename))
                    print("Saved annotation")

                    # Add meta-data
                    row_df = image_seg.compute_scans_tsv(annotation_filename)
                    bids_scans_df = pd.concat((bids_scans_df, row_df))

                    # # Interpolate to small slice thickness domain if large thickness was annotated
                    # if series_name == "large_series":
                    #     small_series_nb = meta_scans_df.loc[(patient_id, session_id), "small_series"]
                    #     image_seg.resize_to_target(path.join(xnat_session_path, str(int(small_series_nb)), "DICOM"))
                    #
                    #     desc = desc_dict["small_series"]
                    #     annotation_filename = f"sub-{patient_id}_ses-{session_id}_" \
                    #                           f"annotator-{annotator_index}_{desc}_annotation.nii.gz"
                    #     image_seg.save(path.join(bids_session_path, annotation_filename))
                    #
                    #     # Add meta-data
                    #     row_df = image_seg.compute_scans_tsv(annotation_filename)
                    #     bids_scans_df = pd.concat((bids_scans_df, row_df))

            except AttributeError:
                warn(f"Segmentation file of patient {patient_id} session {session_id} could not be extracted.")
            except SegmentationLabelError:
                pass

        bids_scans_df.to_csv(
            path.join(bids_path, f"sub-{patient_id}", f"ses-{session_id}",
                      f"sub-{patient_id}_ses-{session_id}_scans.tsv"),
            sep="\t", index=False)

    # Read raw meta data
    raw_demographics_df = pd.read_csv(path.join(meta_path, "demographics.csv"))  # Downloaded from XNAT - Subjects tab
    raw_sessions_df = pd.read_csv(path.join(meta_path, "sessions.csv"))  # Downloaded from XNAT - CT Sessions tab
    raw_diagnosis_df = pd.read_csv(path.join(meta_path, "diagnosis.csv"))  # Sent by Jorieke
    raw_scans_df = pd.read_csv(path.join(meta_path, "scans.csv"))  # Manually created (Elina)

    # Rename columns and set index to match BIDS format
    raw_scans_df.rename({"ID": "participant_id", "side": "hematoma_side"}, axis=1, inplace=True)
    raw_scans_df.set_index("participant_id", inplace=True)
    side_df = raw_scans_df[["hematoma_side"]].copy()
    side_df.dropna(inplace=True)

    raw_diagnosis_df.rename(
        {
            "Study ID": "participant_id",
            "Was surgery performed after initial diagnosis? (0 = no, 1 = yes)": "surgery",
            "Was there a recurrence? (0 = no, 1 = yes)": "recurrence"
        },
        axis=1,
        inplace=True
    )
    raw_diagnosis_df.set_index("participant_id", inplace=True)
    surgery_recurrence_df = raw_diagnosis_df[["surgery", "recurrence"]].copy()
    surgery_recurrence_df["surgery"] = surgery_recurrence_df.surgery.astype(bool)
    surgery_recurrence_df["recurrence"] = surgery_recurrence_df.recurrence.astype(bool)

    raw_sessions_df.set_index('XNAT_CTSESSIONDATA ID', inplace=True)

    # Add participant-level meta-data
    participants_df = raw_demographics_df.copy()
    participants_df.rename({"Subject": "participant_id", "M/F": "sex", "Age": "age"}, axis=1, inplace=True)
    participants_df.set_index("participant_id", inplace=True)
    participants_df = participants_df.merge(side_df, on="participant_id")
    participants_df = participants_df.merge(surgery_recurrence_df, on="participant_id")

    participants_df.to_csv(path.join(bids_path, "participants.tsv"), sep="\t")

    # Add session-level meta-data
    for patient_id, patient_df in meta_scans_df.groupby(level=0):
        sessions_df = pd.DataFrame()
        for _, session_id in patient_df.index.values:
            session_date = datetime.strptime(raw_sessions_df.loc[f"{patient_id}_CT_{session_id[2::]}", "Date"],
                                             "%Y-%m-%d")
            if patient_id in raw_diagnosis_df.index.values:
                diagnosis_date = datetime.strptime(raw_diagnosis_df.loc[patient_id, "Date CSDH diagnosis"], "%d.%m.%Y")
                raw_postop_date = raw_diagnosis_df.loc[patient_id, "Date of postoperative CT"]

                # Compare dates to know what the label of the session should be
                if session_date < diagnosis_date:
                    session_label = "prediag"
                elif session_date == diagnosis_date:
                    session_label = "diag"
                elif isinstance(raw_postop_date, str) and datetime.strptime(raw_postop_date,
                                                                            "%d.%m.%Y") == session_date:
                    session_label = "postop"
                else:
                    session_label = "followup"
            else:
                session_label = "unknown"

            row_df = pd.DataFrame([[session_id, session_label]], columns=["session_id", "session_label"])
            sessions_df = pd.concat((sessions_df, row_df))
        sessions_df.to_csv(
            path.join(bids_path, f"sub-{patient_id}", f"sub-{patient_id}_sessions.tsv"),
            sep="\t",
            index=False
        )


if __name__ == '__main__':
    location = "/home/baris/Desktop/xnat"

    convert(
        bids_path=f"{location}/bids",
        meta_path=f"{location}/meta",
        xnat_path=f"{location}/xnat_new",
        annotations_path=f"{location}/xnat_annotations",
    )
