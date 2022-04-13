# Hephaestus Dataset
Work in progress.

## Dataset Description
A detailed description of the dataset can be seen in the original [Hephaestus](arxivlink) paper.

The raw InSAR data can be found [here](dropboxlink).

The dataset is organized in the following structure:
```
|- FrameID (e.g 124D_05291_081406)
|    |-- interferograms
|    |    |---  Dates (e.g 20181128_20181204)
|    |    |    |--- InSAR.png
|    |    |    |--- Coherence.png
```

### Annotation

The dataset contains both labeled and unlabeled data. The labeled part covers 38 frames summing up to 19,919 annotated InSAR.
The list of the studied volcanoes, along with the temporal distribution of their samples can be seen below. ![below](volcano_distribution.png)

Each labeled InSAR is accompanied by a json file containing the annotation details. Below we present an example of an annotation file. A detailed description can be seen in the original paper (section 2.2):
```json
{
  "uniqueID": 19912, (ID given to the annotation file.)
  "frameID": "103A_07010_121313", (ID of the InSARs frame. This is a unique location identifier.)
  "primary_date": "20210414", (Caption date of the primary SAR image.)
  "secondary_date": "20210426", (Caption date of the secondary SAR image.)
  "corrupted": 0, (Flag for corrupted InSAR.)
  "processing_error": 0, (Error attributed to InSAR processing.)
  "glacier_fringes": 0, (Flag to identify the existence of glaciers)
  "orbital_fringes": 0, (Fringes attributed to orbital errors.)
  "atmospheric_fringes": 2, (Fringes attributed to atmospheric effects. Its value ranges from 0 to 3 with 0 denoting is absense.) 
  "low_coherence": 0, (Flag to denote low coherence.)
  "no_info": 0, (Flag to identify InSAR with very low information signal.)
  "image_artifacts": 0, (Flag to identify InSAR with artificats unrelated to interferograms.)
  "label": [ 
    "Non_Deformation" (Labels. May contain multiple elements.)
  ],
  "activity_type": [], (Activity type of each ground deformation pattern.)
  "intensity_level": "None", (Intensity of the event.)
  "phase": "Rest", (Phase of the volcano. Rest/Unrest/Rebound.)
  "confidence": 0.8, (Confidence of the annotator for this annotation.)
  "segmentation_mask": [], (List of polygons containing the ground deformation patterns.)
  "is_crowd": 0, (Flag to denote whether there are multiple ground deformation patterns in the InSAR.)
  "caption": "Turbulent mixing effect or wave-like patterns caused by liquid and solid particles of the atmosphere can be detected around the area. No deformation activity can be detected."
}
```