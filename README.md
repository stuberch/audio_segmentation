# Background
This project is part of a master thesis researching 'automated segmentation of the Ehrenreich opera collection' written at the Bern University of Applied Science. Goal of the script is to automatically identify  structural audio sequences via a reference recording. The method is called *reference-based audio segmentation* by [Prätzlich & Müller 2013](https://www.researchgate.net/publication/303667411_Freischutz_Digital_a_case_study_for_reference-based_audio_segmentation_of_operas).

# Data
The dataset consists of 23 different versions of Mozarts opera *Die Zauberflöte* stored in the [Ehrenreich collection](https://ehrenreich.bfh.science/data/) (not open to public). To test the script 1'318 manually annotated segments where used, that can be found in [assets](https://github.com/stuberch/audio_segmentation/tree/main/assets).

# Paper
The paper is available opon request via the [author](mailto:christianlukas.stuber@students.bfh.ch) or the [supervisor](mailto:eduard.klein@bfh.ch).

# Requirements
* Python 3.8
* librosa 0.8.0
* matplotlib (only for the notebooks)

# Audio Segmentation Script
The final script is [audioSegmentation_A4.py](https://github.com/stuberch/audio_segmentation/blob/main/audioSegmentation_A4.py) in connection with [supportClasses.py](https://github.com/stuberch/audio_segmentation/blob/main/supportClasses.py). All other files are previous versions or test cases of different functions of the script.

* [Version A1 - Basis](https://github.com/stuberch/audio_segmentation/blob/main/audioSegmentation_A1.py)
* [Version A2 - Semitone Tuning](https://github.com/stuberch/audio_segmentation/blob/main/audioSegmentation_A2.py)
* [Version A3 - Constraints](https://github.com/stuberch/audio_segmentation/blob/main/audioSegmentation_A3.py)
* [Version A4 - Parameter Tweaking (final)](https://github.com/stuberch/audio_segmentation/blob/main/audioSegmentation_A4.py)

The script takes a reference version (.wav), reference segmentation (.csv) and searches for the same audio sequences in the unknown versions (.wav) in the assets folder. If test data (.csv) of the unknown versions is provided, the identified segments are automatically tested. The values are calculated based on the similarity (*Jaccard Index*) of the identified segments in comparrison to the targed segments. Output of the script are the two files *results.csv*, containing the automatically annotated segments and *log.csv*, measuring the performance of the script.
