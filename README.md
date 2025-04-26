# Omnipresent computing

This is the code repository for the manuscript: [Harnessing omnipresent oscillator networks as computational resource](https://arxiv.org/abs/2502.04818) by T.G. de Jong, H. Notsu and K. Nakajima. 

**About the file structure:** The material is subdivided following the structure of the manuscript. Experiments for the main document can be found in the folder _main_ and experiments for the supplementary information can be found in the folder _supp_. The notable exceptions to this structure are the music producing oscillator networks and the additional time-series tests which can both be found in the supp folder.

**About the code:** The code has been mainly written inside notebooks. However, code that requires substantial computational time is in py-files. These py-files generate shelves with the experiments data. For convenience I have already included all pre-generated shelves.     

**About the data:** The time-series already have been generated, but generate_data.py can be used to generate them again. Because most of the underlying time-series correspond to chaotic systems the results might vary a bit depending on the module versions you use. 

**About code updates:** As code refinements and follow-up projects are on their way the current version should really be viewed as running in beta.

