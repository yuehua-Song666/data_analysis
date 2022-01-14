# Data pre-processing -- imbalanced data #
#### Resampling methods are designed to change the composition of a training dataset for an imbalanced classification task. ####
- Library: Imbalanced_Learn Library
	- [https://github.com/scikit-learn-contrib/imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn "Imbalanced learn library")
- Methods
	- Select Examples to keep
		- Near Miss Undersampling
			- Paper: [https://www.site.uottawa.ca/~nat/Workshop2003/jzhang.pdf](https://www.site.uottawa.ca/~nat/Workshop2003/jzhang.pdf "KNN approach to unbalanced data distributions")
			- NearMiss-1 -- Select negative examples that are close to some of the positive examples. (In the paper they select negative examples whose average diatance to the three closest positive examples are the smallest.)
				- Images:
				- ![](https://github.com/yuehua-Song666/data_analysis/blob/main/data_processing/img/Imbalanced_data_examples.png) ![](https://github.com/yuehua-Song666/data_analysis/blob/main/data_processing/img/NearMiss1.png)
			- NearMiss-2 -- Select negative examples that are close to all the positive exsamples. (In the paper examples are selected based on their average distances to three farthest positive examples.)
			- NearMiss-3 -- Select a given number of the closest negative examples for each positive examples. This method guarantees every positive example is surrounded by some negative examples. (In this paper, they choose the negative examples whose average distances to the closest three positive examples are the farthest.) 
