# StarCraftDataAnalysis
Classifier for competitive StarCraft league ranking

### Description
The contents include about 3400 instances of individual matches within the game and the person that was playing. All the data is tied to the person, where each person has an age, league, and number of hours they spend playing the game each week along with the total number of hours they have played in their lifetime since the game came out in 2010. There are 21 attributes in the dataset total that represents the players statistics for the individual game played. I will try to predict the league a player is in by using the data from individual games.

### Process
All of the data is continuous with the exception of the league a player is in. Summary statistics were calculated for each attribute per league, while the most notable was the number of instances per league, where most of the instances had a player in league four or five. This may give false positives later if it were to just guess league 4 or 5, as it would be right a little less than 25% of the time by default.

To determine what attributes were useful and which were not, I graphed each attribute against the league that the player plays in. The useless attributes were: Max time stamp, complex abilities used, complex units made, unique units made, workers made, and total hours played.
These did not vary much for each league.

The other attributes were deemed useful. The plots can be seen in the pictures folder.

An example of a useless attribute:

![UniqueUnitsMade](https://github.com/soaresdominic/StarCraftDataAnalysis/blob/master/Pics/UniqueUnitsMade.jpg)

An example of a useful attribute:

![AssignToHotkeys](https://github.com/soaresdominic/StarCraftDataAnalysis/blob/master/Pics/AssignToHotkeys.jpg)

### Classifiers
The first classifier implemented was the k nearest neighbor classifier. Using all attributes, the k nearest neighbors algorithm gave an accuracy from 35%-40%. The accuracy for +-1 league was 85%.

![k-NN](https://github.com/soaresdominic/StarCraftDataAnalysis/blob/master/Pics/knnMatrix.jpg)


The next classifier implemented was an ensemble of quadratic regressions. After processing the data and visualizing it using boxplots, a few attribute’s means for each league was on a curve, shown above in AssignToHotkeys. The 8 data points were plotted in MATLAB, and using the regression analysis tool to find the quadratic equation for the curve for the three attributes, APM, Assign to Hotkeys, and Action Latency. When classifying league, majority voting from each equation’s result is used, and if the values were all different, the middle value is picked.
This algorithm gave an accuracy of 33%, and 75% for +-1 league.

![ensemble](https://github.com/soaresdominic/StarCraftDataAnalysis/blob/master/Pics/ensembleMatrix.jpg)


The last classifier implemented was a decision tree classifier. The data was normalized to discretize it to make it categorical and possible to use a decision tree.
This algorithm gave an accuracy of 50%, and 80% for +-1 league.

![decision](https://github.com/soaresdominic/StarCraftDataAnalysis/blob/master/Pics/decisionMatrix.jpg)

### Conclusion
The attributes that set apart the players in different leagues were not perfect for classification, as many people with unique or non-standard play styles made a lot of outliers for each data attribute. The outliers made classifying the exact league difficult, but the algorithms were very accurate for being within one league when classifying.

The decision tree having a 50% accuracy was impressive, considering how close the values for each attribute are from league to league. The accuracy for +-1 league in all classifiers was about 80%, showing that if the bins for league were, 1-3,4-5,7-8, the accuracy would be exceptional.
