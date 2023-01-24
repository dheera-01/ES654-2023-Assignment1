We varied the max_depth to compare results of our model with sklearn models. To measure RMSE we spillted data into traning and testing data in 80:20 ratio.

As our model can only take either discrete input or real inputs (not both in same model), we dropped dicrete valued features such as 'origin', 'cylindes' 'car name' and 'model year'.

The following are comparsion results that we got by varying max_depth from 1 to 8.

![image](https://user-images.githubusercontent.com/77908454/214228285-64ce2466-52c8-44de-a021-fbb865ff356e.png)
