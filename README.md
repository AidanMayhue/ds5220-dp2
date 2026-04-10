#Data Project II


This project utilizes a Cron job via kubernetes to collect tide data from NOAA into an S3 bucket and plot expected sea level against actual sea level.

##Summary of Data Source
The NOAA tides and currents API collects observed and predicted water levels at tide stations. This API is updated every six minutes. I chose to use the center in Norfolk Virginia due to proximity to the University of Virginia. This source did not require an API key.

##Process Explanation
Every 6 minutes a Kubernetes CronJob on an EC2 instance utilizies a containerized python script to call the NOAA API and classify the current reading as above, below, or near prediction. Special categories exist for storm weather. The data is then written to a DynamoDB table and updated in an S3 bucket containing a plot and a CSV of the predictions vs measurement.

##Description of Data and Plot
The output data contains a series of columns covering the observed sea level, the station id, predicted sea level, how the level was classified, the surge height, and the timestamp. The plot mirrors this information by including the time on the x axis and height on the y axis. Predicted values are compared against actual values. Storm surge events are also plotted.
