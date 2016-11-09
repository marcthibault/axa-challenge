library(data.table)
library(ggplot2)

# Import the whole data
# data = read.csv("../train_2011_2012_2013.csv", sep = ";")
# Nlines = 2607657
# Extracting a random sample
# sample_data = data.table(data[sample(1:nrow(data), 10000), ])

# Importing the initial sample
sample_data = data.table(read.csv("../sample.csv", sep = ";"))
sample_data = sample_data[sample(1:1000000, 10000), ]
# Test selecting relevant columns
sample_data = sample_data[, list(DATE, DAY_OFF, DAY_DS, WEEK_END, DAY_WE_DS, TPER_TEAM, TPER_HOUR, ASS_ASSIGNMENT, ASS_DIRECTORSHIP, ASS_PARTNER, ASS_POLE, ASS_SOC_MERE, CSPL_RECEIVED_CALLS)]

# Adding the DAY and TIME datas
sample_data[, DAY := as.IDate(DATE, "%Y-%m-%d")]
sample_data[, TIME := as.ITime(DATE, "%H:%M:%S")]

# Watching some features on the data :
# irrelevant columns & missing values
for(i in 1:ncol(sample_data))
{
  cat("Column ", i, " : ", names(sample_data)[i], "\n")
  cat(" Missing values : ", sum(is.na(sample_data[[i]])))
  nuniques = length(unique(sample_data[[i]]))
  if(nuniques == 1)  cat(" ; also has only 1 value")
  cat("\n")
}

plot(sample_data[, mean(CSPL_RECEIVED_CALLS), by = TIME], sample_data[, mean(CSPL_RECEIVED_CALLS) + sqrt(var(CSPL_RECEIVED_CALLS)), by = TIME])

ggplot(sample_data[, list(mean = mean(CSPL_RECEIVED_CALLS), var = var(CSPL_RECEIVED_CALLS)), by = TIME], aes(x = TIME, y = mean)) +
  geom_point() +
  geom_errorbar(aes(ymin = mean - sqrt(var), ymax = mean + sqrt(var)), width=.1)
