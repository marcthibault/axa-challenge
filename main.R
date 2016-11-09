library(data.table)

# Import the whole data
# data = read.csv("../train_2011_2012_2013.csv", sep = ";")
# Extracting a random sample
# sample_data = data.table(data[sample(1:nrow(data), 10000), ])

# Importing the initial sample
sample_data = data.table(read.csv("../sample.csv", sep = ";"))

print(sample_data[1:3, ])
print(dim(sample_data))

for(i in 1:ncol(sample_data))
{
  nuniques = length(unique(sample_data[[i]]))
  if(nuniques == 1)
  {
    cat("Column ", i, " : ", names(sample_data)[i] , " has only 1 value" , "\n")
  }
}
