# script following the vignette of ShapleyOutlier
# to extract the data examples into csv files

library(ShapleyOutlier)
library(robustHD)
library(tidyverse)

data(TopGear)

rownames(TopGear) = paste(TopGear[,1],TopGear[,2]) 
myTopGear <- TopGear[,-31] #removing the verdict variable
myTopGear <- myTopGear[,sapply(myTopGear,function(x)any(is.numeric(x)))]
myTopGear <- myTopGear[!apply(myTopGear,1, function(x)any(is.na(x))),]
myTopGear <- myTopGear[,-2]
# Transform some variables to get roughly gaussianity in the center:
transTG = myTopGear
transTG$Price = log(myTopGear$Price)
transTG$Displacement = log(myTopGear$Displacement)
transTG$BHP = log(myTopGear$BHP)
transTG$Torque = log(myTopGear$Torque)
transTG$TopSpeed = log(myTopGear$TopSpeed)

transTG <- transTG %>% rename("log(Price)" = Price, 
                              "log(Displacement)" = Displacement, 
                              "log(BHP)" = BHP, 
                              "log(Torque)" = Torque, 
                              "log(TopSpeed)" = TopSpeed)

X <- as.matrix(transTG)
X <- robStandardize(X)
# X is preprocessed data, save to csv
as.data.frame(X) |> 
  rownames_to_column("ID") |> 
  write_csv("data_topgear.csv")

# for comparison we also export robust estimates
# for mean and covariance
set.seed(1)
MCD <- covMcd(X, nsamp = "best")
#> Warning in .fastmcd(x, h, nsamp, nmini, kmini, trace = as.integer(trace)): 'nsamp = "best"' allows maximally 100000 subsets;
#> computing these subsets of size 12 out of 245
mu <-MCD$center
Sigma <- MCD$cov
Sigma_inv <- solve(MCD$cov)

# writing to file
write.csv(mu, "center_topgear.csv",
          row.names = F)
write.csv(Sigma, "sigma_topgear.csv", 
          row.names = F)

############## second example ###################
data("WeatherVienna")

weather_summer <- WeatherVienna %>% dplyr::select(-c(`t`, t_max, t_min, p_max, p_min)) %>%
  drop_na() %>%
  filter(month %in% c("JUN", "JUL", "AUG")) %>%
  filter(year >= 1955) %>% 
  group_by(year) %>%
  dplyr::select(-month) %>% 
  summarise(across(.cols = everything(), function(x) mean(x)))

X <- weather_summer %>% dplyr::select(-c(num_frost, num_ice, year))
rownames(X) <- weather_summer$year
#> Warning: Setting row names on a tibble is deprecated.
X <- robStandardize(X)
as.data.frame(X) |> 
  rownames_to_column("year") |> 
  write_csv("data_weather.csv")

set.seed(1)
MCD <- covMcd(X, alpha = 0.5, nsamp = "best")
#> Warning in .fastmcd(x, h, nsamp, nmini, kmini, trace = as.integer(trace)): 'nsamp = "best"' allows maximally 100000 subsets;
#> computing these subsets of size 17 out of 68
mu <-MCD$center
Sigma <- MCD$cov
Sigma_inv <- solve(MCD$cov)

# writing to file
write.csv(mu, "center_weather.csv",
          row.names = F)
write.csv(Sigma, "sigma_weather.csv", 
          row.names = F)
