# Name: Minhwa Lee
# Assignment: CS200 Junior IS Software
# Title: Software for Data Cleaning for Suicide Data set
# Course: CS 200
# Semester: Spring 2020
# Instructor: D. Byrnes
# Date: 04/25/2020
# Sources consulted: 'https://github.com/ashkanmradi/MLP-classifier/blob/master/main.py', 'pandas' and 'scikit-learn' library documentation website
# https://pandas.pydata.org/docs/reference/index.html,https://scikit-learn.org/stable/modules/classes.html
# Program description: This program is to execute data classification of iris plants using multilayer perceptron with propagation.
#                       It is written on Python Scikit-learn library.
# Known bugs: I've used 'warnings' to ignore all the unimportant warnings during running the program.
# Creativity: Except for for-loop codes that make a plot, all remaining codes are written by myself.
# Instructions: Just run the program then you will see the accuracy information as well as the plot.


# 1. Import dplyr library and Open raw data

library(dplyr)

kyrbs2019_edit <- read.csv("~/Desktop/JuniorIS Software/kyrbs2019_edit.csv")
dat <- kyrbs2019_edit # dat is a copy of the original data set.

# 2. Data Cleaning and Renaming

# Parents educational attainment
dat <- dat[dat$E_EDU_F != 8888, ] # Remove un-related response for father column (non-response)
dat <- dat[dat$E_EDU_F != 9999, ] # (Single-mother family)

dat <- dat[dat$E_EDU_M != 8888, ] # Remove un-related response for mother column (non-response)
dat <- dat[dat$E_EDU_M != 9999, ] # (single-father family)

# Sexual Experience

dat <- dat[dat$S_SI != 9999, ] # Remove unrelated response ('n/a')

# School Violence Experience

dat$V_TRT[dat$V_TRT %in% c(2,3,4,5,6,7)] <- 2 # For making a binary response, we convert all 2- 7 values to 2 for saying 'yes'
                                              # For a response 'no' we keep 1.

# Physical Activity Frequence

dat$PA_VIG[dat$PA_VIG %in% c(1,2,3)] <- 1 # For those who are not frequently working out
dat$PA_VIG[dat$PA_VIG %in% c(4,5,6)] <- 2 # For those who work out frequently

# 4. Data Republishing with the newly updated dataframe
write.csv(dat, 'kyrbs2019_edit_1.csv')
