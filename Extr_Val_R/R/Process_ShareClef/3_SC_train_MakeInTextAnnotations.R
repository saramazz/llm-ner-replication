# This script reads processed training text and annotation files, inserts brackets into the text
# based on annotations, and saves the updated text with annotations in an RDS file.
# Additionally, there is an optional step to write the processed text to a text file for Google translation.

# Load required libraries
library(dplyr)       # For data manipulation
source("Extr_Val_R/R/Tool_functions.R")  # Source additional functions from an external script

# Load the training text and annotations data frames
print("Loading SC_train and SC_train_annotations data frames.")
SC_train <- readRDS("Extr_Val_R/Corpora_standardized/SC_train.rds")
SC_train_annotations <- readRDS("Extr_Val_R/Corpora_standardized/SC_train_annotations.rds")
analyze_rds("Extr_Val_R/Corpora_standardized/SC_train.rds")
print("Data frames loaded successfully.")

# Insert brackets into the text based on annotations
print("Inserting brackets into the training text based on annotations.")
SC_train_loc <- SC_train %>%
  rowwise() %>%
  mutate(text = insert_brackets(text, SC_train_annotations[SC_train_annotations$id == id,], from_mod = 1))

# Save the updated text with annotations to an RDS file
print("Saving updated text with annotations to RDS file.")
saveRDS(SC_train_loc, "Extr_Val_R/Corpora_standardized/SC_train_loc.rds")
print("RDS file saved successfully.")

## (optional) Write to text file for Google translation
# print("Writing updated text to a text file for Google translation.")
# con <- file("Extr_Val_R/Corpora_standardized/SC_train_loc.txt", "wb")
# writeLines(paste0("<<<<TOM>>>>\n", SC_train_loc$text, collapse="\n"), con, useBytes=T, sep = "\n")
# close(con)
# print("Text file for Google translation created successfully.")
