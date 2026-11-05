# This script reads processed text and annotation files, inserts brackets into the text
# based on annotations, and saves the updated text with annotations in an RDS file.
# Additionally, there is an optional step to write the processed text to a text file for Google translation.

# Load required libraries
library(dplyr)       # For data manipulation
source("Extr_Val_R/R/Tool_functions.R")  # Source additional functions from an external script

# Load the text and annotations data frames
print("Loading SC_test and SC_test_annotations data frames.")
SC_test <- tryCatch({
  readRDS("Extr_Val_R/Corpora_standardized/SC_test.rds")
}, error = function(e) {
  print(paste("Error loading SC_test:", e$message))
  NULL
})

SC_test_annotations <- tryCatch({
  readRDS("Extr_Val_R/Corpora_standardized/SC_test_annotations.rds")
}, error = function(e) {
  print(paste("Error loading SC_test_annotations:", e$message))
  NULL
})

# Check if data frames are loaded successfully
if (is.null(SC_test) || is.null(SC_test_annotations)) {
  stop("One or both of the data frames failed to load. Please check the file paths and contents.")
}
print("Data frames loaded successfully.")

# Insert brackets into the text based on annotations
print("Inserting brackets into the text based on annotations.")
SC_test_loc <- tryCatch({
  SC_test %>%
    rowwise() %>%
    mutate(text = insert_brackets(text, SC_test_annotations[SC_test_annotations$id == id,], from_mod = 1))
}, error = function(e) {
  print(paste("Error inserting brackets:", e$message))
  NULL
})

# Check if text processing was successful
if (is.null(SC_test_loc)) {
  stop("Failed to insert brackets. Please check the code and data.")
}

# Save the updated text with annotations to an RDS file
print("Saving updated text with annotations to RDS file.")
tryCatch({
  saveRDS(SC_test_loc, "Extr_Val_R/Corpora_standardized/SC_test_loc.rds")
  print("RDS file saved successfully.")
}, error = function(e) {
  print(paste("Error saving RDS file:", e$message))
})

## (optional) Write to text file for Google translation
# print("Writing updated text to a text file for Google translation.")
# tryCatch({
#   con <- file("Extr_Val_R/Corpora_standardized/SC_test_loc.txt", "wb")
#   writeLines(paste0("<<<<TOM>>>>\n", SC_test_loc$text, collapse="\n"), con, useBytes=T, sep = "\n")
#   close(con)
#   print("Text file for Google translation created successfully.")
# }, error = function(e) {
#   print(paste("Error writing text file:", e$message))
# })
