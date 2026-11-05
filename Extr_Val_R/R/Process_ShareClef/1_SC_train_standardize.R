# This script processes text and annotation files from the CLEF2014 dataset.
# It reads text files into a data frame, saves it in RDS and CSV formats.
# It also processes annotation files, cleans and formats the annotations,
# and saves the processed annotations in an RDS file.

# Load required libraries
library(dplyr)       # For data manipulation
library(readr)       # For reading/writing CSV files
source("Extr_Val_R/R/Tool_functions.R")  # Source additional functions from an external script

# Set the location of the text files
location <- "Extr_Val_R/Corpora_raw/CLEF2014/2014ShAReCLEFeHealthTasks2_training_10Jan2014/2014ShAReCLEFeHealthTask2_training_corpus/"

# List all files in the directory
files <- list.files(location)
print(paste("Found", length(files), "files in the location."))

# Filter the list to include only .txt files
files <- files[stringr::str_detect(files, ".txt")]
print(paste("Filtered to", length(files), "text files."))

#### TEXT PROCESSING
filetxts <- ""   # Initialize an empty string for storing text
SC_train <- NULL  # Initialize an empty data frame to store text data

# Loop through each text file
for (file in files) {
  fileName <- paste0(location, file)  # Create full file path
  filetxt <- readChar(fileName, file.info(fileName)$size)  # Read the entire file content
  SC_train <- rbind(SC_train, data.frame(id = file, text = filetxt))  # Append file content to data frame
}
print("Text processing complete. Saving to RDS and CSV formats.")

# Before writing, check the type of SC_train
print(class(SC_train))  # Should return "data.frame"

# Save the text data frame to an RDS file
saveRDS(SC_train, "Extr_Val_R/Corpora_standardized/SC_train.rds")
print("Text data saved to SC_train.rds")

##### EXPORT FOR PYTHON
# Write the text data frame to a CSV file for use in Python
write_csv(SC_train, file = "Extr_Val_R/Corpora_python_ready/SC_train_eng.csv", eol = "\n")
print("Text data exported to SC_train_eng.csv")

#### ANNOTATIONS
# Set the location of the annotation files
ann_location <- "Extr_Val_R/Corpora_raw/CLEF2014/2014ShAReCLEFeHealthTasks2_training_10Jan2014/2014ShAReCLEFeHealthTask2_training_pipedelimited/"

# List all annotation files in the directory
ann_files <- list.files(ann_location)
print(paste("Found", length(ann_files), "annotation files in the location."))

# Filter the list to include only .txt files
ann_files <- ann_files[stringr::str_detect(ann_files, ".txt")]
print(paste("Filtered to", length(ann_files), "annotation files."))

SC_train_annotations <- NULL  # Initialize an empty data frame to store annotations

# Loop through each annotation file
for (file in ann_files) {
  dat <- read.csv(paste0(ann_location, file), sep = "|", header = F, encoding = 'UTF-8')  # Read the annotation file
  colnames(dat) <- c("DD_DocName", "DD_Spans", "DD_CUI", "Norm_NI", "Cue_NI", "Norm_SC", "Cue_SC", 
                      "Norm_UI", "Cue_UI", "Norm_CC", "Cue_CC", "Norm_SV", "Cue_SV", "Norm_CO", 
                      "Cue_CO", "Norm_GC", "Cue_GC", "Norm_BL", "Cue_BL", "Norm_DT", "Norm_TE", "Cue_TE")  # Assign column names
  SC_train_annotations <- rbind(SC_train_annotations, dat)  # Append data to the annotations data frame
}
print("Annotation files processed. Cleaning and formatting annotations.")

# Remove the last column from the annotations data frame
SC_train_annotations <- select(SC_train_annotations, -ncol(SC_train_annotations))

# Process the annotations data frame
SC_train_annotations <- SC_train_annotations %>%
  tidyr::separate_longer_delim(DD_Spans, ",") %>%   # Split the DD_Spans column by commas
  mutate(DD_Spans = trimws(DD_Spans),                # Trim whitespace from DD_Spans
         DD_CUI = trimws(DD_CUI)) %>%                # Trim whitespace from DD_CUI
  tidyr::separate(DD_Spans, c("from", "to"), "-") %>% # Separate the spans into 'from' and 'to'
  select(DD_DocName, DD_CUI, from, to) %>%           # Select relevant columns
  mutate(from = as.numeric(from),                     # Convert 'from' to numeric
         to = as.numeric(to)) %>%                     # Convert 'to' to numeric
  arrange(DD_DocName, from, to, DD_CUI) %>%           # Arrange by document name, 'from', 'to', and CUI
  group_by(DD_DocName, from, to) %>%                  # Group by document name, 'from', and 'to'
  summarise(DD_CUI = first(DD_CUI)) %>%               # Summarize to keep only the first CUI for each group
  ungroup() %>%                                       # Ungroup the data
  select(DD_DocName, DD_CUI, from, to)                # Select relevant columns

# Rename columns
colnames(SC_train_annotations) <- c("id", "CUI", "from", "to")

# Filter overlapping spans using an external function
SC_train_annotations <- SC_train_annotations %>%
  group_by(id) %>%
  do(filter_overlapping_spans(.)) %>%
  ungroup()
print("Annotations processed and overlapping spans filtered.")

# Save the processed annotations data frame to an RDS file
saveRDS(SC_train_annotations, "Extr_Val_R/Corpora_standardized/SC_train_annotations.rds")
print("Annotations data saved to SC_train_annotations.rds")
