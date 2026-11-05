# This script processes text and annotation files for the CLEF2014 ShARe-CLEF eHealth 2014 Task 2 dataset.
# 
# 1. It reads and compiles text files from a specified directory into a data frame.
# 2. The compiled text data is saved to an RDS file and exported as a CSV file for use in Python.
# 3. It reads annotation files, processes them to extract relevant information, and performs data cleaning.
# 4. The processed annotation data is saved to an RDS file.
#
# Libraries used:
# - dplyr: For data manipulation
# - readr: For reading/writing CSV files
# - tidyr: For tidying data (used for separating and processing columns)
#
# Additional functions are sourced from an external script ("Extr_Val_R/R/Tool_functions.R").

library(dplyr)       # For data manipulation
library(readr)       # For reading/writing CSV files
source("Extr_Val_R/R/Tool_functions.R")  # Source additional functions from an external script

print("Libraries and functions loaded.")

# Set the location of the text files
location <- "Extr_Val_R/Corpora_raw/CLEF2014/ShAReCLEFeHealth2014Task2_test/ShAReCLEfeHealth2104Task2_test_data_corpus/"
print(paste("Text files location set to:", location))

# List all files in the directory
files <- list.files(location)
print(paste("Files found in directory:", length(files)))

# Filter the list to include only .txt files
files <- files[stringr::str_detect(files, ".txt")]
print(paste("Text files filtered:", length(files)))

#### TEXT PROCESSING
filetxts <- ""   # Initialize an empty string for storing text
SC_test <- NULL  # Initialize an empty data frame to store text data
print("Starting text processing...")

# Loop through each text file
for (file in files) {
  fileName <- paste0(location, file)  # Create full file path
  filetxt <- readChar(fileName, file.info(fileName)$size)  # Read the entire file content
  SC_test <- rbind(SC_test, data.frame(id = file, text = filetxt))  # Append file content to data frame
  print(paste("Processed file:", file))
}

# Before writing, check the type of x
print(paste("SC_test is of type:", class(SC_test)))

# Save the text data frame to an RDS file
saveRDS(SC_test, "Extr_Val_R/Corpora_standardized/SC_test.rds")
print("Text data saved to RDS file.")

##### EXPORT FOR PYTHON
# Write the text data frame to a CSV file for use in Python
write_csv(SC_test, file = "Extr_Val_R/Corpora_python_ready/SC_test_eng.csv", eol = "\n")
print("Text data exported to CSV for Python.")

#### ANNOTATIONS
# Set the location of the annotation files
ann_location <- "Extr_Val_R/Corpora_raw/CLEF2014/ShAReCLEFeHealth2014Task2_test/ShAReCLEFeHealth2014Task2_test_data_default_values/"
print(paste("Annotation files location set to:", ann_location))

# List all annotation files in the directory
ann_files <- list.files(ann_location)
print(paste("Annotation files found in directory:", length(ann_files)))

# Filter the list to include only .txt files
ann_files <- ann_files[stringr::str_detect(ann_files, ".txt")]
print(paste("Annotation files filtered:", length(ann_files)))

SC_test_annotations <- NULL  # Initialize an empty data frame to store annotations
print("Starting annotation processing...")

# Loop through each annotation file
for (file in ann_files) {
  dat <- read.csv(paste0(ann_location, file), sep = "|", header = F, encoding = 'UTF-8')  # Read the annotation file
  colnames(dat) <- c("DD_DocName", "DD_Spans", "DD_CUI", "Norm_NI", "Cue_NI", "Norm_SC", "Cue_SC", 
                      "Norm_UI", "Cue_UI", "Norm_CC", "Cue_CC", "Norm_SV", "Cue_SV", "Norm_CO", 
                      "Cue_CO", "Norm_GC", "Cue_GC", "Norm_BL", "Cue_BL", "Norm_DT", "Norm_TE", "Cue_TE")  # Assign column names
  SC_test_annotations <- rbind(SC_test_annotations, dat)  # Append data to the annotations data frame
  print(paste("Processed annotation file:", file))
}

# Remove the last column from the annotations data frame
SC_test_annotations <- select(SC_test_annotations, -ncol(SC_test_annotations))
print("Removed last column from annotations data.")

# Process the annotations data frame
SC_test_annotations <- SC_test_annotations %>%
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
print("Processed annotations data.")

# Rename columns
colnames(SC_test_annotations) <- c("id", "CUI", "from", "to")
print("Renamed columns in annotations data.")

# Filter overlapping spans using an external function
SC_test_annotations <- SC_test_annotations %>%
  group_by(id) %>%
  do(filter_overlapping_spans(.)) %>%
  ungroup()
print("Filtered overlapping spans.")

# Save the processed annotations data frame to an RDS file
saveRDS(SC_test_annotations, "Extr_Val_R/Corpora_standardized/SC_test_annotations.rds")
print("Annotations data saved to RDS file.")
