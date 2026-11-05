# This script processes translated text files from Google and GPT-4o, extracting annotations
# and saving the results in RDS files. It also exports the extracted text to CSV files for use in Python.

# Load required libraries
library(dplyr)   # For data manipulation
library(readr)   # For reading and writing CSV files
source("Extr_Val_R/R/Tool_functions.R")  # Source additional functions from an external script

#### FUNCTION TO PROCESS TRANSLATED DATA BASED ON LANGUAGE AND TRANSLATION SOURCE ####
# This function processes the translations for a specific language and source (Google or GPT-4o).
# It extracts the annotations and text, and then saves the results in RDS files and exports the text to CSV.
process_translation_data <- function(language_code, translation_source, file_path) {
  # Conditionally process Google translations only for Dutch
  if (language_code == "dutch" && translation_source == "google") {
    #### GOOGLE TRANSLATION PROCESSING ####
    print("Loading Google translation data for Dutch.")
    
    # Load the standardized Dutch text
    SC_test_loc <- readRDS(paste0("Extr_Val_R/Corpora_standardized/SC_test_loc.rds"))
    
    # Read the Google translation text from the file path
    SC_test_loc_translation <- readr::read_file(file_path)
    
    # Split the translation text by the delimiter "<<<<TOM>>>>\n" (specific to the data format)
    SC_test_loc_translation <- unlist(stringr::str_split(SC_test_loc_translation, "<<<<TOM>>>>\n"))
    
    # Create a data frame with the original IDs and corresponding translated text
    SC_test_loc_translation <- data.frame(id = SC_test_loc$id, text = SC_test_loc_translation[-1])
    print("Google translation data loaded and processed for Dutch.")
    
    # Initialize variables to store the extracted annotations and text
    SC_test_loc_translation_extr <- NULL
    SC_test_loc_translation_annot <- NULL
    
    # Loop through each row to extract annotations and text
    for (row in 1:nrow(SC_test_loc_translation)) {
      id <- SC_test_loc_translation$id[row]
      
      # Extract annotations and text using the custom function extract_annotatinos_from_loctext
      result <- extract_annotatinos_from_loctext(SC_test_loc_translation$text[row], start_at_zero = TRUE)
      
      # Create a data frame for the extracted text (without annotations)
      df_extr <- data.frame(id = id, text = result$text)
      
      # If annotations are found, create a data frame for the annotations and add the ID
      if (!is.null(result$annotations)) {
        df_annot <- result$annotations %>% mutate(id = id, .before = 1)
      }
      
      # Append the extracted text and annotations to the result variables
      SC_test_loc_translation_extr <- rbind(SC_test_loc_translation_extr, df_extr)
      SC_test_loc_translation_annot <- rbind(SC_test_loc_translation_annot, df_annot)
    }
    print("Annotations extracted from Google-translated text for Dutch.")
    
    # Save the extracted annotations and text into RDS files
    print("Saving extracted annotations and text for Google translation in Dutch.")
    saveRDS(SC_test_loc_translation_annot, paste0("Extr_Val_R/Corpora_translated/SC_test_loc_", language_code, "_", translation_source, "_annot.rds"))
    saveRDS(SC_test_loc_translation_extr, paste0("Extr_Val_R/Corpora_translated/SC_test_loc_", language_code, "_", translation_source, "_extr.rds"))
    print("Google translation RDS files saved for Dutch.")
    
    # Export the extracted text to CSV for Python processing
    print("Exporting extracted text to CSV for Google translation in Dutch.")
    write_csv(SC_test_loc_translation_extr, file = paste0("Extr_Val_R/Corpora_python_ready/SC_test_", language_code, "_", translation_source, ".csv"))
    print("Google translation CSV file exported for Dutch.")
  } else {

    #### GPT-4o TRANSLATION PROCESSING ####
    print(paste("Loading GPT-4o translation data for", language_code, "."))
    
    # Load the standardized text for the specific language
    SC_test_loc <- readRDS(paste0("Extr_Val_R/Corpora_translated/SC_test_loc_gpt_", language_code, ".rds"))
    
    # Load the GPT-4o translation text from the file path (this is an RDS file)
    SC_test_loc_translation <- readRDS(file_path)
    
    print(paste("GPT-4o translation data loaded for", language_code, "."))
    
    # Initialize variables to store the extracted annotations and text
    SC_test_loc_translation_extr <- NULL
    SC_test_loc_translation_annot <- NULL
    
    # Loop through each row to extract annotations and text
    for (row in 1:nrow(SC_test_loc_translation)) {
      id <- SC_test_loc_translation$id[row]
      
      # Extract annotations and text using the custom function extract_annotatinos_from_loctext
      result <- extract_annotatinos_from_loctext(SC_test_loc_translation$text[row], start_at_zero = TRUE)
      
      # Create a data frame for the extracted text (without annotations)
      df_extr <- data.frame(id = id, text = result$text)
      
      # If annotations are found, create a data frame for the annotations and add the ID
      if (!is.null(result$annotations)) {
        df_annot <- result$annotations %>% mutate(id = id, .before = 1)
      }
      
      # Append the extracted text and annotations to the result variables
      SC_test_loc_translation_extr <- rbind(SC_test_loc_translation_extr, df_extr)
      SC_test_loc_translation_annot <- rbind(SC_test_loc_translation_annot, df_annot)
    }
    print(paste("Annotations extracted from GPT-4o-translated text for", language_code, "."))
    
    # Save the extracted annotations and text into RDS files
    print(paste("Saving extracted annotations and text for GPT-4o translation in", language_code, "."))
    saveRDS(SC_test_loc_translation_annot, paste0("Extr_Val_R/Corpora_translated/SC_test_loc_", language_code, "_", translation_source, "_annot.rds"))
    saveRDS(SC_test_loc_translation_extr, paste0("Extr_Val_R/Corpora_translated/SC_test_loc_", language_code, "_", translation_source, "_extr.rds"))
    print(paste("GPT-4o translation RDS files saved for", language_code, "."))
    
    # Export the extracted text to CSV for Python processing
    print(paste("Exporting extracted text to CSV for GPT-4o translation in", language_code, "."))
    write_csv(SC_test_loc_translation_extr, file = paste0("Extr_Val_R/Corpora_python_ready/SC_test_", language_code, "_", translation_source, ".csv"))
    print(paste("GPT-4o translation CSV file exported for", language_code, "."))
  }
}

#### PROCESS TRANSLATIONS ####

# Processing the Google translation for Dutch text (only for Dutch)
process_translation_data("dutch", "google", "Extr_Val_R/Corpora_translated/SC_test_loc_google_dutch.txt")

# Processing the GPT-4o translation for Italian text (only GPT for Italian)
process_translation_data("italian", "gpt", "Extr_Val_R/Corpora_translated/SC_test_loc_gpt_italian.rds")

# Processing the GPT-4o translation for Dutch text
process_translation_data("dutch", "gpt", "Extr_Val_R/Corpora_translated/SC_test_loc_gpt_dutch.rds")
