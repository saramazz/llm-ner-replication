# This script translates text documents from English to Italian or Dutch using GPT-4o,
# while keeping the formatting and in-text annotations intact. It processes both
# training and test datasets, saving the results to RDS files for further use.

# Load required libraries
library(openai)  # For interfacing with OpenAI API
library(stringr) # For string manipulation
if (file.exists(".env")) readRenviron(".env")  # OPENAI_API_KEY is read from the environment (see .env.example); never hard-code it
source("Extr_Val_R/R/Tool_functions.R")  # Source additional functions from an external script

# Select language
language <- "Italian"  # Change to "Dutch" as needed
if (language == "Italian") {
  system_message <- "Translate the document to Italian (Italiano). Keep the formatting the same, do not remove the in-text annotations [[span][code]], include the annotations in the translated text."
} else if (language == "Dutch") {
  system_message <- "Translate the document to Dutch (Nederlands). Keep the formatting the same, do not remove the in-text annotations [[span][code]], include the annotations in the translated text."
} else {
  stop("Unsupported language selected.")
}

# Load the processed text with annotations for training and testing
print("Loading training and test data.")
SC_train_loc <- readRDS("Extr_Val_R/Corpora_standardized/SC_train_loc.rds")
SC_test_loc <- readRDS("Extr_Val_R/Corpora_standardized/SC_test_loc.rds")
print("Training and test data loaded successfully.")

#save the train and test data in the same format of the translated data
# Save original training data
original_train_df <- SC_train_loc[, c("id", "text")]
saveRDS(original_train_df, file = "Extr_Val_R/Corpora_standardized/SC_train_loc_english.rds")
print("Original training data saved successfully.")

# Save original test data
original_test_df <- SC_test_loc[, c("id", "text")]
saveRDS(original_test_df, file = "Extr_Val_R/Corpora_standardized/SC_test_loc_english.rds")
print("Original test data saved successfully.")


# Define paths for temporary files
temp_file_train <- paste0("Extr_Val_R/Corpora_translated/SC_train_loc_gpt_", tolower(language), "_temp.rds")
temp_file_test <- paste0("Extr_Val_R/Corpora_translated/SC_test_loc_gpt_", tolower(language), "_temp.rds")

####### TRAINING DATA TRANSLATION
print("Initializing training data translation.")
if (file.exists(temp_file_train)) {
  SC_train_loc_gpt <- readRDS(temp_file_train)
  start_index_train <- sum(!is.na(SC_train_loc_gpt)) + 1
} else {
  SC_train_loc_gpt <- rep(NA, nrow(SC_train_loc))  # Initialize a vector to store translated text
  start_index_train <- 1
}

print("Starting translation for training data.")
start_time <- Sys.time()  # Record the start time

for (i in start_index_train:nrow(SC_train_loc)) {
  if (is.na(SC_train_loc_gpt[i])) {
    print(paste("Translating document", i, "of", nrow(SC_train_loc)))
    t0 <- Sys.time()  # Start timer for this document
    filetxt <- SC_train_loc$text[i]
    nchar_org <- nchar(filetxt)
    
    # Translate using GPT-4o model
    result <- translate_gpt(filetxt, system = system_message, model = "gpt-4o-mini")
    
    # Retry if the result doesn't meet the criteria
    if (!is.na(result) && nchar_org > 200) {
      nchar_result <- nchar(result)
      while (nchar_result < (nchar_org - 100) || !str_detect(result, "\\[ *\\[([^\\]\\[]*)\\]")) {
        print("nchar_result not within range, retrying.")
        print(paste("Original:", nchar_org, "Result:", nchar_result))
        result <- translate_gpt(filetxt, system = system_message, model = "gpt-4o-mini")
        nchar_result <- nchar(result)
      }
    }
    
    SC_train_loc_gpt[i] <- result
    time_taken <- Sys.time() - t0
    print(paste("Translation completed in", round(time_taken, 2), "seconds for document", i))
    saveRDS(SC_train_loc_gpt, file = temp_file_train)
    
    # Print progress percentage
    percentage_done <- (i / nrow(SC_train_loc)) * 100
    print(paste("Progress: ", round(percentage_done, 2), "%"))
  }
}

# Save the translated training data
print("Saving translated training data.")
SC_train_loc_gpt_df <- data.frame(id = SC_train_loc$id, text = SC_train_loc_gpt)
saveRDS(SC_train_loc_gpt_df, file = paste0("Extr_Val_R/Corpora_translated/SC_train_loc_gpt_", tolower(language), ".rds"))
print("Translated training data saved successfully.")

####### TEST DATA TRANSLATION
print("Initializing test data translation.")
if (file.exists(temp_file_test)) {
  SC_test_loc_gpt <- readRDS(temp_file_test)
  start_index_test <- sum(!is.na(SC_test_loc_gpt)) + 1
} else {
  SC_test_loc_gpt <- rep(NA, nrow(SC_test_loc))  # Initialize a vector to store translated text
  start_index_test <- 1
}

print("Starting translation for test data.")
start_time <- Sys.time()  # Record the start time

for (i in start_index_test:nrow(SC_test_loc)) {
  if (is.na(SC_test_loc_gpt[i])) {
    print(paste("Translating document", i, "of", nrow(SC_test_loc)))
    t0 <- Sys.time()  # Start timer for this document
    filetxt <- SC_test_loc$text[i]
    nchar_org <- nchar(filetxt)
    
    # Translate using GPT-4o model
    result <- translate_gpt(filetxt, system = system_message, model = "gpt-4o-mini")
    
    # Retry if the result doesn't meet the criteria
    if (!is.na(result) && nchar_org < 8000 && nchar_org > 200) {
      nchar_result <- nchar(result)
      while (nchar_result < (nchar_org - 100) || !str_detect(result, "\\[ *\\[([^\\]\\[]*)\\]")) {
        print("nchar_result not within range, retrying.")
        print(paste("Original:", nchar_org, "Result:", nchar_result))
        result <- translate_gpt(filetxt, system = system_message, model = "gpt-4o-mini")
        nchar_result <- nchar(result)
      }
    }
    
    SC_test_loc_gpt[i] <- result
    time_taken <- Sys.time() - t0
    print(paste("Translation completed in", round(time_taken, 2), "seconds for document", i))
    saveRDS(SC_test_loc_gpt, file = temp_file_test)
    
    # Print progress percentage
    percentage_done <- (i / nrow(SC_test_loc)) * 100
    print(paste("Progress: ", round(percentage_done, 2), "%"))
  }
}

# Save the translated test data
print("Saving translated test data.")
SC_test_loc_gpt_df <- data.frame(id = SC_test_loc$id, text = SC_test_loc_gpt)
saveRDS(SC_test_loc_gpt_df, file = paste0("Extr_Val_R/Corpora_translated/SC_test_loc_gpt_", tolower(language), ".rds"))
print("Translated test data saved successfully.")
