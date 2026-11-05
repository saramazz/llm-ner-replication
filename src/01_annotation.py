"""
This script automates the annotation of medical anamneses using OpenAI's GPT models.
It loads unannotated data, checks for previously annotated entries, and processes new ones in parallel.
The script constructs prompts based on annotation guidelines and interacts with the OpenAI API to generate annotations.
It handles API rate limits and errors, supports resuming from previous runs, and saves results incrementally.
Execution details and parameters are logged for reproducibility.
Configuration is managed via external paths and settings.
The script supports both English and non-English datasets.
User interaction is required to confirm annotation execution.
"""

import pandas as pd
import os
from openai import OpenAI
import openai
import datetime
from config import current_path, global_path, saving_path, corpora_path
from openai import OpenAI
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAIError, RateLimitError

# Set your OpenAI API key

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # set in the environment (see .env.example)

MAX_WORKERS = 5
MAX_RETRIES = 3  # Maximum retry attempts in case of API errors


# Dataset and language configuration
dataset_name = "SC"
language = "italian"
language_long = "italian"
# input the solit: train or test
# split = input("Enter the split train or test:")
split = "train"

# Save results to a new DataFrame
saving_folder = os.path.join(saving_path, "sample_gpt")
print("Saving folder: ", saving_folder)
os.makedirs(saving_folder, exist_ok=True)

# if language is en
if language == "en":
    unannotated_file = os.path.join(
        corpora_path, f"{dataset_name}_{split}_{language}.csv"
    )
    # annotated_file = os.path.join(
    # corpora_path, f"{dataset_name}_{split}_loc_{language_long}.csv"
    # )  # file with annotations
else:
    unannotated_file = os.path.join(
        corpora_path, f"{dataset_name}_{split}_{language}_gpt.csv"
    )  # add gpt if the language is not english
    # annotated_file = os.path.join(
    # corpora_path, f"{dataset_name}_{split}_loc_{language_long}.csv"
    # )


# Save the annotations to a new CSV file
annotated_file_llm = os.path.join(
    saving_folder, f"{dataset_name}_{split}_loc_{language_long}_zero_shot.csv"
)

print("Annotated file LLM: ", annotated_file_llm)


# Load unannotated anamneses
try:
    unannotated_df = pd.read_csv(unannotated_file)
    # Restrict the number of anamneses to 2 (change this as needed)
    # unannotated_df = unannotated_df.head(2)
    print("\nLoaded unannotated anamneses file successfully.")
    # print the columns of the unannotated file


except FileNotFoundError:
    print(f"Error: The file {unannotated_file} was not found.")
    unannotated_df = pd.DataFrame()

print("Columns unannotated: ", unannotated_df.columns)
# print head of the unannotated file
print("Head of the unannotated file: ", unannotated_df.head())
input("Press Enter to continue...")
"""
# Load annotated anamneses for reference (if needed)
try:
    annotated_df = pd.read_csv(annotated_file)
    print("\nLoaded annotated anamneses file successfully.")
except FileNotFoundError:
    print(f"Error: The file {annotated_file} was not found.")
    annotated_df = pd.DataFrame()


#print the cols of the annotated file
print("Columns annotated: ", annotated_df.columns)
"""

# Load previously annotated LLM file
already_annotated_ids = set()
if os.path.exists(annotated_file_llm):
    try:
        annotated_llm_df = pd.read_csv(annotated_file_llm)
        already_annotated_ids = set(annotated_llm_df["id"].tolist())
        print("Loaded previously annotated LLM file.")
        # Filter out already processed IDs
        df = unannotated_df[~unannotated_df["id"].isin(already_annotated_ids)]
    except FileNotFoundError:
        print(f"Error: The file {annotated_file_llm} was not found.")
else:
    df = unannotated_df  # [~unannotated_df["id"].isin(already_annotated_ids)]


# print("Filtered dataframe to annotate: ", df)


# Load the dataset
# df = unannotated_df  # pd.read_csv(unannotated_file)
# print the path of the unannotated file
# print("Unannotated file: ", unannotated_file)

# print detail of the unannotated file as lenght
print("Unannotated file details: ")
print(df.info())
# print the column of the unannotated file
print("Columns: ", df.columns)

# Reduce to the number of anamneses to annotate
n_anamneses = len(df)

"""
# Extract and concatenate the text of the first 3 anamneses
long_text = " ".join(df["text"].head(3))

# Display the result
print("Long Anamnesis Text:\n")
print(long_text)

#put the long text in the first anamnesis
df["text"].iloc[0] = long_text
"""

df = df.iloc[:n_anamneses]
# print the first anamneses
# print("First anamneses: ", df)


base_prompt = None
# Declare global variables
global_max_tokens = None
global_temperature = None


def annotate_anamnesis(index, row, max_tokens=2000, temperature=0.5):
    global base_prompt, global_max_tokens, global_temperature

    # Update global variables if they are not already set
    if global_max_tokens is None:
        global_max_tokens = max_tokens
    if global_temperature is None:
        global_temperature = temperature

    # Initialize the base prompt if it is not set
    if base_prompt is None:
        # Read the annotation guidelines from the file
        with open(os.path.join(saving_path, "annotation_guidelines_SC.txt"), "r") as f:
            base_prompt = (
                f.read() + "\n'''"  # adding the triple quote to the annotation
            )

    # print("Base prompt: ", base_prompt)

    # Full text of the anamnesis
    full_text = row["text"]
    index = row["id"]  # save the index of the anamnesis
    remaining_text = full_text
    annotated_text = ""
    retries = 0

    # Flag to use the base_prompt only for the first segment
    first_segment = True

    while remaining_text:
        print(f"Annotating anamnesis {index}...")
        print(f"ID: {row['id']}")
        progress = 100 * (len(full_text) - len(remaining_text)) / len(full_text)
        print(f"Progress: {progress:.2f}%")

        try:
            # Create the current prompt
            if first_segment:
                # Properly concatenate the base prompt to the remaining text
                current_prompt = base_prompt + remaining_text + "'''"
                first_segment = False
            else:
                # Truncate only the guideline part for ongoing segments, si the context stays in sync

                print("Using the base prompt to continue ...")

            # Log current prompt
            # print("Current prompt:", current_prompt)
            # input("Press Enter to continue with this prompt...")

            # Call the ChatCompletion endpoint
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": current_prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Extract the annotation and finish reason
            annotation = response.choices[0].message.content.strip()
            finish_reason = response.choices[0].finish_reason
            print(f"Finish reason: {finish_reason}")

            # Append the annotation to the final result
            annotated_text += annotation

            # If the finish reason is 'length', truncate the remaining text
            while finish_reason == "length":
                # Calculate how much of the text was processed and update remaining_text
                annotated_portion = len(annotation)
                remaining_text = remaining_text[annotated_portion:]
                # print the answer
                # print("Answer: ", annotation)
                # print("Continuing annotation due to 'length' finish reason...")

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": current_prompt}]
                    + [{"role": "assistant", "content": annotated_text}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                # Extract the annotation and finish reason
                annotation = response.choices[0].message.content.strip()
                finish_reason = response.choices[0].finish_reason

                # print("New part of the answer: ", annotation)
                # print(f"Finish reason: {finish_reason}")

                # Append the annotation to the final result
                annotated_text += annotation
                # remaining_text = None

            else:
                # If finish_reason is not 'length', we are done
                # print("Final answer reached due to finish reason:", finish_reason)
                remaining_text = None

        except RateLimitError:  # openai.error.RateLimitError:
            print("Rate limit error. Waiting 5 seconds...")
            if retries < MAX_RETRIES:
                print(
                    f"Rate limit exceeded. Waiting 10 seconds... (attempt {retries+1}/{MAX_RETRIES})"
                )
                time.sleep(10)  # Wait 10 seconds before retrying
                retries += 1
            else:
                print("Rate limit error. Skipping this anamnesis.")
                return index, None
        except OpenAIError as e:  # openai.error.OpenAIError as e:
            print("OpenAI error:", e)
            break
        except Exception as e:
            print(f"General error on anamnesis {index}: {e}")
            return index, None
            # break

    return index, annotated_text


# Annotate all anamneses
annotations = []  # List to store the results
start_time = datetime.datetime.now()

# Ask if do the annotation with llm
print("Do you want to annotate with LLM?")
answer = input("y or n: ")

if answer == "y":
    print("You have chosen to annotate with LLM")
    # print the number of anamneses to annotate
    print("Number of anamneses to annotate: ", n_anamneses)
    # Parallel execution
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(annotate_anamnesis, i, row): i for i, row in df.iterrows()
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                print(
                    f"\nStarting annotation process at {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                result = future.result()
                if result[1] is not None:
                    annotations.append({"id": result[0], "annotation": result[1]})
            except Exception as e:
                print(f"Error in the annotation process for anamnesis {index}: {e}")

            # results = executor.map(
            # lambda item: annotate_anamnesis(item[0], item[1]),
            # zip(df.index, (row for _, row in df.iterrows())),
            # )
            # results = executor.map(annotate_anamnesis, df.index, df.iterrows())
    # Save annotated file
    annotated_df_llm = pd.DataFrame(annotations)
    if os.path.exists(annotated_file_llm):
        annotated_df_llm.to_csv(
            annotated_file_llm, mode="a", header=False, index=False
        )  # append if already exists
    else:
        annotated_df_llm.to_csv(annotated_file_llm, index=False)  # create if not exists

    print("Annotated file saved in: ", annotated_file_llm)
    """
    # Converti i risultati in una lista
    results = list(results)

    # Print the results and store them in the annotations list
    for index, annotation in results:
        print(f"Index: {index}")
        # print(f"Annotation: {annotation}")
        # print("\n")

        annotations.append({"id": index, "annotation": annotation})

    # Save the annotations to a new DataFrame
    annotated_df_llm = pd.DataFrame(annotations)
    """
    # load annotations
    # annotated_df = pd.read_csv(annotated_file)

    # print the annotated_df
    # print("Annotated file: ", annotated_df)

    # print the path of the annotated file
    print("Annotated file saved in: ", annotated_file_llm)
else:
    print("Loading the annotated file")
    annotated_df_llm = pd.read_csv(annotated_file_llm)

    # print the annotated file llm and the annotated file from clinicians
    # print("Annotated file llm: ", annotated_df_llm["annotation"])
    # print("Annotated file clinicians: ", annotated_df.iloc[:n_anamneses]["text"])


# Log execution details
log_file = os.path.join(
    saving_folder,
    # f"{dataset_name}_{split}_{language}_execution_log_{start_time.strftime('%Y%m%d_%H%M%S')}.txt",
    f"{dataset_name}_{split}_{language}_annotation_log.txt",
)
with open(log_file, "a") as log:
    log.write(f"Execution Date and Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    elapsed_time = datetime.datetime.now() - start_time
    log.write(f"Total Elapsed Time: {elapsed_time}\n")
    log.write(f"Model: gpt-4o\n")
    log.write(f"Max Tokens: {global_max_tokens}\n")
    log.write(f"Temperature: {global_temperature}\n")
    log.write(f"Number of Anamneses Processed: {n_anamneses}\n")

    # Log the base prompt
    if base_prompt:
        log.write(f"\nBase Prompt:\n{base_prompt}\n")

print(f"\nLog file saved to {log_file}")
