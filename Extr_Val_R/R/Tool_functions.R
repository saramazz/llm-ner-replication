# This R script provides a set of functions for NLP tasks related to concept extraction, 
# annotation insertion, overlap filtering, and evaluation of extraction models. The script uses the OpenAI API for GPT-based 
# translation and other text processing tasks. It includes utilities for handling text annotations, comparing extracted concepts 
# with reference annotations, and computing similarity metrics using cosine similarity between embeddings.

# Load the OpenAI key from the environment and the 'openai' library
if (file.exists(".env")) readRenviron(".env")  # OPENAI_API_KEY is read from the environment (see .env.example); never hard-code it

library(openai)

# Function to insert brackets or HTML tags around text spans based on provided locations
insert_brackets <- function(text, locations, html=FALSE, from_mod=0, to_mod=0, no_cui_id="CUI-less") {
  # Ensure locations are sorted in descending order by 'from'
  if(nrow(locations) > 0) {
    # Select and order the necessary columns to avoid redundant entries
    locations <- locations %>%
      select(from, to, CUI) %>%
      distinct()
    locations <- locations[order(-locations$from),]

    # Loop through the locations and insert brackets or HTML tags
    for (i in 1:nrow(locations)) {
      from <- locations$from[i] + from_mod
      to <- locations$to[i] + to_mod
      cui <- locations$CUI[i]
      # Check if HTML tags are to be inserted instead of brackets
      if (html) {
        text <- paste0(substr(text, 1, from - 1),
                       "<span", ifelse(trimws(cui) != no_cui_id, paste0(" code=\"", cui, "\""), ""), ">", 
                       substr(text, from, to), "</span>",
                       substr(text, to + 1, nchar(text)))
      } else {
        # Insert plain brackets with CUIs
        text <- paste0(substr(text, 1, from - 1),
                       "[[", substr(text, from, to), "]",
                       ifelse(trimws(cui) != no_cui_id, paste0("[", cui, "]"), "[C0000000]"),
                       "]", substr(text, to + 1, nchar(text)))
      }
    }
  }
  # Replace newline characters with <br> tags if HTML formatting is selected
  if (html) text <- stringr::str_replace_all(text, "\n", "<br>")
  return(text)
}

# Function to filter overlapping spans, keeping the longest span or the span with a valid CUI
filter_overlapping_spans <- function(spans, no_cui_id="CUI-less") {
  # Sort spans by 'from' and 'to' in ascending order
  spans <- spans %>% arrange(from, to)

  # Initialize an empty dataframe to store non-overlapping spans
  non_overlapping_spans <- spans[0, ]

  # Iterate over each span to filter out overlapping spans
  for (i in 1:nrow(spans)) {
    current_span <- spans[i, ]
    if (i == 1) {
      # Add the first span to the non-overlapping list
      non_overlapping_spans <- current_span
    } else {
      last_span <- tail(non_overlapping_spans, 1)
      # Check for overlap between the current span and the last span in the non-overlapping list
      if (current_span$from <= last_span$to) {
        # Replace the last span if the current one is longer or has a non-NA CUI
        if ((current_span$to - current_span$from) > (last_span$to - last_span$from) ||
            (current_span$CUI != no_cui_id && last_span$CUI == no_cui_id)) {
          non_overlapping_spans[nrow(non_overlapping_spans), ] <- current_span
        }
      } else {
        # If there is no overlap, add the current span to the list
        non_overlapping_spans <- rbind(non_overlapping_spans, current_span)
      }
    }
  }
  return(non_overlapping_spans)
}

# Function to evaluate the performance of concept extraction models against annotated corpora
evalutate_extractions <- function(extraction_results, corpora, frameworks, language, annotations, cui_filter, check_cui = TRUE, check_sim = FALSE, embeddings = NULL) {
  # Load the 'data.table' library
  require(data.table)

  # Ensure that embeddings are provided if similarity checking is enabled
  if (is.null(embeddings) & check_sim) stop("Provide embeddings")

  id_list <- unique(annotations$id)
  resall <- NULL
  extraction_results <- as.data.table(extraction_results)
  
  # Loop through each corpus, framework, and library to evaluate extractions
  for (corpus_sel in corpora) {
    writeLines(corpus_sel)
    for (framework_sel in frameworks) {
      writeLines(paste("\t", framework_sel))
      extraction_results_sel <- extraction_results[framework == framework_sel & corpus == corpus_sel, ]
      libraries <- unique(extraction_results_sel$library)
      for (library_sel in libraries) {
        writeLines(paste("\t\t", library_sel))
        t0 <- Sys.time()
        # Filter extraction results by library and CUI filter
        if (!is.null(cui_filter)) {
          selection <- extraction_results_sel[library == library_sel & CUI %in% cui_filter, ]
        } else {
          selection <- extraction_results_sel[library == library_sel, ]
        }
        # Loop through each document ID to compute evaluation metrics
        for (sel_id in id_list) {
          CUIs_obs <<- annotations %>%
            filter(id == sel_id) %>%
            select(CUI, from, to) %>%
            mutate(rn = row_number())
          CUIs_pred <<- selection[id == sel_id, c("CUI", "from", "to")] %>%
            mutate(rn = row_number())
          res <- ComparePredsAndObs2(CUIs_obs, CUIs_pred, checkcui = check_cui, checksim = check_sim, embeddings = embeddings) %>%
            mutate(id = sel_id,
                   n_obs = nrow(CUIs_obs),
                   n_pred = nrow(CUIs_pred),
                   language = language,
                   framework = framework_sel,
                   library = library_sel,
                   corpus = corpus_sel, .before = 1)
          resall <- rbind(resall, res)
        }
        writeLines(paste("\t\t time:", round(as.numeric(Sys.time() - t0), 2)))
      }
    }
  }

  return(resall)
}

# Function to compare predicted and observed spans to compute TP, FP, FN, precision, recall, and F1-score
ComparePredsAndObs <- function(obs, preds, checkcui = TRUE) {
  TP <- 0
  FP <- 0
  FN <- 0

  # Convert 'from' and 'to' columns to numeric
  obs <- obs %>% mutate(to = as.numeric(to), from = as.numeric(from))
  preds <- preds %>% mutate(to = as.numeric(to), from = as.numeric(from))

  # Handle case with no predictions
  if (nrow(preds) == 0) {
    TP <- 0
    FP <- 0
    FN <- nrow(obs)
  } else {
    # Evaluate each predicted span for overlaps and CUI matches
    for (i in 1:nrow(preds)) {
      overlap_obs <- obs[(obs$from <= preds$to[i] & obs$to >= preds$from[i]),]
      overlap <- nrow(overlap_obs) > 0

      # Check if the predicted CUI matches any overlapping observed CUIs
      if (checkcui & overlap) {
        same_code <- if (all(overlap_obs$CUI == "CUI-less")) TRUE else any(overlap_obs$CUI == preds$CUI[i], na.rm = TRUE)
      } else {
        same_code <- TRUE
      }

      # Update TP and FP counters
      if (overlap & same_code) {
        TP <- TP + 1
      } else {
        FP <- FP + 1
      }
    }

    # Evaluate each observed span for overlaps and CUI matches
    for (i in 1:nrow(obs)) {
      overlap_preds <- preds[(preds$from <= obs$to[i] & preds$to >= obs$from[i]),]
      overlap <- nrow(overlap_preds) > 0

      if (checkcui & overlap) {
        same_code <- if (obs$CUI[i] == "CUI-less") TRUE else any(overlap_preds$CUI == obs$CUI[i], na.rm = TRUE)
      } else {
        same_code <- TRUE
      }

      # Update FN counter
      if (!(overlap & same_code)) {
        FN <- FN + 1
      }
    }
  }
  # Calculate precision, recall, and F1-score
  prec <- ifelse(TP + FP > 0, TP / (TP + FP), 0)
  rec <- ifelse(TP + FN > 0, TP / (TP + FN), 0)
  f1 <- ifelse(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)

  return(data.frame(TP = TP, FP = FP, FN = FN, precision = prec, recall = rec, F1 = f1))
}
ComparePredsAndObs2<-function(CUIs_obs,CUIs_pred,checkcui = F, checksim=F,embeddings=NULL){
  colnames(CUIs_pred)<-paste0(colnames(CUIs_pred),"_pred")
  colnames(CUIs_obs)<-paste0(colnames(CUIs_obs),"_obs")

  CUIs_TP_location <- tidyr::crossing(CUIs_obs, CUIs_pred) %>%
    filter(from_obs<=to_pred) %>%
    filter(to_obs>=from_pred)

  if(nrow(CUIs_TP_location)>0){
    if(checksim){
      # Goede manier, met concept combos, maar heeft meer logic nodig:
      # CUIs_TP <- tidyr::crossing(CUIs_obs, CUIs_pred) %>%
      #   filter(from_obs<=to_pred,to_obs>=from_pred) %>%
      #   group_by(from_obs,to_obs) %>%
      #   mutate(sim=getSim(CUI_obs,CUI_pred,embeddings=embeddings,n=n())) %>%
      #   filter(sim>0.9)
      # Greedy manier:

      CUIs_TP <- CUIs_TP_location %>%
        rowwise() %>%
        mutate(sim=getSim(CUI_obs,CUI_pred,embeddings=embeddings)) %>%
        filter(sim>0.9)
    } else if(checkcui){
      CUIs_TP <- CUIs_TP_location %>%
        filter(CUI_obs==CUI_pred)
    } else {
      CUIs_TP <- CUIs_TP_location
    }
  } else {
    CUIs_TP <- CUIs_TP_location
  }
  TP<-length(unique(CUIs_TP$rn_pred))

  CUIs_FP <- CUIs_pred %>%
    filter(!rn_pred %in% CUIs_TP$rn_pred)
  FP<-nrow(CUIs_FP)

  CUIs_FN <- CUIs_obs %>%
    filter(!rn_obs %in% CUIs_TP_location$rn_obs)
  FN<-nrow(CUIs_FN)

  prec <- ifelse(TP + FP > 0, TP / (TP + FP), 0)
  rec <- ifelse(TP + FN > 0, TP / (TP + FN), 0)
  f1 <- ifelse(prec + rec > 0, 2 * ((prec * rec) / (prec + rec)), 0)
  result<-data.frame(TP,FP,FN,prec,rec,f1)
  return(result)
}

getSim<-function(CUI_obs,CUI_preds,model=NULL, embeddings=NULL){
  if(is.null(model)) model<-"text-embedding-ada-002"

  CUI_obs<-unique(CUI_obs)
  CUI_preds<-unique(CUI_preds)

  #print(paste("obs",CUI_obs))
  #print(paste("pred",CUI_preds))

  sims<-NULL
  for (cui_ob in CUI_obs) {
    #print(cui_ob)
    for (cui_pred in CUI_preds){
      #print(cui_pred)
      if(cui_pred==cui_ob){
        sims<-c(sims,1)
      } else {
        # print(cui_pred)
        if(cui_pred %in% rownames(embeddings) & cui_ob %in% rownames(embeddings)){
          CUI_pred_vector <- embeddings[cui_pred,]
          CUI_obs_vector <- embeddings[cui_ob,]

          sim<-lsa::cosine(CUI_pred_vector,CUI_obs_vector)
          sims<-c(sims,sim)
        } else {
          sims<-c(sims,0)
        }
      }
      #print(sims)
    }
  }
  #print(sims)
  return(sims)
}


translate_gpt<-function(filetxt,system=NULL,temp=NULL,model=NULL){
  if(Sys.getenv("OPENAI_API_KEY")==""){
    stop("OPENAI_API_KEY is not set: export it or put it in .env (see .env.example)")
  }
  if(is.null(system)){
    system<-"Translate the document to Dutch (Nederlands). Keep the formatting the same."
  }
  if(is.null(temp)){
    temp<-0.6
  }
  if(is.null(model)){
    model<-"gpt-3.5-turbo-0125"
  }

  chat_data <- data.frame()
  system_data <- data.frame(role = "system", content = system, stringsAsFactors = FALSE)
  chat_data<-rbind(chat_data, system_data)

  prompt<-filetxt
  new_data <- data.frame(role = "user", content = prompt, stringsAsFactors = FALSE)
  chat_data<-rbind(chat_data, new_data)

  chat_data_prompt<-purrr::transpose(chat_data)

  result<-NA
  tryCatch(
    expr = {
      gpt_res<-create_chat_completion(
        model = model,
        temperature = temp,
        messages = chat_data_prompt
      )
      result<-gpt_res$choices$message.content
      print(gpt_res$choices$finish_reason)
      while(gpt_res$choices$finish_reason=="length"){
        print("Finish reason = length")
        new_data <- data.frame(role = "assistant", content = result, stringsAsFactors = FALSE)
        chat_data<-rbind(chat_data, new_data)
        chat_data_prompt<-purrr::transpose(chat_data)
        gpt_res<-create_chat_completion(
          model = model,
          temperature = temp,
          messages = chat_data_prompt
        )
        result<-paste0(result,gpt_res$choices$message.content)
      }
    },
    error = function(e){
      print(e)
      result<-NA
      print(result)
    },
    warning = function(w){
      print(w)
    }
  )
  return(result)
}

replace_span <- function(text, from, to, replace) {
  # Check if the span is valid
  if(from < 1 || to > nchar(text) || from > to) {
    stop("Invalid span provided.")
  }
  # Extract the parts of the string
  before <- substr(text, 1, from - 1)
  after <- substr(text, to + 1, nchar(text))
  # Combine the parts with the replacement
  result <- paste0(before, replace, after)
  return(result)
}

analyze_rds <- function(file_path) {
  # Read the RDS file
  data <- readRDS(file_path)
  
  # Get dimensions
  dimensions <- dim(data)
  
  # Object type
  data_type <- class(data)
  
  # Number of rows
  num_rows <- ifelse(is.null(dim(data)), length(data), nrow(data))
  
  # Rows with and without NA
  #rows_with_na <- sum(apply(data, 1, function(row) any(is.na(row))))
  #rows_without_na <- num_rows - rows_with_na
  
  # Print the information
  cat("File:", file_path, "\n")
  cat("Dimensions:", ifelse(is.null(dim(data)), "Not applicable (not a dataframe or matrix)", paste(dimensions, collapse = " x ")), "\n")
  cat("Type:", data_type, "\n")
  cat("Number of rows:", num_rows, "\n")
  #find NA rows in a vector
  cat("Number of rows with NA in the translation file:", sum(is.na(data)), "\n")
  #cat("Rows with NA:", rows_with_na, "\n")
  #cat("Rows without NA:", rows_without_na, "\n")
  cat("-------------------------------\n")
}

# extract_annotatinos_from_loctext<-function(annotated_text,start_at_zero=FALSE){
#   concept_pattern_1 <- "\\[ *\\[([^\\]\\[]*)\\] *\\[ *([^\\]\\[]*) *\\] *\\]"
#   concept_pattern_2 <- "\\[ *\\[([^\\]\\[]*)\\] *\\]"
#   annotations<-NULL
#   cont<-TRUE
#   result_text<-annotated_text
#   while (cont) {
#     ann_location1<-stringr::str_locate(result_text,concept_pattern_1)
#     ExistAnn1<-!all(is.na(ann_location1))
#     ann_location2<-stringr::str_locate(result_text,concept_pattern_2)
#     ExistAnn2<-!all(is.na(ann_location2))
#
#     if(ExistAnn1 & ExistAnn2){ # if both are present
#       if(ann_location1[1,"start"]<ann_location2[1,"start"]){ # if 1 is earlier than 2, do 1
#         ann_location<-ann_location1
#         concept_pattern_temp<-concept_pattern_1
#         no_cui<-FALSE
#       } else { # if 2 is earlier than 1, do 2
#         ann_location<-ann_location2
#         concept_pattern_temp<-concept_pattern_2
#         no_cui<-TRUE
#       }
#     } else if(ExistAnn1){ # if 1 is present
#       ann_location<-ann_location1
#       concept_pattern_temp<-concept_pattern_1
#       no_cui<-FALSE
#     } else if(ExistAnn2){ # if 2 is present
#       ann_location<-ann_location2
#       concept_pattern_temp<-concept_pattern_2
#       no_cui<-TRUE
#     } else { # if none are present
#       cont<-FALSE
#     }
#
#     if(cont){
#       ann_concept<-stringr::str_match(result_text,concept_pattern_temp)
#       ann_start<-ann_location[1,"start"]
#       ann_end<-ann_location[1,"end"]
#       ann_full<-ann_concept[1,1]
#       ann_text<-ann_concept[1,2]
#       if(no_cui){
#         ann_cui<-NA
#       } else {
#         ann_cui<-ann_concept[1,3]
#       }
#
#       result_text<-replace_span(result_text,ann_start,ann_end,ann_text)
#       #print(result_text)
#       if(start_at_zero) ann_from<-ann_start-1
#       else ann_from<-ann_start
#       removedchars<-(nchar(ann_full)-nchar(ann_text))
#       if(start_at_zero) ann_to<-ann_end-removedchars
#       else ann_to<-ann_end-removedchars+1
#
#       annots<-data.frame(CUI=ann_cui,
#                          from=ann_from,
#                          to=ann_to,
#                          value=ann_text)
#       annotations<-rbind(annotations,annots)
#     }
#   }
#
#   return(list(text=result_text,annotations=annotations))
# }

extract_annotatinos_from_loctext <- function(annotated_text, start_at_zero = FALSE) {
  concept_patterns <- list(
    "\\[ *\\[([^\\]\\[]*)\\] *\\[ *([^\\]\\[]*) *\\] *\\]", # perfect pattern
    "\\[ *\\[([^\\]\\[]*)\\] *\\]"#, # no code
    #"\\[ *\\[([^ ]*?)\\] *\\[ *(C[0-9]*) *\\] *\\]"#, # no spaces in span, allows for [] in side the span
    # "\\[([^\\]\\[]*)\\] *\\[ *(C[0-9]*) *\\] *\\]", # missing  [][]]
    # "\\[ *\\[([^\\]\\[]*)\\[ *(C[0-9]*) *\\] *\\]", # missing [[ []]
    # "\\[ *\\[([^\\]\\[]*)\\] *(C[0-9]*) *\\] *\\]", # missing [[] ]]
    # "\\[ *\\[([^\\]\\[]*)\\] *\\[ *(C[0-9]*) *\\]" # missing [[][]
    # Add more patterns here as needed
  )

  annotations <- NULL
  cont <- TRUE
  result_text <- annotated_text

  while (cont) {
    ann_location <- list()
    ExistAnn <- logical(length(concept_patterns))

    for (i in seq_along(concept_patterns)) {
      loc <- stringr::str_locate(result_text, concept_patterns[[i]])
      ann_location[[i]] <- loc
      ExistAnn[i] <- !all(is.na(loc))
    }

    if (!any(ExistAnn)) {
      cont <- FALSE
      next
    }

    # Find the earliest match
    earliest <- which.min(sapply(ann_location, function(x) if (is.na(x[1])) Inf else x[1]))
    ann_location <- ann_location[[earliest]]
    concept_pattern_temp <- concept_patterns[[earliest]]

    ann_concept <- stringr::str_match(result_text, concept_pattern_temp)
    ann_start <- ann_location[1]
    ann_end <- ann_location[2]
    ann_full <- ann_concept[1, 1]
    ann_text <- ann_concept[1, 2]
    ann_cui <- if (ncol(ann_concept) > 2) ann_concept[1, 3] else NA

    # Replace the span in the text
    result_text <- replace_span(result_text, ann_start, ann_end, ann_text)
    #print(result_text)

    # Calculate the from and to
    ann_from <- if (start_at_zero) ann_start - 1 else ann_start
    removedchars <- (nchar(ann_full) - nchar(ann_text))
    ann_to <- if (start_at_zero) ann_end - removedchars else ann_end - removedchars #+ 1

    # Combine
    annots <- data.frame(CUI = ann_cui, from = ann_from, to = ann_to, value = ann_text)
    annotations <- rbind(annotations, annots)
  }

  return(list(text = result_text, annotations = annotations))
}

# clean_text<-function(text){
#   text<-str_replace_all(text,"\\] *\\[ *C[0-9]*\\] *\\]","")
#   text<-str_replace_all(text,"\\[ *C[0-9]*\\] *\\]","")
#   text<-str_replace_all(text,"C[0-9]*\\] *\\]","")
#   text<-str_replace_all(text,"\\] *C[0-9]* *\\]","")
#   text<-str_replace_all(text,"C[0-9]* *\\]","")
#   text<-str_replace_all(text,"\\] *C[0-9]*","")
#   text<-str_replace_all(text,"\\[\\[","")
#   text<-str_replace_all(text,"\\[\\[","")
# }

summ<-function(resall){
  resall %>%
    group_by(corpus,framework,library, language,eval) %>%
    summarise(sum=n(),
              TP=sum(TP),
              FP=sum(FP),
              FN=sum(FN),
              prec_m=mean(prec, na.rm=TRUE),
              rec_m=mean(rec, na.rm=TRUE),
              #f1_m_m=mean(f1, na.rm=TRUE),
              f1_m=2*((prec_m*rec_m)/(prec_m+rec_m)),
              prec=TP/(TP+FP),
              rec=TP/(TP+FN),
              f1=2*((prec*rec)/(prec+rec)))
}


