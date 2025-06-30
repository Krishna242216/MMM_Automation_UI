#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)

personal_lib <- "C:/Users/MM3815/Documents/R/library"
if (!dir.exists(personal_lib)) dir.create(personal_lib, recursive = TRUE)
.libPaths(personal_lib)

required_packages <- c("Robyn", "dplyr", "jsonlite", "ggplot2", "doSNOW", "reticulate")

missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop("Error: The following required packages are missing: ", paste(missing_packages, collapse = ", "), 
       "\nPlease install them manually in R with:\n",
       "install.packages(c('", paste(missing_packages, collapse = "', '"), "'), dependencies = TRUE, lib = '", personal_lib, "')")
}

for (pkg in required_packages) {
  suppressPackageStartupMessages(library(pkg, character.only = TRUE, quietly = TRUE))
}

message("Robyn version: ", packageVersion("Robyn"))

if (length(args) == 0) {
  config_path <- "C:\\Users\\MM3815\\Downloads\\data_cleaning_app\\robyn_output\\config.json"
  
  if (file.exists(config_path)) {
    message("No command-line arguments provided. Reading from ", config_path)
    config <- jsonlite::fromJSON(config_path)
    args <- c(
      config$data_path,
      config$dep_var,
      config$media_vars,
      config$control_vars,
      config$date_var,
      config$adstock_type,
      config$iterations,
      config$trials,
      config$User,
      config$Project_Name,
      config$ID
    )
  } else {
    stop("Error: No command-line arguments provided, and ", config_path, " does not exist.")
  }
}

if (length(args) < 11) {
  stop("Error: Insufficient arguments. Expected 8 arguments: ",
       "data_path, dep_var, media_vars, control_vars, date_var, adstock_type, iterations, trials")
}

data_path <- args[1]
dep_var <- args[2]
media_vars <- unlist(strsplit(args[3], ","))
control_vars <- if (is.na(args[4]) || args[4] == "" || args[4] == "none") NULL else unlist(strsplit(args[4], ","))
date_var <- if (is.na(args[5]) || args[5] == "" || args[5] == "none") NULL else args[5]
adstock_type <- args[6]
iterations <- as.integer(args[7])
trials <- as.integer(args[8])
User <- args[9]
Project_Name <- sub("\\.csv$", "", args[10])
ID <- args[11]


data <- read.csv(data_path)
if (!is.null(date_var)) data[[date_var]] <- as.Date(data[[date_var]])

hyperparameters <- list(
  train_size = c(0.5, 0.9)
)
if (adstock_type == "geometric") {
  for (media_var in media_vars) {
    hyperparameters[[paste0(media_var, "_alphas")]] <- c(0.001, 4)
    hyperparameters[[paste0(media_var, "_gammas")]] <- c(0.2, 0.95)
    hyperparameters[[paste0(media_var, "_thetas")]] <- c(0.01, 0.5)
  }
} else if (adstock_type == "weibull") {
  for (media_var in media_vars) {
    hyperparameters[[paste0(media_var, "_shapes")]] <- c(0.0001, 3)
    hyperparameters[[paste0(media_var, "_scales")]] <- c(0.05, 5)
    hyperparameters[[paste0(media_var, "_gammas")]] <- c(0.2, 0.95)
  }
}
message("Hyperparameters: ", paste(names(hyperparameters), collapse = ", "))

writeLines(paste(names(hyperparameters), collapse = "\n"), file.path("robyn_output", "hyperparameters.txt"))
writeLines(c(
  paste("Robyn version:", packageVersion("Robyn")),
  paste("Media variables:", paste(media_vars, collapse = ", ")),
  paste("Control variables:", if(is.null(control_vars)) "none" else paste(control_vars, collapse = ", ")),
  paste("Dependent variable:", dep_var),
  paste("Date variable:", if(is.null(date_var)) "none" else date_var),
  paste("Adstock type:", adstock_type),
  paste("Iterations:", iterations),
  paste("Trials:", trials)
), file.path("robyn_output", "inputs.txt"))

output_dir <- "C:\\Users\\MM3815\\Downloads\\data_cleaning_app\\robyn_output"
dir.create(output_dir, showWarnings = FALSE)
dir.create(file.path(output_dir, "plots"), showWarnings = FALSE)
timestamp <- format(Sys.time(), "%Y%m%d%H%M")

# Define the custom folder name
custom_folder_name <- paste(User, Project_Name, ID, timestamp, sep = "~")
custom_output_dir <- file.path(output_dir, "plots", custom_folder_name)
#dir.create(custom_output_dir, recursive = TRUE, showWarnings = FALSE)

write.csv(data, file.path(output_dir, "input_data.csv"), row.names = FALSE)
message("Starting robyn_inputs")
InputCollect <- Robyn::robyn_inputs(
  dt_input = data,
  dt_holidays = Robyn::dt_prophet_holidays,
  date_var = date_var,
  dep_var = dep_var,
  dep_var_type = "revenue",
  prophet_vars = c("trend", "season", "holiday"),
  prophet_country = "US",
  context_vars = control_vars,
  paid_media_spends = media_vars,
  paid_media_vars = media_vars,
  adstock = adstock_type,
  hyperparameters = hyperparameters,
  cores = 1
)
message("robyn_inputs completed")

message("Starting robyn_run")
OutputModels <- Robyn::robyn_run(
  InputCollect = InputCollect,
  iterations = iterations,
  trials = trials,
  cores = 1,
  ts_validation = TRUE,
  verbose = TRUE
)
message("robyn_run completed")

# Get list of folders in plots directory before running robyn_outputs
plots_dir <- file.path(output_dir, "plots")
pre_folders <- list.dirs(plots_dir, recursive = FALSE, full.names = TRUE)

message("Starting robyn_outputs")
OutputCollect <- Robyn::robyn_outputs(
  InputCollect = InputCollect,
  OutputModels = OutputModels,
  pareto_fronts = 1,
  plot_folder = file.path(output_dir, "plots"),
  csv_out = "all",
  cores = 1
)
message("robyn_outputs completed")
# saveRDS(OutputCollect, InputCollect, file = file.path(output_dir, "all_models.rds"))
# Save OutputCollect as JSON
json_file <- file.path(output_dir, "all_models.json")
remove_unserializable <- function(x) {
  # Remove ggplot and try-error objects recursively
  if (inherits(x, "ggplot") || inherits(x, "try-error")) {
    return(NULL)
  } else if (is.list(x)) {
    return(lapply(x, remove_unserializable))
  } else {
    return(x)
  }
}

OutputCollect_clean <- remove_unserializable(OutputCollect)
InputCollect_clean <- remove_unserializable(InputCollect)

library(jsonlite)
write_json(
  list(OutputCollect = OutputCollect_clean, InputCollect = InputCollect_clean),
  path = json_file,
  pretty = TRUE,
  auto_unbox = TRUE
)


message("OutputCollect saved as JSON at: ", json_file)

# Rename the most recently modified folder in plots directory
plots_dir <- file.path(output_dir, "plots")
Sys.sleep(1)  # Brief delay to ensure folder creation
folder_info <- file.info(list.dirs(plots_dir, recursive = FALSE, full.names = TRUE))
if (nrow(folder_info) > 0) {
  latest_folder <- rownames(folder_info)[which.max(folder_info$mtime)]
  tryCatch({
    file.rename(latest_folder, custom_output_dir)
    message("Renamed latest folder from ", latest_folder, " to ", custom_output_dir)
  }, error = function(e) {
    message("Failed to rename folder ", latest_folder, ": ", e$message)
  })
} else {
  message("No folders found in ", plots_dir)
}

if (!is.null(OutputCollect$allPareto$resultHypParam)) {
  hyp_param <- OutputCollect$allPareto$resultHypParam
  
  nrmse_col <- grep("nrmse", colnames(hyp_param), ignore.case = TRUE, value = TRUE)
  rsq_col <- grep("rsq|r_squared|adj_r2", colnames(hyp_param), ignore.case = TRUE, value = TRUE)
  sol_id_col <- grep("solID|solution_id|model_id", colnames(hyp_param), ignore.case = TRUE, value = TRUE)
  
  if (length(nrmse_col) > 0 && length(rsq_col) > 0 && length(sol_id_col) > 0) {
    # Normalize NRMSE and Adjusted R² for combined ranking
    nrmse_norm <- (hyp_param[[nrmse_col[1]]] - min(hyp_param[[nrmse_col[1]]])) / 
      (max(hyp_param[[nrmse_col[1]]]) - min(hyp_param[[nrmse_col[1]]]))
    adj_r2_norm <- (hyp_param[[rsq_col[1]]] - min(hyp_param[[rsq_col[1]]])) / 
      (max(hyp_param[[rsq_col[1]]]) - min(hyp_param[[rsq_col[1]]]))
    
    # Combine metrics: minimize NRMSE (lower is better) and maximize Adj R² (higher is better)
    combined_score <- nrmse_norm - adj_r2_norm
    best_model_idx <- which.min(combined_score)
    
    # Extract metrics for the best model
    model_id <- hyp_param[best_model_idx, sol_id_col[1]][[1]]
    nrmse <- hyp_param[best_model_idx, nrmse_col[1]][[1]]
    adj_r2 <- hyp_param[best_model_idx, rsq_col[1]][[1]]
    best_model_metrics <- hyp_param[best_model_idx, c(sol_id_col[1], nrmse_col[1], rsq_col[1], "robynPareto")]
    
    message("Selected best model ID: ", model_id)
    message("Best model metrics: ", paste(capture.output(print(best_model_metrics)), collapse = "; "))
    
    # Save initial model metrics to JSON
    model_metrics <- list(
      Model_ID = model_id,
      NRMSE = nrmse,
      Adjusted_R2 = adj_r2
    )
    json_file <- file.path(output_dir, "model_metrics.json")
    tryCatch({
      jsonlite::write_json(model_metrics, json_file, auto_unbox = TRUE, pretty = TRUE)
      message("Initial model metrics saved to ", json_file)
    }, error = function(e) {
      message("Failed to write initial model_metrics.json: ", e$message)
    })
    
    # Check if allSolutions exists and contains model_id
    if (!is.null(OutputCollect$allSolutions) && length(OutputCollect$allSolutions) > 0) {
      message("Available solution IDs in OutputCollect$allSolutions: ", paste(OutputCollect$allSolutions, collapse = ", "))
      # Filter hyp_param to only include models in allSolutions
      valid_hyp_param <- hyp_param[hyp_param[[sol_id_col[1]]] %in% OutputCollect$allSolutions, ]
      if (nrow(valid_hyp_param) > 0) {
        # Recompute best model among valid solutions
        nrmse_norm_valid <- (valid_hyp_param[[nrmse_col[1]]] - min(valid_hyp_param[[nrmse_col[1]]])) / 
          (max(valid_hyp_param[[nrmse_col[1]]]) - min(valid_hyp_param[[nrmse_col[1]]]))
        adj_r2_norm_valid <- (valid_hyp_param[[rsq_col[1]]] - min(valid_hyp_param[[rsq_col[1]]])) / 
          (max(valid_hyp_param[[rsq_col[1]]]) - min(valid_hyp_param[[rsq_col[1]]]))
        combined_score_valid <- nrmse_norm_valid - adj_r2_norm_valid
        best_valid_idx <- which.min(combined_score_valid)
        
        # Update model_id and metrics for valid model
        model_id <- valid_hyp_param[best_valid_idx, sol_id_col[1]][[1]]
        nrmse <- valid_hyp_param[best_valid_idx, nrmse_col[1]][[1]]
        adj_r2 <- valid_hyp_param[best_valid_idx, rsq_col[1]][[1]]
        best_model_metrics <- valid_hyp_param[best_valid_idx, c(sol_id_col[1], nrmse_col[1], rsq_col[1], "robynPareto")]
        
        message("Updated best model ID (valid in allSolutions): ", model_id)
        message("Updated best model metrics: ", paste(capture.output(print(best_model_metrics)), collapse = "; "))
        
        # Update JSON with valid model metrics
        model_metrics <- list(
          Model_ID = model_id,
          NRMSE = nrmse,
          Adjusted_R2 = adj_r2
        )
        tryCatch({
          jsonlite::write_json(model_metrics, json_file, auto_unbox = TRUE, pretty = TRUE)
          message("Updated model metrics saved to ", json_file)
        }, error = function(e) {
          message("Failed to update model_metrics.json: ", e$message)
        })
        
        # Create Best_Models folder
        # Create Best_Models folder if it doesn't exist
        # Define paths
        best_models_dir <- file.path(output_dir, "Best_Models")
        
        # Ensure Best_Models folder exists
        if (!dir.exists(best_models_dir)) {
          dir.create(best_models_dir, recursive = TRUE)
        }
        
        # Timestamp for unique filename
        timestamp <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
        
        # Generate one-pager plot
        message("Generating one-pager plot for best model ID: ", model_id)
        
        tryCatch({
          # Run robyn_onepagers
          Robyn::robyn_onepagers(
            InputCollect = InputCollect,
            OutputCollect = OutputCollect,
            select_model = model_id,
            plot_folder = best_models_dir,
            quiet = TRUE
          )
          
          Sys.sleep(1)  # Small delay to ensure file saves
          
          # 💥 Scan output_dir for any misplaced plot matching 'Best_Models*.png'
          misplaced_png_files <- list.files(
            path = output_dir,
            pattern = paste0("^Best_Models.*", model_id, ".*\\.png$"),
            full.names = TRUE
          )
          
          if (length(misplaced_png_files) > 0) {
            for (file in misplaced_png_files) {
              # Build new name with timestamp
              new_file_name <- paste0(model_id, "_", timestamp, "_onepager.png")
              new_file_path <- file.path(best_models_dir, new_file_name)
              file.rename(file, new_file_path)
              message("✅ Moved and renamed one-pager PNG to: ", new_file_path)
            }
          } else {
            message("⚠️ No misplaced PNG file found for model ID: ", model_id)
          }
          
          # Optional: Do same for PDF if needed
          misplaced_pdf_files <- list.files(
            path = output_dir,
            pattern = paste0("^Best_Models.*", model_id, ".*\\.pdf$"),
            full.names = TRUE
          )
          
          if (length(misplaced_pdf_files) > 0) {
            for (file in misplaced_pdf_files) {
              new_file_name <- paste0(model_id, "_", timestamp, "_onepager.pdf")
              new_file_path <- file.path(best_models_dir, new_file_name)
              file.rename(file, new_file_path)
              message("✅ Moved and renamed one-pager PDF to: ", new_file_path)
            }
          }
        }, error = function(e) {
          message("❌ Failed to generate or save one-pager: ", e$message)
        })
        
        
        
        
        
      } else {
        message("Error: No models in resultHypParam match OutputCollect$allSolutions")
        message("Available solution IDs: ", paste(OutputCollect$allSolutions, collapse = ", "))
      }
    } else {
      message("Error: OutputCollect$allSolutions is empty or NULL. Skipping one-pager plot generation.")
      message("Check robyn_run or robyn_outputs for issues.")
    }
  } else {
    message("Error: Required columns (NRMSE, Adjusted R², or solID) not found in resultHypParam")
    message("Available columns: ", paste(colnames(hyp_param), collapse = ", "))
  }
} else {
  message("Error: resultHypParam is NULL or unavailable")
}

# Save Robyn object
saveRDS(InputCollect, file = file.path(output_dir, "InputCollect.rds"))
saveRDS(OutputCollect, file = file.path(output_dir, "OutputCollect.rds"))



if (!is.null(OutputCollect$allSolutions) && length(OutputCollect$allSolutions) > 0) {
  message("✅ allSolutions length: ", length(OutputCollect$allSolutions))
  
  if (!is.null(OutputCollect$allPareto$resultHypParam)) {
    hyp_param <- OutputCollect$allPareto$resultHypParam
    sol_id_col <- grep("solID|solution_id|model_id", colnames(hyp_param), ignore.case = TRUE, value = TRUE)
    
    if (length(sol_id_col) == 0) stop("🚨 No solution ID column found.")
    if (!"robynPareto" %in% colnames(hyp_param)) stop("🚨 robynPareto column missing.")
    
    pareto_solutions <- hyp_param[hyp_param$robynPareto == 1, , drop = FALSE]
    if (nrow(pareto_solutions) == 0) stop("🚨 No Pareto front solutions found.")
    
    exclude_cols <- c(
      sol_id_col, "robynPareto", "cluster",
      grep("_alphas|_gammas|_thetas|_shapes|_scales|train_size|lambda|pos|Elapsed|iterNG|iterPar|iterations|coef0|trial", 
           colnames(hyp_param), value = TRUE)
    )
    metric_cols <- names(pareto_solutions)[
      sapply(pareto_solutions, is.numeric) & !names(pareto_solutions) %in% exclude_cols
    ]
    
    if (length(metric_cols) == 0) {
      model_metrics_df <- data.frame(Model_ID = pareto_solutions[[sol_id_col[1]]])
    } else {
      model_metrics_df <- pareto_solutions[, c(sol_id_col[1], metric_cols), drop = FALSE]
      names(model_metrics_df)[1] <- "Model_ID"
      if ("nrmse" %in% names(model_metrics_df)) {
        model_metrics_df <- model_metrics_df[order(model_metrics_df$nrmse), ]
      }
    }
    
    # Save Pareto front metrics to CSV (preserve NAs)
    solutions_csv <- file.path(output_dir, "model_solutions.csv")
    write.csv(model_metrics_df, solutions_csv, row.names = FALSE, na = "")
    message("✅ Saved Pareto front model solutions to ", solutions_csv)
    
    # Save summary
    summary_txt <- file.path(output_dir, "summary.txt")
    sink(summary_txt)
    cat("Pareto Front Model Metrics:\n\n")
    print(model_metrics_df)
    sink()
    message("✅ Saved summary to ", summary_txt)
    
    # Save plots
    plot_dir <- file.path(output_dir, "plots")
    dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)
    
    message("✅ Saved model decomposition and response curve plots")
    
  } else {
    stop("🚨 resultHypParam is NULL or unavailable.")
  }
} else {
  stop("🚨 No valid solutions found in OutputCollect$allSolutions.")
}

message("Modeling completed successfully")
#quit(status = 0)

