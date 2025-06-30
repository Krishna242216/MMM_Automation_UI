args <- commandArgs(trailingOnly = TRUE)

personal_lib <- "C:/Users/MM3815/Documents/R/library"
if (!dir.exists(personal_lib)) dir.create(personal_lib, recursive = TRUE)
.libPaths(personal_lib)

library(Robyn)
library(jsonlite)
# required_packages <- c("Robyn", "jsonlite")
# 
# # Check for missing packages
# missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]
# if (length(missing_packages) > 0) {
#   stop("Error: The following required packages are missing: ", paste(missing_packages, collapse = ", "), 
#        "\nPlease install them manually in R with:\n",
#        "install.packages(c('", paste(missing_packages, collapse = "', '"), "'), dependencies = TRUE, lib = '", personal_lib, "')")
# }

# Define paths
output_dir <- "C:\\Users\\MM3815\\Downloads\\data_cleaning_app\\robyn_output"
input_rds_path <- file.path(output_dir, "InputCollect.rds")
output_rds_path <- file.path(output_dir, "OutputCollect.rds")
json_path <- file.path(output_dir, "model.json")

# Load Robyn objects
InputCollect <- readRDS(input_rds_path)
OutputCollect <- readRDS(output_rds_path)
model_info <- fromJSON(json_path)

# Read selected model ID from JSON
model_id <- model_info$model_id
allocation_type <- model_info$allocation_type
total_budget <- model_info$fixed_budget
# constraint_lower <- model_info$constraints$lower
# constraint_upper <- model_info$constraints$upper
constraints_list <- model_info$constraints

# Convert to named vectors for lower and upper constraints
channel_constr_low <- sapply(constraints_list, function(x) x$lower)
channel_constr_up <- sapply(constraints_list, function(x) x$upper)

# Optionally convert to named vectors if you want names retained
# names(channel_constr_low) <- names(constraints_list)
# names(channel_constr_up) <- names(constraints_list)

# If needed: convert to un-named numeric vectors
channel_constr_low <- as.numeric(channel_constr_low)
channel_constr_up <- as.numeric(channel_constr_up)
time_range <- model_info$time_range
message("✅ Selected Model ID: ", model_id)

# Save the selected model object
selected_model <- OutputCollect$resultHypParam[[model_id]]

# saveRDS(selected_model, file = file.path(output_dir, paste0("SelectedModel_", model_id, ".rds")))
# option -2
# full_model <- list(
#   InputCollect = InputCollect,
#   OutputCollect = OutputCollect,
#   selected_model_id = model_id
# )
# 
# saveRDS(full_model, file = file.path(output_dir, paste0("RobynModel_", model_id, ".rds")))

# Extract media variables
media_vars <- InputCollect$paid_media_vars
if (is.null(media_vars)) stop("🚨 Paid media variables not found in InputCollect.")

# Budget Optimization Section
message("🔄 Starting budget optimization")
budget_output_dir <- file.path(output_dir, "budget_optimization")
dir.create(budget_output_dir, showWarnings = FALSE)

scenarios <- c("max_historical_response", "max_response", "target_efficiency")
allocator_results <- list()

for (scenario in scenarios) {
  message("📊 Running budget allocation for scenario: ", scenario)
  target_value <- if (scenario == "target_efficiency") 1.5 else NULL
  # 
  # # Handle lower constraints
  # if (!is.null(constraint_lower)) {
  #   if (length(constraint_lower) == length(media_vars)) {
  #     channel_constr_low <- constraint_lower  # Already a vector → use as-is
  #   } else {
  #     channel_constr_low <- rep(constraint_lower, length(media_vars))  # Scalar → replicate
  #   }
  # } else {
  #   channel_constr_low <- rep(0, length(media_vars))  # Not provided → use default
  # }
  # 
  # # Handle upper constraints
  # if (!is.null(constraint_upper)) {
  #   if (length(constraint_upper) == length(media_vars)) {
  #     channel_constr_up <- constraint_upper
  #   } else {
  #     channel_constr_up <- rep(constraint_upper, length(media_vars))
  #   }
  # } else {
  #   channel_constr_up <- rep(10, length(media_vars))
  # }
  
  message("mediavaerssssss", media_vars)
  
  # Assign names
  names(channel_constr_low) <- media_vars
  names(channel_constr_up) <- media_vars
  print("below:")
  message(channel_constr_low)
  message(channel_constr_up)
  
  cat("🔧 Media Variable Constraints:\n")
  for (i in seq_along(media_vars)) {
    cat(sprintf("• %s → Lower: %.3f | Upper: %.3f\n", 
                media_vars[i], 
                channel_constr_low[i], 
                channel_constr_up[i]))
  }
  
  AllocatorCollect <- Robyn::robyn_allocator(
    InputCollect = InputCollect,
    OutputCollect = OutputCollect,
    select_model = model_id,
    scenario = scenario,
    target_value = target_value,
    channel_constr_low = channel_constr_low,
    channel_constr_up = channel_constr_up,
    total_budget = if (!is.null(total_budget)) total_budget else NULL,  # 👈 ADD THIS LINE
    date_range = if (!is.null(time_range)) time_range else NULL,
    export = TRUE,
    plot_folder = budget_output_dir
  )
  
  
  
  print("jgdjwlsn")
  print(target_value)
  
  allocator_results[[scenario]] <- AllocatorCollect
  
  if (!is.null(AllocatorCollect$dt_optimisation)) {
    write.csv(
      AllocatorCollect$dt_optimisation,
      file.path(budget_output_dir, paste0("budget_allocation_", scenario, ".csv")),
      row.names = FALSE
    )
  }
  
  if (!is.null(AllocatorCollect$plot)) {
    png(file.path(budget_output_dir, paste0("budget_allocation_plot_", scenario, ".png")),
        width = 1000, height = 800)
    print(AllocatorCollect$plot)
    dev.off()
  }
}

message("✅ Budget allocation completed for all scenarios.")
