danger.import_dangerfile(github)

# Define custom rules and checks
def check_file_sizes
  files = git.modified_files + git.added_files
  large_files = files.select { |file| File.size(file) > 500_000 } # Modify the threshold as needed
  return if large_files.empty?

  message = "The following files are larger than 500 KB:\n\n"
  large_files.each { |file| message += "- #{file}\n" }

  warn message
end

# Specify additional rules and checks
GitHub.checks if: danger.github.pr_title.include?('WIP')

check_file_sizes

# Define other rules, checks, and messages as needed
