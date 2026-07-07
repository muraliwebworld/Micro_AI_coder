import json

input_file = "datasets/generated_projects_final.jsonl"
output_file = "datasets/ai_training_data.jsonl"

# Define the system context to instruct the AI on its persona/role
SYSTEM_PROMPT = (
    "You are an expert software engineer. Provide high-quality code implementations, "
    "clean architectural examples, and thorough structural documentation based on the user's request."
)

converted_count = 0

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        if not line.strip():
            continue
        
        try:
            # Parse the current JSON row
            data = json.loads(line)
            
            project_name = data.get("project", "Unknown Project")
            file_name = data.get("file", "code_snippet")
            description = data.get("description", "")
            code_content = data.get("code", "")
            
            # 1. Structure the User Prompt dynamically using available attributes
            user_message = f"Write the content for '{file_name}' belonging to {project_name}.\nContext: {description}"
            
            # 2. Structure the Assistant Response
            assistant_message = f"Here is the implementation file `{file_name}`:\n\n```\n{code_content}\n```"
            
            # 3. Combine into standard ChatML / OpenAI conversational fine-tuning format
            training_row = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message}
                ]
            }
            
            # Save as JSONL line
            outfile.write(json.dumps(training_row) + "\n")
            converted_count += 1
            
        except json.JSONDecodeError:
            print(f"Skipping malformed row: {line[:100]}...")
        except Exception as e:
            print(f"Error processing row: {e}")

print(f"Successfully converted {converted_count} entries to AI training format saved in '{output_file}'.")