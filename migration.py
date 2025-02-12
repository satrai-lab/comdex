import os

folder_path = "./docs"
old_string = "satrai-lab.github.io/comdex//"
new_string = "satrai-lab.github.io/comdex/"

for root, _, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            new_content = content.replace(old_string, new_string)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated: {file_path}")
        except Exception as e:
            print(f"Skipping {file_path}: {e}")