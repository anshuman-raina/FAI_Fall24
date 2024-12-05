import re

with open('Empty_Shelf_Detection.py', 'r') as file:
    content = file.read()

imports = re.findall(r'^\s*import\s+(\w+)', content, re.MULTILINE) + \
          re.findall(r'^\s*from\s+(\w+)', content, re.MULTILINE)

with open('requirements.txt', 'w') as req_file:
    for library in set(imports):
        req_file.write(f"{library}\n")