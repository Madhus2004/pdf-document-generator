from build_index import extract_text

pdf = "sample.pdf"   # change to your new file name
text = extract_text(pdf)

# Print the first 2000 characters
print(text[:2000])

# Print number of characters
print("\n--- LENGTH OF EXTRACTED TEXT ---")
print(len(text))
