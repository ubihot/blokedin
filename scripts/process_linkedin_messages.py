import csv
from messages import messages

# NOTE what happens when you have a corpus with lots of examples of one label???

"""
use this script to add messages into one column in the csv by inserting new lines as '\n'
"""

def convert_text_into_one_line(txt):
  row = []
  for character in txt:
    if character == "\n":
      row.append("\\")
      row.append("n ")
    elif character == " ":
      row.append(character)
    else:
      row.append(character)
  return "".join(row) # the reason we use this is cause the spaces are already in the row array

if __name__ == "__main__":
  csv_file_path = "../datasets/messages.csv"

  # NOTE maybe add the has_link feature from here instead of directly in the messages
  with open(csv_file_path, "w", newline='') as f:
    csv_headers = ["block", "content", "subject", "has_attachment"]

    writer = csv.writer(f, delimiter=",")
    writer.writerow(csv_headers)

    for message in messages:
      content, block, subject, has_attachment = message.values()
      assert isinstance(block, int) and block == 1 or block == 0, f"block should be either 1 or 0 but got {block}"
      assert isinstance(has_attachment, int) and has_attachment == 0 or has_attachment == 1, f"has_attachment should be either 0 or 1 but got {has_attachment}"
      assert isinstance(subject, str) and len(subject) >= 0, f"subject cannot be not defined, got {subject}"
      content = content.strip()
      one_line_txt = convert_text_into_one_line(content)
      #print(one_line_txt)
      writer.writerow([block, one_line_txt, subject, has_attachment])
