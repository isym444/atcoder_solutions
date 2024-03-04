import random
import string

def generate_good_string_part(available_length):
    if available_length < 2:
        return "", 0

    actions = ["pair"]
    if available_length >= 3:
        actions.extend(["letter"] * 1)  # Making letters 3 times more likely
    if available_length >= 4:
        actions.append("nest")

    action = random.choice(actions)

    if action == "pair":
        return "()", 2
    elif action == "letter":
        # Choose a random number of letters to include, up to 25 or the available length minus 2 for the parentheses
        max_letters = min(available_length - 2, 5)  # Allow up to 25 letters if space permits
        num_letters = random.randint(1, max_letters)
        letters = ''.join(random.choice(string.ascii_lowercase) for _ in range(num_letters))
        return f"({letters})", num_letters + 2
    else:
        nested_string, nested_length = generate_good_string_part(available_length - 2)
        return f"({nested_string})", nested_length + 2

def generate_good_string(max_length=100):
    good_string = ""
    total_length = 0

    while total_length < max_length:
        part, part_length = generate_good_string_part(max_length - total_length)
        if not part:
            break
        good_string += part
        total_length += part_length

    return good_string

def main():
    good_string = generate_good_string()
    good_string = good_string.rstrip()
    print(good_string)

if __name__ == "__main__":
    main()
