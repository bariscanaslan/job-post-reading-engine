def sort_file_lines(input_filename, output_filename):
    try:
        with open(input_filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        lines.sort()

        with open(output_filename, 'w', encoding='utf-8') as file:
            file.writelines(lines)
        
        print(f"Success: Sorted data has been saved to '{output_filename}'.")

    except FileNotFoundError:
        print("Error: The source file was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

sort_file_lines('most_used_words_for_skills.txt', 'most_used_words_for_skills_sorted.txt')
