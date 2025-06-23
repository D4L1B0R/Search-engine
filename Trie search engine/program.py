import fitz  # PyMuPDF
import re
from collections import defaultdict
from difflib import get_close_matches
import os
import pickle

def save_object(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def load_object(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def save_search_results_as_pdf(results, pages, output_file, original_pdf_path='Data Structures and Algorithms in Python.pdf'):
    doc = fitz.open(original_pdf_path)
    output_doc = fitz.open()
    result_pages = [result[0] for result in results[:10]]
    for page_num in result_pages:
        page_text = pages[page_num][1]
        original_page = doc.load_page(page_num)
        new_page = output_doc.new_page(width=original_page.rect.width, height=original_page.rect.height)
        new_page.insert_text((50, 50), page_text) 

    output_doc.save(output_file)

# Function to highlight keywords in a PDF
def highlight_keywords_in_pdf(keywords, input_pdf, output_pdf, phrase):
    # Open the input PDF document
    doc = fitz.open(input_pdf)
    output_doc = fitz.open()

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        operators = {'and', 'or', 'not', '(', ')'}  # Define the set of operators

        if phrase:
            instances = page.search_for(keywords[0][1:-1])
            for rect in instances:
                page.add_highlight_annot(rect)

            output_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        else:
            for keyword in keywords:
                found = False
                for operator in operators:
                    if operator == keyword.lower():
                        found = True
                if not found:
                    instances = page.search_for(keyword)
                    for rect in instances:
                        page.add_highlight_annot(rect)

            output_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

    # Save the highlighted PDF document
    output_doc.save(output_pdf)

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.page_numbers = set()  # Use set for unique page numbers

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, page_num):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.page_numbers.add(page_num)  # Store page numbers in a set for uniqueness

    def search(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return set()  # Return empty set if prefix not found
            node = node.children[char]
        return self._collect_all_words(node)

    def _collect_all_words(self, node):
        result = set()
        if node.is_end_of_word:
            result.update(node.page_numbers)
        for child in node.children.values():
            result.update(self._collect_all_words(child))
        return result

class PageGraph:
    def __init__(self):
        self.graph = defaultdict(set)
    
    def add_edge(self, from_page, to_page):
        self.graph[from_page].add(to_page)
    
    def get_links(self, page):
        return self.graph[page]
    
    def page_rank(self, index):
        scores = defaultdict(float)
        damping_factor = 0.85
        num_iterations = 10
        
        for page in index:
            scores[page] = 1.0 / len(index)
        
        for _ in range(num_iterations):
            new_scores = defaultdict(float)
            for page in scores:
                link_sum = sum(scores[linked_page] / len(self.graph[linked_page]) for linked_page in self.graph[page])
                new_scores[page] = (1 - damping_factor) / len(index) + damping_factor * link_sum
            scores = new_scores
        
        return scores

def parse_pdf(file_path):
    doc = fitz.open(file_path)
    pages = []
    links = defaultdict(list)

    link_pattern = re.compile(r'\b(?:see|on|page)\s+pages?\s+(\d+)', re.IGNORECASE)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")  # Get all text elements
        text += page.get_text("blocks")  # Get text from blocks
        text += page.get_text("dict")  # Get text from dictionaries
        tables = page.get_table()  # Extract tables if any

        # Extract words from regular text
        words = re.findall(r'\b\w+\b', text.lower())
        pages.append((page_num, words))

        # Extract words from tables
        for table in tables:
            for row in table:
                for cell in row:
                    cell_words = re.findall(r'\b\w+\b', cell.lower())
                    words.extend(cell_words)

        # Extract words from headers and footers
        header = page.get_text("header")
        footer = page.get_text("footer")
        header_words = re.findall(r'\b\w+\b', header.lower())
        footer_words = re.findall(r'\b\w+\b', footer.lower())
        words.extend(header_words)
        words.extend(footer_words)

        # Extract links (if any) from the page text
        for match in link_pattern.finditer(text):
            target_page = int(match.group(1)) - 1
            if 0 <= target_page < len(doc):
                links[page_num].append(target_page)

    return pages, links

def parse_query(query):
    query = query.lower()
    # Use regex to handle tokens with AND, OR, NOT, parentheses, and phrases within quotes
    if query.endswith('"') and query.startswith('"'):
        tokens = []
        tokens.append(query)
    else:   
        tokens = re.findall(r'\b(?:and|or|not|\(|\)|"\b[^"]*"\b|\w+)\b', query)

    return tokens
    
def infix_to_postfix(tokens):
    precedence = {'and': 2, 'or': 1, 'not': 3}  # Add 'not' with highest precedence
    associativity = {'and': 'left', 'or': 'left', 'not': 'right'}  # 'not' is right-associative
    output = []
    operator_stack = []
    
    for token in tokens:
        if isinstance(token, str) and token.lower() in precedence:
            while (operator_stack and operator_stack[-1].lower() != '(' and
                   (precedence[operator_stack[-1].lower()] > precedence[token.lower()] or
                    (precedence[operator_stack[-1].lower()] == precedence[token.lower()] and associativity[token.lower()] == 'left'))):
                output.append(operator_stack.pop())
            operator_stack.append(token.lower())
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output.append(operator_stack.pop())
            operator_stack.pop()  # pop '('
        else:
            output.append(token)
    
    while operator_stack:
        output.append(operator_stack.pop())
    
    return output
    
def evaluate_query(tokens, trie, pages):
    postfix_tokens = infix_to_postfix(tokens)  # Convert tokens to postfix notation
    stack = []
    phrase = False
    for token in postfix_tokens:
        if token.startswith('"') and token.endswith('"'):
            phrase = token[1:-1].lower()  # Remove quotes and normalize
            results = evaluate_phrase(phrase, pages)
            stack.append(results)
            phrase = True
        elif str(token).lower() == 'not':
            if len(stack) >= 2:
                op2 = stack.pop()
                op1 = stack.pop()
                changed_op1 = op1 - (op1 & op2)  # Remove common elements from op1
                stack.append(changed_op1)
        elif str(token).lower() == 'and':
            if len(stack) >= 2:
                op2 = stack.pop()
                op1 = stack.pop()
                result = op1 & op2
                stack.append(result)
        elif str(token).lower() == 'or':
            if len(stack) >= 2:
                op2 = stack.pop()
                op1 = stack.pop()
                result = op1 | op2
                stack.append(result)
        else:
            results = trie.search(token)
            stack.append(results)
    
    # At the end, stack should contain the final result
    if stack:
        return stack[0], phrase
    else:
        return set(), phrase

def evaluate_phrase(phrase, pages):
    phrase_length = len(phrase)
    results = set()

    for page_num, (page_index, words) in enumerate(pages):
        word_positions = defaultdict(list)
        for pos, word in enumerate(words):
            word_positions[pos].append(word)
        candidate_positions = []
        for position, letter in word_positions.items():
            if letter[0].lower() == phrase[0].lower():
                candidate_positions.append(position)
        for start_pos in candidate_positions:
            match = True
            for j in range(1, phrase_length):
                if word_positions[start_pos + j][0].lower() != phrase[j].lower():
                    match = False
                    break
            if match:
                results.add(page_num)
                break

    return results

def search(query, pages, page_graph, trie):
    tokens = parse_query(query)
    page_nums, phrase = evaluate_query(tokens, trie, pages)

    if not page_nums:
        return [], phrase

    page_scores = defaultdict(int)
    
    if phrase:
        tokens = tokens[0][1:-1]
        for page_num in page_nums:
            word_count = 0
            word_count += pages[page_num][1].lower().count(tokens)
            
            page_scores[page_num] += word_count
            incoming_links = [page for page, links in page_graph.graph.items() if page_num in links]
            page_scores[page_num] += len(incoming_links) * 0.5

            for linked_page in incoming_links:
                word_count_in_linked_page = pages[linked_page][1].lower().count(tokens)
                page_scores[page_num] += word_count_in_linked_page * 0.2
        
        sorted_pages = sorted(page_scores.items(), key=lambda x: (-x[1], x[0]))

        results = []
        for page_num, score in sorted_pages:
            page_text = pages[page_num][1]
            snippets = []
            for match in re.finditer(tokens, page_text, re.IGNORECASE):
                start = max(match.start() - 30, 0)
                end = min(match.end() + 30, len(page_text))
                snippet = page_text[start:end].replace('\n', ' ')
                snippets.append(snippet)
            results.append((page_num, score, snippets))

        return results, phrase

    # Calculate scores for each page based on the query
    for page_num in page_nums:
        word_count = 0
        for word in tokens:
            if word not in ('and', 'or', 'not', '(', ')'):
                word_count += pages[page_num][1].lower().count(word)
        
        page_scores[page_num] += word_count

        incoming_links = [page for page, links in page_graph.graph.items() if page_num in links]
        page_scores[page_num] += len(incoming_links) * 0.5

        for linked_page in incoming_links:
            word_count_in_linked_page = sum(pages[linked_page][1].lower().count(word) for word in tokens if word not in ('and', 'or', 'not', '(', ')'))
            page_scores[page_num] += word_count_in_linked_page * 0.2

    sorted_pages = sorted(page_scores.items(), key=lambda x: (-x[1], x[0]))

    results = []
    for page_num, score in sorted_pages:
        page_text = pages[page_num][1]
        snippets = []
        for word in tokens:
            if word not in ('and', 'or', 'not', '(', ')'):
                for match in re.finditer(word, page_text, re.IGNORECASE):
                    start = max(match.start() - 30, 0)
                    end = min(match.end() + 30, len(page_text))
                    snippet = page_text[start:end].replace('\n', ' ')
                    snippets.append(snippet)
        results.append((page_num, score, snippets))

    return results, phrase

def display_results(results, query_words, phrase, page_number=1, results_per_page=10):
    start_index = (page_number - 1) * results_per_page
    end_index = start_index + results_per_page
    paginated_results = results[start_index:end_index]

    if not paginated_results:
        print("No more results.")
        return False

    operators = {'and', 'or', 'not', '(', ')'}  # Define the set of operators

    for rank, (page_num, score, snippets) in enumerate(paginated_results, start=start_index + 1):
        print("\n" + f"{rank}. Page {page_num} (score: {score})")
        for snippet in snippets:
            highlighted_snippet = ""
            tokens = re.findall(r'\w+|[\(\)"]', snippet)  # Split snippet into words and operators

            if phrase:
                query = query_words[0][1:-1]
                query_list = query.split()
                length = len(query_list)
                pairs = []
                for i in range(len(tokens)):
                    if i + length - 1 > len(tokens[i:]):
                        break
                    sentence = ""
                    for j in range(length - 1):
                        sentence += tokens[i + j] + " "
                    sentence += tokens[i + length - 1]
                    pairs.append(sentence)
                
                for pair in pairs:
                    highlighted_snippet += re.sub(
                        f"(?i)({re.escape(query)})",
                        r'\033[44;33m\1\033[m',
                        pair
                    ) + " "
            else:
                for token in tokens:
                    found = False
                    for operator in operators:
                        if operator in token.lower():
                            found = True
                    if not found:
                        highlighted_snippet += re.sub(
                            f"(?i)({'|'.join(map(re.escape, query_words))})",
                            r'\033[44;33m\1\033[m',
                            token
                        ) + " "

            print(f"  ...{highlighted_snippet}...")

    if end_index < len(results):
        print("\nThere are more pages to be shown.")
    else:
        return False
    
    return True

def paginate_results(results, phrase, query_words):
    page_number = 1
    while display_results(results, query_words, phrase, page_number):
        while True:
            command = input("See more results? [y/n]: ").strip().lower()
            if command == 'y':
                page_number += 1
                break
            elif command == 'n':
                return
            else:
                print("Invalid command. Please enter 'y' or 'n'.")

def suggest_alternatives(query, index, trie, pages, page_graph):
    query_words = re.findall(r'\b\w+\b', query.lower())
    suggestions = {}
    
    for word in query_words:
        if not trie.search(word):
            close_matches = get_close_matches(word, index.keys(), n=5, cutoff=0.6)
            word_scores = {}
            if close_matches:
                for close_match in close_matches:
                    results, _ = search(close_match, pages, page_graph, trie)
                    total_score = sum(score for _, score, _ in results)
                    word_scores[close_match] = total_score
            close_matches = sorted(word_scores, key=word_scores.get, reverse=True)
            suggestions[word] = close_matches[:5]
            print(word_scores)
    
    return suggestions

def autocomplete(prefix, index, trie, pages, page_graph):
    # Filter words that start with the prefix
    words = [word for word in index.keys() if word.startswith(prefix[:-1])]
    
    # Calculate scores based on search results
    word_scores = {}
    for word in words:
        results, _ = search(word, pages, page_graph, trie)
        total_score = sum(score for _, score, _ in results)
        word_scores[word] = total_score
    
    # Sort words based on their scores and return the top 3-5 suggestions
    sorted_words = sorted(word_scores, key=word_scores.get, reverse=True)
    suggestions = sorted_words[:5]
    
    return suggestions

def main():
    pdf_file_path = 'Data Structures and Algorithms in Python.pdf'
    pages_pickle_path = 'pages.pkl'
    index_pickle_path = 'index.pkl'
    links_pickle_path = 'links.pkl'
    trie_pickle_path = 'trie.pkl'
    graph_pickle_path = 'graph.pkl'

    if all(os.path.exists(path) for path in [pages_pickle_path, index_pickle_path, links_pickle_path, trie_pickle_path, graph_pickle_path]):
        pages = load_object(pages_pickle_path)
        index = load_object(index_pickle_path)
        links = load_object(links_pickle_path)
        trie = load_object(trie_pickle_path)
        graph = load_object(graph_pickle_path)
    else:
        pages, links = parse_pdf(pdf_file_path)

        trie = Trie()
        for word, page_nums in index.items():
            for page_num in page_nums:
                trie.insert(word, page_num)

        graph = PageGraph()
        for from_page, to_pages in links.items():
            for to_page in to_pages:
                graph.add_edge(from_page, to_page)

        save_object(pages, pages_pickle_path)
        save_object(index, index_pickle_path)
        save_object(links, links_pickle_path)
        save_object(trie, trie_pickle_path)
        save_object(graph, graph_pickle_path)

    while True:
        query = input("Enter search query (or '*help' for commands): ")
        if query.lower() == '*help':
            print("Special commands:\n"
                  "  '*exit' or '*x' - Exit the program\n"
                  "  'word*' - For autocomplete functionality\n")
            continue

        if query.lower() == '*exit' or query.lower() == '*x':
            break
        
        if query.endswith('*'):
            # Autocomplete suggestions
            prefix = query.split()[-1]  # Get the last word in the query
            completions = autocomplete(prefix, index, trie, pages, graph)
            if completions:
                print("Autocomplete suggestions: ", ", ".join(completions))
            continue

        results, phrase = search(query, pages, graph, trie)

        if not results:
            suggestions = suggest_alternatives(query, index, trie, pages, graph)
            if suggestions:
                print("No results found. Did you mean:")
                for word, alternatives in suggestions.items():
                    print(f"  {word}: {', '.join(alternatives)}")
            else:
                print("No results found.")

        # Save search results as a PDF
        else:
            # Save search results as a PDF
            save_file = 'search_results.pdf'
            save_search_results_as_pdf(results, pages, save_file, pdf_file_path)

            # Highlight keywords in the saved PDF
            keywords = parse_query(query)
            highlight_keywords_in_pdf(keywords, save_file, 'highlighted_search_results.pdf', phrase)

            paginate_results(results, phrase, query_words=parse_query(query))

        print("\n <" + "-" * 100 + ">\n")

    print("Exiting the program. Goodbye!")

if __name__ == "__main__":
    main()