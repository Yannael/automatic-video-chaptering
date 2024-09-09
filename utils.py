import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


########################### LLM call ###########################

price_token={'gpt-4o': {'input': 5/1000000, 'output': 15/1000000},
             'gpt-4o-2024-08-06': {'input': 2.5/1000000, 'output': 10/1000000},
             'gpt-4o-mini-2024-07-18': {'input': 0.15/1000000, 'output': 0.6/1000000},
             'llama3-8b-8192' : {'input': 0.05 / 1000000, 'output': 0.08 / 1000000},
             'llama3-70b-8192' : {'input': 0.59 / 1000000, 'output': 0.79 / 1000000},
             'claude-3-5-sonnet-20240620': {'input': 3/1000000, 'output': 15/1000000},
             'claude-3-haiku-20240307': {'input': 0.25/1000000, 'output': 1.25/1000000},
             }
def call_llm(client, model, system_prompt, prompt,
             temperature=0, seed=42, response_format=None, max_tokens=5000):

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=model,
        temperature=temperature,
        seed=seed,
        response_format=response_format,
        max_tokens=max_tokens
    )

    nb_input_tokens = response.usage.prompt_tokens
    nb_output_tokens = response.usage.completion_tokens
    price = nb_input_tokens * price_token[model]['input'] + nb_output_tokens * price_token[model]['output']

    print(f"input tokens: {nb_input_tokens}; output tokens: {nb_output_tokens}, price: {price}")

    response_content=response.choices[0].message.content

    return response_content, nb_input_tokens, nb_output_tokens, price

########################### Step 2: Transcript to paragraph ###########################

system_prompt_transcript_to_paragraphs = f"""

You are a helpful assistant.

Your task is to improve the user input's readability: add punctuation if needed, remove verbal tics, correct grammatical errors, and add appropriate line breaks with '\n\n'.

Put your answer within <answer></answer> tags.

"""



def transcript_to_paragraphs(transcript, llm_client, llm_model, chunk_size=5000, progress=None):

    transcript_as_text = ' '.join([s['text'] for s in transcript])

    paragraphs = []
    last_paragraph = ""

    total_nb_input_tokens, total_nb_output_tokens, total_price = 0, 0, 0

    nb_chunks = int(len(transcript_as_text) / chunk_size) + 1
    progress_i = 0
    print(f"Number of chunks: {nb_chunks}")

    # for i in range(0, 10000, chunk_size):
    for i in range(0, len(transcript_as_text), chunk_size):

        print("i is: " + str(i))

        chunk = last_paragraph + " " + transcript_as_text[i:i + chunk_size]

        if progress is not None:
            progress_i += 1
            progress(progress_i / nb_chunks, desc="Processing")

        found_edited_transcript = False

        while not found_edited_transcript:

            response_content, nb_input_tokens, nb_output_tokens, price = \
                call_llm(llm_client, llm_model,
                         system_prompt=system_prompt_transcript_to_paragraphs, prompt=chunk,
                         temperature=0.2, seed=42, response_format=None)

            if not "</answer>" in response_content:
                response_content += "</answer>"

            # Extract content from <edited_transcript> tags
            pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
            response_content_edited = pattern.findall(response_content)

            if len(response_content_edited) > 0:
                found_edited_transcript = True
                response_content_edited = response_content_edited[0]

            else:
                print("No edited transcript found. Trying again.")
                print(response_content[0:100])
                print(response_content[-100:])

        total_nb_input_tokens += nb_input_tokens
        total_nb_output_tokens += nb_output_tokens
        total_price += price

        paragraphs_chunk = response_content_edited.strip().split('\n\n')

        print('Found paragraphs:', len(paragraphs_chunk))
        last_paragraph = paragraphs_chunk[-1]

        paragraphs += paragraphs_chunk[:-1]

    paragraphs += [last_paragraph]

    paragraphs_dict = [{'paragraph_number': i, 'paragraph_text': paragraph} for i, paragraph in enumerate(paragraphs)]

    return paragraphs_dict, total_nb_input_tokens, total_nb_output_tokens, total_price

########################### Step 3: Infer timestamps ###########################

def transform_text_segments(text_segments, num_words=50):
    # Initialize variables
    transformed_segments = []
    current_index = 0
    num_segments = len(text_segments)

    for i in range(num_segments):

        current_index = i

        # Get the current segment's starting timestamp and text
        current_segment = text_segments[current_index]
        current_text = current_segment['text']

        # Initialize a list to hold the combined text
        combined_text = " ".join(current_text.split()[:num_words])
        number_words_collected = len(current_text.split())

        # Collect words from subsequent segments
        while number_words_collected < num_words and (current_index + 1) < num_segments:
            current_index += 1
            next_segment = text_segments[current_index]
            next_text = next_segment['text']
            next_words = next_text.split()

            # Append words from the next segment
            if number_words_collected + len(next_words) <= num_words:
                combined_text += ' ' + next_text
                number_words_collected += len(next_words)
            else:
                # Only append enough words to reach the num_words limit
                words_needed = num_words - number_words_collected
                combined_text += ' ' + ' '.join(next_words[:words_needed])
                number_words_collected = num_words

        # Append the combined segment to the result
        transformed_segments.append(combined_text)

    return transformed_segments


def add_timestamps_to_paragraphs(transcript, paragraphs, num_words=50):
    list_indices = []

    transcript_num_words = transform_text_segments(transcript, num_words=num_words)

    paragraphs_start_text = [{"start": p['paragraph_number'], "text": p['paragraph_text']} for p in paragraphs]
    paragraphs_num_words = transform_text_segments(paragraphs_start_text, num_words=num_words)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer().fit_transform(transcript_num_words + paragraphs_num_words)
    # Get the TF-IDF vectors for the transcript and the excerpt
    vectors = vectorizer.toarray()

    for i in range(len(paragraphs_num_words)):

        # Extract the TF-IDF vector for the paragraph
        paragraph_vector = vectors[len(transcript_num_words) + i]

        # Calculate the cosine similarity between the paragraph vector and each transcript chunk
        similarities = cosine_similarity(vectors[:len(transcript_num_words)], paragraph_vector.reshape(1, -1))
        # Find the index of the most similar chunk
        best_match_index = int(np.argmax(similarities))

        list_indices.append(best_match_index)

        paragraphs[i]['matched_index'] = best_match_index
        paragraphs[i]['matched_text'] = transcript[best_match_index]['text']
        paragraphs[i]['start_time'] = int(transcript[best_match_index]['start']) - 2
        if paragraphs[i]['start_time'] < 0:
            paragraphs[i]['start_time'] = 0

    return paragraphs

########################### Step 4: Generate table of content ###########################


system_prompt_paragraphs_to_toc = """

	You are a helpful assistant.

	You are given a transcript of a course in JSON format as a list of paragraphs, each containing 'paragraph_number' and 'paragraph_text' keys.

	Your task is to group consecutive paragraphs in chapters for the course and identify meaningful chapter titles.

	Here are the steps to follow:

1. Read the transcript carefully to understand its general structure and the main topics covered.
2. Look for clues that a new chapter is about to start. This could be a change of topic, a change of time or setting, the introduction of new themes or topics, or the speaker's explicit mention of a new part.
3. For each chapter, keep track of the paragraph number that starts the chapter and identify a meaningful chapter title.
4. Chapters should ideally be equally spaced throughout the transcript, and discuss a specific topic.
5. A chapter MUST have more than 4 paragraphs.

	Format your result in JSON, with a list dictionaries for chapters, with 'start_paragraph_number':integer and 'title':string as key:value.

	Example: 
    {"chapters": 
        [{"start_paragraph_number": 0, "title": "Introduction"}, 
         {"start_paragraph_number": 10, "title": "Chapter 1"}
        ]
    }

"""


def paragraphs_to_toc(paragraphs, llm_client, llm_model, chunk_size=100):
    chapters = []
    number_last_chapter = 0

    total_nb_input_tokens, total_nb_output_tokens, total_price = 0, 0, 0

    while number_last_chapter < len(paragraphs):

        print(number_last_chapter)

        chunk = paragraphs[number_last_chapter:(number_last_chapter + chunk_size)]
        chunk = [{'paragraph_number': p['paragraph_number'], 'paragraph_text': p['paragraph_text']} for p in chunk]

        chunk_json_dump = json.dumps(chunk)

        content, nb_input_tokens, nb_output_tokens, price = call_llm( \
            llm_client, llm_model, \
            system_prompt_paragraphs_to_toc, chunk_json_dump, \
            temperature=0, seed=42, response_format={"type": "json_object"})

        total_nb_input_tokens += nb_input_tokens
        total_nb_output_tokens += nb_output_tokens

        chapters_chunk = json.loads(content)['chapters']

        if number_last_chapter == chapters_chunk[-1]['start_paragraph_number']:
            break

        chapters += chapters_chunk[:-1]

        number_last_chapter = chapters_chunk[-1]['start_paragraph_number']
        if number_last_chapter >= len(paragraphs) - 5:
            break

    total_price = (total_nb_input_tokens * price_token[llm_model]['input'] +
                   total_nb_output_tokens * price_token[llm_model]['output'])

    chapters += [chapters_chunk[-1]]

    return chapters, total_nb_input_tokens, total_nb_output_tokens, total_price


########################### Step 5: Chapter rendering functions ###########################

def get_chapters(paragraphs, table_of_content):

    chapters = []

    for i in range(len(table_of_content)):


        if i < len(table_of_content) - 1:

            chapter = {'num_chapter': i,
                       'title': table_of_content[i]['title'],
                       'start_paragraph_number': table_of_content[i]['start_paragraph_number'],
                       'end_paragraph_number': table_of_content[i + 1]['start_paragraph_number'],
                       'start_time': paragraphs[table_of_content[i]['start_paragraph_number']]['start_time'],
                       'end_time': paragraphs[table_of_content[i + 1]['start_paragraph_number']]['start_time'],
                      }

        else:
            chapter = {'num_chapter': i,
                       'title': table_of_content[i]['title'],
                       'start_paragraph_number': table_of_content[i]['start_paragraph_number'],
                       'end_paragraph_number': len(paragraphs),
                       'start_time': paragraphs[table_of_content[i]['start_paragraph_number']]['start_time'],
                       'end_time': paragraphs[-1]['start_time'],
                      }

        paragraphs_chapter = [paragraphs[j]['paragraph_text'] for j in
                                range(chapter['start_paragraph_number'], chapter['end_paragraph_number'])]

        paragraph_timestamps_chapter = [paragraphs[j]['start_time'] for j in
                                          range(chapter['start_paragraph_number'], chapter['end_paragraph_number'])]

        chapter['paragraphs'] = paragraphs_chapter
        chapter['paragraph_timestamps'] = paragraph_timestamps_chapter

        chapters.append(chapter)

    return chapters

def convert_seconds_to_hms(seconds):
    # Calculate hours, minutes, and remaining seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    # Format the result as HH:MM:SS
    return f"{hours:02}:{minutes:02}:{remaining_seconds:02}"

def toc_to_html(chapters):

    toc_html = "<h1>Video chapters</h1><p>\n"

    for chapter in chapters:
        num_chapter = chapter['num_chapter']
        title = chapter['title']

        from_to = convert_seconds_to_hms(int(chapter['start_time'])) + " - "

        toc_html += f"""{from_to}<a href = "#{num_chapter}" >{num_chapter+1} - {title}</a><br>\n"""

    return toc_html


def section_to_html(section_json_data):
    formatted_section = ""

    paragraphs = section_json_data['paragraphs']
    paragraphs_timestamp_hms = [convert_seconds_to_hms(int(section_json_data['paragraph_timestamps'][i])) for i in range(len(paragraphs))]

    for i, (paragraph, paragraph_timestamp_hms) in enumerate(zip(paragraphs, paragraphs_timestamp_hms)):

        formatted_section += f"""
        <div class="row mb-4">
            <div class="col-md-1">
                {paragraph_timestamp_hms}
            </div>
            <div class="col-md-11">
                <p>{paragraph}</p>
            </div>
        </div>"""

    num_section = section_json_data['num_chapter']

    from_to = "From "+convert_seconds_to_hms(int(section_json_data['start_time'])) + " to " + convert_seconds_to_hms(
        int(section_json_data['end_time']))

    title = f"{section_json_data['title']}"

    title_link = f"""<div class="transcript-title-icon" " id="{num_section}">{num_section+1} - {title}</div>"""

    summary_section = f"""
            <h2>{title_link}</h2>
            {from_to}
            <p>
            <div class="summary-section">
                <div class="summary-text" >
                    {formatted_section}
                </div>
            </div>
            """

    return summary_section


def get_result_as_html(chapters, video_id):
    video_embed = f"""
<iframe width="100%" height="400" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
"""

    toc = toc_to_html(chapters)

    edited_transcript = f"""
<h1>Structured transcript</h1>
<p>
"""

    for i in range(len(chapters)):
        chapter_json_data = chapters[i]

        edited_transcript += section_to_html(chapter_json_data)

    result_as_html = f"""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
<div class="container mt-4">
    <div class="content">
        {video_embed}
    </div>
    <p>
    <div class="content">
        {toc}
    </div>
    <p>
    <div class="content">
        {edited_transcript}
    </div>
</div>"""

    return result_as_html

def load_json_chapters(video_id):
    file_name = f"{video_id}.json"
    with open(file_name, 'r') as file:
        chapters = json.load(file)

    return chapters