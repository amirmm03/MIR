import streamlit as st
import sys


import os
sys.path.insert(0, os.path.join(sys.path[0], '../..'))
# sys.path.append('../')
# print('#'*30)
# print(os.getcwd())
from Logic import utils
import time
from enum import Enum
import random
from Logic.core.utility.snippet import Snippet
from Logic.core.link_analysis.analyzer import LinkAnalyzer
from Logic.core.indexer.index_reader import Index_reader, Indexes


# Insert your path

snippet_obj = Snippet(number_of_words_on_each_side=5)


class color(Enum):
    RED = "#FF0000"
    GREEN = "#00FF00"
    BLUE = "#0000FF"
    YELLOW = "#FFFF00"
    WHITE = "#FFFFFF"
    CYAN = "#00FFFF"
    MAGENTA = "#FF00FF"


# Function to get top X movies by rank
def get_top_x_movies_by_rank(x: int, results: list):
    path = "../Logic/core/index/"
    document_index = Index_reader(path, Indexes.DOCUMENTS)
    corpus = []
    root_set = []
    for movie_id, movie_detail in document_index.index.items():
        movie_title = movie_detail["title"]
        stars = movie_detail["stars"]
        corpus.append({"id": movie_id, "title": movie_title, "stars": stars})

    for element in results:
        movie_id = element[0]
        movie_detail = document_index.index[movie_id]
        movie_title = movie_detail["title"]
        stars = movie_detail["stars"]
        root_set.append({"id": movie_id, "title": movie_title, "stars": stars})
    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=x)
    return actors, movies


# Function to get summary with snippets highlighted
def get_summary_with_snippet(movie_info, query):
    summary = movie_info["first_page_summary"]
    snippet, not_exist_words = snippet_obj.find_snippet(summary, query)
    if "***" in snippet:
        snippet = snippet.split()
        for i in range(len(snippet)):
            current_word = snippet[i]
            if current_word.startswith("***") and current_word.endswith("***"):
                current_word_without_star = current_word[3:-3]
                summary = summary.lower().replace(
                    current_word_without_star,
                    f"<b><font size='4' color={random.choice(list(color)).value}>{current_word_without_star}</font></b>",
                )
    return summary


import base64 
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

imdb_logo_base64 =  get_base64('IMDb_Logo_Square.svg.png')
star_base64 = get_base64('start2.png')
# Function to display search time
def search_time(start, end):
    st.success("Search took: {:.6f} milliseconds".format((end - start) * 1e3))


# Function to handle the search process
def search_handling(
        search_button, search_term, search_max_num, search_weights, search_method,
        unigram_smoothing, alpha, lamda, filter_button, num_filter_results):
    all_documents = []
    for i in utils.movies_dataset:
        summaries = " ".join(utils.movies_dataset[i]['summaries']) if utils.movies_dataset[i]['summaries'] else ''
        genres = " ".join(utils.movies_dataset[i]['genres']) if utils.movies_dataset[i]['genres'] else ''
        stars = " ".join(utils.movies_dataset[i]['stars']) if utils.movies_dataset[i]['stars'] else ''
        all_documents.append(summaries + " " + genres + " " + stars)

    if filter_button:
        if "search_results" in st.session_state:
            top_actors, top_movies = get_top_x_movies_by_rank(num_filter_results, st.session_state["search_results"])
            st.markdown(f"**Top {num_filter_results} Actors:**")
            actors_ = ", ".join(top_actors)
            st.markdown(
                f"<span style='color:{random.choice(list(color)).value}'>{actors_}</span>",
                unsafe_allow_html=True,
            )
            st.markdown("---")

        st.markdown(f"**Top {num_filter_results} Movies:**")
        for movie_id in top_movies:
            info = utils.get_movie_by_id(movie_id, utils.movies_dataset)
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.header(info["title"])
                    st.markdown(f"[Link to movie]({info['URL']})")
                    st.markdown(
                        f"<b>Summary:</b> {get_summary_with_snippet(info, search_term)}",
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Directors:**")
                    for director in info["directors"]:
                        st.markdown(f"- {director}")

                    st.markdown("**Stars:**")
                    for star in info["stars"]:
                        st.markdown(f"- {star}")

                    st.markdown("**Genres:**")
                    for genre in info["genres"]:
                        st.markdown(
                            f"<span style='color:{random.choice(list(color)).value}'>{genre}</span>",
                            unsafe_allow_html=True,
                        )
                with col2:
                    st.image(info["Image_URL"], use_column_width=True)
                st.markdown("---")
        return

    if search_button:
        corrected_query = utils.correct_text(search_term, all_documents)
        if corrected_query != search_term:
            st.warning(f"Your search terms were corrected to: {corrected_query}")
            search_term = corrected_query

        with st.spinner("Searching..."):
            time.sleep(0.5)  # For showing the spinner (optional)
            start_time = time.time()
            result = utils.search(
                search_term, search_max_num, search_method, search_weights,
                unigram_smoothing=unigram_smoothing, alpha=alpha, lamda=lamda,
            )
            end_time = time.time()
            if not result:
                st.warning("No results found!")
                return

            st.session_state["search_results"] = result
            search_time(start_time, end_time)


            for movie_id, relevance in result:
                info = utils.get_movie_by_id(movie_id, utils.movies_dataset)
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        stars_string = f'<img src="data:image/png;base64,{star_base64}" width="30" height="30">' * int(float(info["rating"]))
                        st.header(info["title"],info["title"],help=info['release_year'])
                        st.markdown(f"""<div style="display: flex; align-items: center;"> 
                        <img src="data:image/png;base64,{imdb_logo_base64}" width="30" height="30">
                        <h4>IMDB Rating: {info['rating']}</h4>{stars_string}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"[Link to movie]({info['URL']})")
                        st.write(f"Relevance Score: {relevance}")
                        st.markdown(
                            f"<b>Summary:</b> {get_summary_with_snippet(info, search_term)}",
                            unsafe_allow_html=True,
                        )
                        

                        st.markdown("**Directors:**")
                        for director in info["directors"]:
                            st.markdown(f"- {director}")

                        st.markdown("**Stars:**")
                        for star in info["stars"]:
                            st.markdown(f"- {star}")

                        st.markdown("**Genres:**")
                        string = ''
                        for genre in info["genres"]:
                            string += f"   <span style='color: rgb(125, 12, 173);margin-right: 15px;'>{genre.upper()}   </span>"
                        st.markdown(
                            string,
                            unsafe_allow_html=True,
                        )

                        st.markdown("**Related movies:**")

                        string = '<div style="display: flex; margin-right: 30px; ">'
                        for movie in info["related_links"][:5]:
                            parts = movie.split('/')
                            for part in parts:
                                if part.startswith('tt'):
                                    related_id = part
                            try:
                                string += f'<a style="margin-right: 15px; text-decoration: none; color: rgb(125, 12, 173);" href="{movie}">{utils.get_movie_by_id(related_id, utils.movies_dataset)["title"]}</a> '
                            except:
                                pass
                            
    
                        string +=  '</div>'
                        
                        st.markdown(string,unsafe_allow_html=True)




                    with col2:
                        st.image(info["Image_URL"], use_column_width=True)
                    st.markdown("---")


# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="IMDB", layout="wide")

    # Custom CSS for page theme and scrollbar
    st.markdown("""
        <style>
            /* Page theme */
            .stApp {

            }

            .image{
                background-image: url("https://sprcdn-assets.sprinklr.com/674/8b955864-7307-4d41-8ded-c194170f5305-2729152590.jpg");
                background-repeat: no-repeat;
                background-size: 100% ;
                background-attachment: fixed;
                padding: 40px;
                margin: -40px;
            }

            /* Scrollbar styles */
            ::-webkit-scrollbar {
                width: 10px;
            }
            ::-webkit-scrollbar-track {
                background: #f0f0f5;
            }
            ::-webkit-scrollbar-thumb {
                background: #888;
                border-radius: 10px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #555;
            }

            /* Card styles */
            .card {
                padding: 20px;
                margin: 20px 0;
                border-radius: 30px;
                background-color: #fff;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.8);
            }

            /* Header styling */
            .header {
                text-align: center;
                
                padding: 20px;
                background-color: rgb(125, 12, 173,0.9);
                color:rgb(187, 181, 188);
                border-radius: 30px;
                margin-bottom: 20px;
                font-family: Arial, sans-serif;
            }

            /* Intro paragraph */
            .intro {
                font-size: 1.2em;
                # background-color:rgba(150, 77, 182, 0.469);
                margin-bottom: 20px;
                text-align: center;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: #555;
            }

            /* Developer badge */
            .badge {
                display: inline-block;
                padding: 10px 15px;
                border-radius: 30px;
                background-color: rgb(150, 87, 177);
                color: #333;
                font-size: 1em;
                margin-top: 20px;
                text-align: center;
                font-family: 'Courier New', Courier, monospace;
                border: 1px solid #DAA520;
            }

            /* Slider label */
            .slider-label {
                font-size: 1.2em;
                color: rgb(127, 35, 167);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin-bottom: 10px;
            }

            /* Button hover effect */
            .stButton > button {
                background-color:rgb(127, 35, 167);
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 20px;
                transition-duration: 0.4s;
            }

            .stButton > button:hover {
                background-color: white;
                color: black;
                border: 2px solid rgb(127, 35, 167);
            }

            /* Text input styling */
            .stTextInput > div > div > input {
                border: 2px solid rgb(127, 35, 167);
                border-radius: 20px;
                padding: 10px;
                transition: border-color 0.3s;
            }

            .stTextInput > div > div > input:focus {
                border-color: rgb(127, 35, 167);
                box-shadow: 0 0 20px rgb(127, 35, 167);
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="image">' +\
        '<div class="header"><h1 style="color: white;">IMDB Search</h1></div>' +\
        '<p class="intro" style="color: white;">Search through the IMDB dataset to find the most relevant movies to your search terms.</p>'+\
        '<div  style="color: white;" class="badge">Developed By: MIR Team at Sharif University & Amirmahdi Meighani</div></div>',
        unsafe_allow_html=True,
    )

    search_term = st.text_input("Search Term")
    with st.expander("Advanced Options"):
        search_max_num = st.number_input("Maximum number of results", min_value=5, max_value=100, value=5, step=5)
        weight_stars = st.slider("Weight of stars in search", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
        weight_genres = st.slider("Weight of genres in search", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
        weight_summary = st.slider("Weight of summary in search", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
        slider_ = st.slider("Select the number of top movies to show", 1, 10, 5)

        search_weights = [weight_stars, weight_genres, weight_summary]
        search_method = st.selectbox("Search method", ("ltn.lnn", "ltc.lnc", "OkapiBM25", "unigram"))

        unigram_smoothing, alpha, lamda = None, None, None
        if search_method == "unigram":
            unigram_smoothing = st.selectbox("Smoothing method", ("naive", "bayes", "mixture"))
            if unigram_smoothing in ["bayes", "mixture"]:
                st.markdown('<div class="slider-label">Alpha</div>', unsafe_allow_html=True)
                alpha = st.slider("", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
            if unigram_smoothing == "mixture":
                st.markdown('<div class="slider-label">Lambda</div>', unsafe_allow_html=True)
                lamda = st.slider("", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    if "search_results" not in st.session_state:
        st.session_state["search_results"] = []

    search_button = st.button("Search!")
    filter_button = st.button("Filter movies by ranking")

    search_handling(
        search_button, search_term, search_max_num, search_weights, search_method,
        unigram_smoothing, alpha, lamda, filter_button, slider_,
    )


if __name__ == "__main__":
    main()
