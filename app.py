#core packages
import re
from unittest import result
import streamlit as st
import streamlit.components.v1 as stc

#eda package
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel


#loading data
def load_data(data):
    df = pd.read_csv(data)
    return df


#vectorize and cosine similarity matrix
def vectorize_text_to_cosine_matrix(data):
    count_vect = CountVectorizer()
    cv_matrix = count_vect.fit_transform(data)
    #get the cosine
    cosine_sim_mat = cosine_similarity(cv_matrix)
    return cosine_sim_mat

#recommender sys
def get_recommendation(title, cosine_sim_mat, df, num_of_rec=5):
    # indicces of the course 
    course_indices = pd.Series(df.index, index=df['course_title']).drop_duplicates()
    # index of course
    idx = course_indices[title]
    #look into the cosine matrix for that index
    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1],reverse=True)
    selected_course_indices = [i[0] for i in sim_scores[1:]]
    selected_course_soores = [i[0] for i in sim_scores[1:]]

    # get the dataframe and title
    result_df = df.iloc[selected_course_indices]
    result_df['similarity_score'] = selected_course_soores
    return result_df[['course_title', 'similarity_score', 'url', 'price', 'num_subscribers']]

#app main starter function
def main():
    st.title("Course Recommendation App")
    menu = ["Home","Recommend","About"]
    choice = st.sidebar.selectbox("Menu", menu)

    #df = load_data("https://raw.githubusercontent.com/Jcharis/Streamlit_DataScience_Apps/master/course_recommendation_sys_app/data/udemy_course_data.csv")
    df = load_data("data/udemy_course_data.csv")

    if choice == "Home":
        st.subheader("Home")
        st.dataframe(df.head(10))
        st.write(df.shape)
    elif choice == "Recommend":
        st.subheader("Recommend")
        num_of_rec = st.sidebar.number_input("Home recs?",4,20,5)
        search_term = st.text_input("Search")

        cosine_sim_mat = vectorize_text_to_cosine_matrix(df['course_title'])

        if st.button("Recommend"):
            if search_term is not None:
                try:
                    result = get_recommendation(search_term, cosine_sim_mat, df, num_of_rec)
                    
                except:
                    result = "Nothing found"
                
                st.write(result)
                #for row in result.iterrows():

    else:
        st.subheader("About")
        
    st.text("Built with love (Streamlit and Pandas)")
        


if __name__ == '__main__':
    main()